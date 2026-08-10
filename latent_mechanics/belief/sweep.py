"""
Reduced-dimension sweep: d = 4, 5, 6, adaptive vs fixed R.

Reports numbers for the user to choose from. It deliberately does not pick a
winner: d, the adaptive-R window and floor, and whether Q should adapt are all
flagged in the brief as decisions for the user, so this script surfaces the
evidence and stops.

Accuracy is reported **relative to the per-family oracle ceiling** from the
geometry report -- the error achieved by the best latent that exists for each
object, fitted offline on all of its data. Absolute error is not comparable
across families (a drawer moves in metres, a door in radians), and more
importantly it conflates "the filter found a bad latent" with "no latent is any
good for this object", which is a distinction the ceiling makes explicit. A
ratio of 1.0 means the filter recovered everything a perfect belief could.

Filter stability is reported alongside, because a filter can look accurate while
being one bad Cholesky from divergence.

    python3.10 -m latent_mechanics.belief.sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
from latent_mechanics.belief.basis import DEFAULT_BASIS, DEFAULT_TABLE, load_or_create
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor, StaticLatentAdaptor
from latent_mechanics.online.config import load_config as load_online_config
from latent_mechanics.online.loop import (
    episode_boundaries,
    episode_stream,
    init_strategies,
    run_online_adaptation,
)

PREDICTOR = DEFAULT_TABLE
DATA = "runs/latent_mechanics/geometry/data_all_families.npz"

# Per-family oracle ceilings, measured in the geometry investigation (Step 6 of
# latent_mechanics/geometry/README.md). Normalised angle error of the best
# latent fitted offline per object.
ORACLE_CEILING = {
    "door": 1.671e-2, "nonlinear_hinge": 1.500e-2, "soft_close": 1.880e-2,
    "drawer": 1.642e-1, "bifold": 2.011e-2, "laptop": 6.147e-2,
}


def _safe(fn, vals) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(fn(vals)) if vals else float("nan")


def _nrmse(err: np.ndarray, st: np.ndarray, nx: np.ndarray) -> float:
    d = nx - st
    scale = max(float(np.sqrt(np.mean(d[:, 0] ** 2))), 1e-12)
    return float(np.sqrt(np.mean(err[:, 0] ** 2)) / scale)


def evaluate(
    label: str, make_adaptor, ds, families, out_rows: list[dict],
    max_objects: int | None = None,
) -> dict:
    errs, ratios, stability = [], [], []
    ids = list(ds.door_ids)[: max_objects or len(ds.door_ids)]
    for did in ids:
        did = int(did)
        stream = episode_stream(ds, did, exclude_near_limit=False)
        if len(stream) < 100:
            continue
        bounds = episode_boundaries(ds, did, exclude_near_limit=False)
        st = np.stack([s for s, _, _ in stream])
        nx = np.stack([n for _, _, n in stream])
        tail = max(1, len(stream) // 4)

        ad = make_adaptor()
        log = run_online_adaptation(ad, stream, door_id=did, boundaries=bounds,
                                    verify_frozen=False)
        e = _nrmse(log.error[-tail:], st[-tail:], nx[-tail:])
        fam = str(families[did])
        ceiling = ORACLE_CEILING.get(fam, np.nan)
        ratio = e / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan

        # Stability: NaNs, filter resets, and how the covariance behaved.
        ex = log.extras
        n_nan = int(np.sum(~np.isfinite(log.loss)))
        resets = float(np.nansum(ex.get("filter_reset", np.zeros(1))))
        p_trace = ex.get("P_trace")
        p_end = float(p_trace[-1]) if p_trace is not None and len(p_trace) else np.nan
        p_max = float(np.nanmax(p_trace)) if p_trace is not None and len(p_trace) else np.nan

        errs.append(e); ratios.append(ratio)
        stability.append({"nan": n_nan, "resets": resets, "p_end": p_end, "p_max": p_max})
        out_rows.append({
            "config": label, "door_id": did, "family": fam,
            "nrmse": e, "oracle_ceiling": ceiling, "ratio_to_ceiling": ratio,
            "n_steps": len(log), "us_per_update": 1e6 * log.seconds_per_update,
            "nan_steps": n_nan, "filter_resets": resets,
            "P_trace_final": p_end, "P_trace_max": p_max,
            "R_trace_final": (float(ex["R_trace"][-1]) if "R_trace" in ex
                              and len(ex["R_trace"]) else np.nan),
            "gain_norm_final": (float(ex["gain_norm"][-1]) if "gain_norm" in ex
                                and len(ex["gain_norm"]) else np.nan),
        })

    finite = [r for r in ratios if np.isfinite(r)]
    return {
        "config": label, "n_objects": len(errs),
        "median_nrmse": float(np.median(errs)) if errs else np.nan,
        "median_ratio_to_ceiling": float(np.median(finite)) if finite else np.nan,
        "frac_within_2x_ceiling": float(np.mean(np.array(finite) <= 2.0)) if finite else np.nan,
        "total_nan_steps": int(sum(s["nan"] for s in stability)),
        "total_filter_resets": float(sum(s["resets"] for s in stability)),
        # Baselines carry no covariance, so these are legitimately absent.
        "median_P_trace_final": _safe(np.nanmedian, [s["p_end"] for s in stability]),
        "max_P_trace": _safe(np.nanmax, [s["p_max"] for s in stability]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="runs/latent_mechanics/belief")
    ap.add_argument("--dims", default="4,5,6")
    ap.add_argument("--objects", type=int, default=30)
    ap.add_argument("--windows", default="20,50,100")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dims = [int(x) for x in args.dims.split(",")]
    windows = [int(x) for x in args.windows.split(",")]

    print("Step 0/2 -- fixed latent basis")
    basis = load_or_create(out / "latent_basis.npz", PREDICTOR, n_components=8)
    print(basis.summary())
    model, table, _, _ = load_checkpoint(PREDICTOR, device="cpu", stage="belief_ukf:sweep")
    model.freeze()
    train_z = table.weight.detach().cpu().numpy().astype(np.float64)
    for d in dims:
        b = basis.truncate(d)
        print(f"    d={d}: reconstruction error of the training latents = "
              f"{np.median(b.reconstruction_error(train_z)):.3f} "
              f"(median ||z||={np.median(np.linalg.norm(train_z - train_z.mean(0), axis=1)):.2f})")

    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=False)
    with np.load(DATA, allow_pickle=False) as a:
        families = np.array([str(x) for x in a["mechanism_family"]])
    init = init_strategies(train_z, 0)["medoid"]
    oc = load_online_config("configs/online_adaptation.yaml").adaptor

    rows: list[dict] = []
    summaries: list[dict] = []

    print(f"\nEvaluating on {min(args.objects, len(ds.door_ids))} unseen objects "
          f"of {len(set(families[list(ds.door_ids)]))} families\n")

    # Reference points, both independent of d.
    summaries.append(evaluate(
        "baseline:no-adaptation",
        lambda: StaticLatentAdaptor(model, init=init), ds, families, rows, args.objects))
    summaries.append(evaluate(
        "baseline:gradient-descent",
        lambda: GradientLatentAdaptor(model, init=init, lr=oc.lr, window=oc.window,
                                      lr_decay=oc.lr_decay,
                                      n_inner_steps=oc.n_inner_steps),
        ds, families, rows, args.objects))

    # d sweep, adaptive vs fixed R.
    for d in dims:
        for kind in ("adaptive", "fixed"):
            cfg = UKFConfig(dim=d, noise_kind=kind)
            summaries.append(evaluate(
                f"ukf:d={d}:R={kind}",
                lambda c=cfg: UKFLatentAdaptor(model, basis, c, init=init,
                                               prior_latents=train_z),
                ds, families, rows, args.objects))

    # Adaptive-R window sensitivity, at the middle d only.
    d_mid = dims[len(dims) // 2]
    for w in windows:
        cfg = UKFConfig(dim=d_mid, noise_kind="adaptive", window=w)
        summaries.append(evaluate(
            f"ukf:d={d_mid}:window={w}",
            lambda c=cfg: UKFLatentAdaptor(model, basis, c, init=init,
                                           prior_latents=train_z),
            ds, families, rows, args.objects))

    hdr = (f"  {'config':30s} {'nrmse':>10} {'x ceiling':>10} {'<=2x':>7} "
           f"{'NaN':>5} {'resets':>7} {'P_end':>10} {'us/upd':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in summaries:
        us = np.median([r["us_per_update"] for r in rows if r["config"] == s["config"]])
        print(f"  {s['config']:30s} {s['median_nrmse']:>10.3e} "
              f"{s['median_ratio_to_ceiling']:>9.2f}x {100*s['frac_within_2x_ceiling']:>6.0f}% "
              f"{s['total_nan_steps']:>5d} {s['total_filter_resets']:>7.0f} "
              f"{s['median_P_trace_final']:>10.2e} {us:>8.0f}")

    print(f"\n  per-family ratio to oracle ceiling (median):")
    fams = list(dict.fromkeys(r["family"] for r in rows))
    print("  " + " " * 30 + "".join(f"{f[:9]:>11}" for f in fams))
    for s in summaries:
        cells = []
        for f in fams:
            v = [r["ratio_to_ceiling"] for r in rows
                 if r["config"] == s["config"] and r["family"] == f]
            cells.append(np.nanmedian(v) if v else np.nan)
        print(f"  {s['config']:30s}" + "".join(f"{c:>10.2f}x" for c in cells))

    with (out / "sweep_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "sweep_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\n  tables -> {out}")


if __name__ == "__main__":
    main()
