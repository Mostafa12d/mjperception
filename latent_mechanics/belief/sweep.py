"""Reduced-dimension sweep: d = 4, 5, 6, adaptive vs fixed R.

Accuracy is reported relative to the per-object oracle ceiling, since absolute
error is not comparable across families. Ratio 1.0 = the filter recovered
everything a perfect belief could. Stability is reported alongside.

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

# Family medians over each object's FULL stream. Reference only -- not used to
# score, because the methods are scored on the tail. Use ``object_ceilings``.
ORACLE_CEILING_LEGACY = {
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


def _spread_inits(train_z: np.ndarray, k: int = 4, seed: int = 0) -> list[np.ndarray]:
    """``k`` well-separated training latents, as extra oracle restarts. A single
    medoid start stalls badly on drawers and inflates the ceiling."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=min(k, len(train_z)), n_init=10, random_state=seed).fit(train_z)
    out = []
    for c in km.cluster_centers_:
        out.append(train_z[int(np.argmin(np.linalg.norm(train_z - c, axis=1)))])
    return out


def object_ceilings(model, ds, max_objects: int | None = None,
                    init: np.ndarray | None = None, steps: int = 1200,
                    train_z: np.ndarray | None = None) -> dict[int, float]:
    """Per-object oracle ceiling on the same tail window the methods are scored over.

    ``steps`` is deliberately high: 400 Adam steps does not converge and an
    under-converged oracle overstates the ceiling, flattering every method.
    """
    from latent_mechanics.geometry.analyses import fit_oracle_latent

    extra = _spread_inits(train_z) if train_z is not None else None
    out: dict[int, float] = {}
    ids = list(ds.door_ids)[: max_objects or len(ds.door_ids)]
    door_ids = ds.door_id.numpy()
    for did in ids:
        did = int(did)
        stream = episode_stream(ds, did, exclude_near_limit=False)
        if len(stream) < 100:
            continue
        st = np.stack([s for s, _, _ in stream])
        ac = np.stack([a for _, a, _ in stream])
        nx = np.stack([n for _, _, n in stream])
        tail = max(1, len(stream) // 4)
        T = lambda v: torch.as_tensor(v[-tail:], dtype=torch.float32)

        # fitted ON the scoring window, minimising the SCORED quantity
        z = fit_oracle_latent(model, T(st), T(ac), T(nx), init, steps=steps,
                              extra_inits=extra, objective="angle")
        with torch.no_grad():
            pred = model(T(st), T(ac),
                         torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)).numpy()
        out[did] = _nrmse(pred - nx[-tail:], st[-tail:], nx[-tail:])
    return out


def evaluate(
    label: str, make_adaptor, ds, families, out_rows: list[dict],
    max_objects: int | None = None, ceilings: dict[int, float] | None = None,
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
        ceiling = (ceilings or {}).get(did, np.nan)
        ratio = e / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan

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
        # absent for baselines, which carry no covariance
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
          f"of {len(set(families[list(ds.door_ids)]))} families")
    print("  fitting per-object oracle ceilings on the scoring window...")
    ceilings = object_ceilings(model, ds, args.objects, init, train_z=train_z)
    print(f"  {len(ceilings)} ceilings, median {np.median(list(ceilings.values())):.3e}\n")

    def ev(label, factory):
        summaries.append(evaluate(label, factory, ds, families, rows,
                                  args.objects, ceilings))

    # reference points, both independent of d
    ev("baseline:no-adaptation", lambda: StaticLatentAdaptor(model, init=init))
    ev("baseline:gradient-descent",
       lambda: GradientLatentAdaptor(model, init=init, lr=oc.lr, window=oc.window,
                                     lr_decay=oc.lr_decay,
                                     n_inner_steps=oc.n_inner_steps))

    # d sweep; all three noise models kept so the fix stays next to the defect
    for d in dims:
        for kind in ("residual", "adaptive", "fixed"):
            cfg = UKFConfig(dim=d, noise_kind=kind)
            ev(f"ukf:d={d}:R={kind}",
               lambda c=cfg: UKFLatentAdaptor(model, basis, c, init=init,
                                              prior_latents=train_z))

    d_mid = dims[len(dims) // 2]

    # transform settings, held at one noise model so the two stay separable
    for label, kw in (("alpha=0.3,iter=1 (old transform)", dict(alpha=0.3, n_iterations=1)),
                      ("alpha=1.0,iter=1", dict(alpha=1.0, n_iterations=1)),
                      ("alpha=1.0,iter=3 (chosen)", dict(alpha=1.0, n_iterations=3)),
                      ("alpha=1.0,iter=6", dict(alpha=1.0, n_iterations=6))):
        cfg = UKFConfig(dim=d_mid, noise_kind="residual", **kw)
        ev(f"ukf:d={d_mid}:{label}",
           lambda c=cfg: UKFLatentAdaptor(model, basis, c, init=init,
                                          prior_latents=train_z))

    # R-window sensitivity, at the middle d
    for w in windows:
        cfg = UKFConfig(dim=d_mid, noise_kind="residual", window=w)
        ev(f"ukf:d={d_mid}:window={w}",
           lambda c=cfg: UKFLatentAdaptor(model, basis, c, init=init,
                                          prior_latents=train_z))

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
