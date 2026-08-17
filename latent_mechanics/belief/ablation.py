"""Ablation of the two measurement-path defects, on the 60 held-out objects.

Both were invisible to prediction error but destroyed the latent, so this reports
prediction alongside whether the latent still decodes the physical parameters.
The defects: an unscented transform invalid over the prior (fixed by non-negative
weights, then the iterated update), and adaptive R collapsing onto the
uninformative channel (fixed by the residual form and a matrix floor).

    python3.10 -m latent_mechanics.belief.ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
from latent_mechanics.belief.basis import DEFAULT_TABLE, load_or_create
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.geometry import analyses as an
from latent_mechanics.mechanisms.analysis import (family_separability,
                                                  mechanics_readout)
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor
from latent_mechanics.online.loop import (episode_boundaries, episode_stream,
                                          init_strategies, run_online_adaptation)

DATA = "runs/latent_mechanics/geometry/data_all_families.npz"
OUT = Path("runs/latent_mechanics/belief/ablation")
BASIS = "runs/latent_mechanics/belief/latent_basis.npz"
PARAMS = ("inertia", "frictionloss", "stiffness", "damping")
FAMILIES = ["door", "nonlinear_hinge", "soft_close", "drawer", "bifold", "laptop"]

# one change per row, cumulative
LADDER = [
    ("old (as measured)", dict(alpha=0.3, noise_kind="adaptive", n_iterations=1)),
    ("+ non-neg weights", dict(alpha=1.0, noise_kind="adaptive", n_iterations=1)),
    ("+ residual R, matrix floor", dict(alpha=1.0, noise_kind="residual", n_iterations=1)),
    ("+ iterated update (FIXED)", dict(alpha=1.0, noise_kind="residual", n_iterations=3)),
]


def nrmse(model, z, s, a, ns) -> float:
    """Error of a frozen latent, normalised by the object's own motion."""
    with torch.no_grad():
        e = model(s, a, torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)) - ns
    scale = torch.sqrt(((ns - s)[:, 0] ** 2).mean()).clamp_min(1e-12)
    return float(torch.sqrt((e[:, 0] ** 2).mean()) / scale)


def run(n_objects: int | None = None, oracle_steps: int = 1500,
        device: str = "cpu") -> dict:
    model, table, _, _ = load_checkpoint(DEFAULT_TABLE, device=device,
                                         stage="belief:ablation")
    model.freeze()
    z_train = table.weight.detach().cpu().numpy().astype(np.float64)
    basis = load_or_create(BASIS, DEFAULT_TABLE, n_components=8)
    init = init_strategies(z_train, 0)["medoid"]

    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=False)
    with np.load(DATA, allow_pickle=False) as a:
        fam_all = np.array([str(x) for x in a["mechanism_family"]])
        gt_all = a["door_params"]
        cols = [str(c) for c in a["door_params_columns"]]

    ids = [int(d) for d in ds.door_ids][: n_objects or len(ds.door_ids)]
    methods = [n for n, _ in LADDER] + ["gradient descent", "oracle (offline)", "prior (no filter)"]
    store: dict[str, list] = {m: [] for m in methods}
    fams, gts = [], []

    for k, did in enumerate(ids):
        stream = episode_stream(ds, did, exclude_near_limit=False)
        if len(stream) < 200:
            continue
        bounds = episode_boundaries(ds, did, exclude_near_limit=False)
        i = np.nonzero(ds.door_id.numpy() == did)[0]
        s, a, ns = ds.state[i], ds.action[i], ds.next_state[i]
        fams.append(str(fam_all[did]))
        gts.append(gt_all[did])

        def record(name, z, log, secs):
            ex = getattr(log, "extras", {}) or {}
            g = np.asarray(ex.get("gain_norm", [np.nan]), float)
            me = np.asarray(ex.get("R_min_eig", [np.nan]), float)
            Pt = np.asarray(ex.get("P_trace", [np.nan]), float)
            store[name].append(dict(
                door_id=did, family=str(fam_all[did]), z=np.asarray(z, float),
                err=nrmse(model, z, s, a, ns),
                travel=float(np.linalg.norm(np.asarray(z, float) - init)),
                gain_med=float(np.nanmedian(g)), gain_max=float(np.nanmax(g)),
                frac_floored=float(np.mean(me[np.isfinite(me)] <= 1.001e-6))
                if np.isfinite(me).any() else np.nan,
                P_final=float(Pt[-1]) if np.isfinite(Pt).any() else np.nan,
                us_per_step=1e6 * secs / max(len(stream), 1)))

        for name, kw in LADDER:
            ad = UKFLatentAdaptor(model, basis, UKFConfig(**kw), init=init,
                                  prior_latents=z_train, device=device)
            t0 = time.perf_counter()
            log = run_online_adaptation(ad, stream, door_id=did,
                                        boundaries=bounds, verify_frozen=False)
            record(name, ad.latent, log, time.perf_counter() - t0)

        ad = GradientLatentAdaptor(model, init=init, device=device)
        t0 = time.perf_counter()
        log = run_online_adaptation(ad, stream, door_id=did, boundaries=bounds,
                                    verify_frozen=False)
        record("gradient descent", ad.latent, log, time.perf_counter() - t0)

        # restart from every online estimate too, so the oracle is a strict upper
        # bound; objective="angle" matches what nrmse scores
        zo = an.fit_oracle_latent(
            model, s, a, ns, init, steps=oracle_steps, objective="angle",
            extra_inits=[store[m][-1]["z"] for m in
                         ("+ iterated update (FIXED)", "gradient descent")])
        record("oracle (offline)", zo, None, 0.0)
        record("prior (no filter)", init, None, 0.0)

        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(ids)} objects")

    return dict(store=store, fams=np.array(fams), gts=np.array(gts), cols=cols,
                methods=methods, init=init, z_train=z_train, basis=basis)


def summarise(res: dict) -> list[dict]:
    fams, gts, cols = res["fams"], res["gts"], res["cols"]
    targets = {"inertia": np.log10(np.maximum(gts[:, cols.index("inertia")], 1e-9))}
    for p in PARAMS[1:]:
        targets[p] = gts[:, cols.index(p)]

    rows = []
    for m in res["methods"]:
        recs = res["store"][m]
        Z = np.stack([r["z"] for r in recs])
        row = {"method": m, "n": len(recs),
               "err": float(np.median([r["err"] for r in recs])),
               "travel": float(np.median([r["travel"] for r in recs])),
               "us_per_step": float(np.median([r["us_per_step"] for r in recs])),
               "gain_med": float(np.nanmedian([r["gain_med"] for r in recs])),
               "gain_max": float(np.nanmax([r["gain_max"] for r in recs])),
               "frac_floored": float(np.nanmedian([r["frac_floored"] for r in recs])),
               "fam_sep": family_separability(Z, fams)["accuracy"]}
        for p in PARAMS:
            row[f"r2_{p}"] = mechanics_readout(Z, targets[p])
        row["r2_mean"] = float(np.mean([row[f"r2_{p}"] for p in PARAMS]))
        # paired ratio to each object's own oracle, so families are commensurable
        oracle = {r["door_id"]: r["err"] for r in res["store"]["oracle (offline)"]}
        ratios = np.array([r["err"] / oracle[r["door_id"]] for r in recs])
        row["ratio"] = float(np.median(ratios))
        row["frac_at_or_below_oracle"] = float(np.mean(ratios <= 1.0))
        for f in FAMILIES:
            sel = [r["err"] for r in recs if r["family"] == f]
            row[f"err_{f}"] = float(np.median(sel)) if sel else np.nan
            sr = ratios[np.array([r["family"] == f for r in recs])]
            row[f"ratio_{f}"] = float(np.median(sr)) if len(sr) else np.nan
        rows.append(row)
    return rows


def figure(rows: list[dict], res: dict, out: Path) -> Path:
    fixed = "+ iterated update (FIXED)"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ladder = [r for r in rows if r["method"] in dict(LADDER)]
    ref = {r["method"]: r for r in rows}

    # (a) the ablation: prediction is flat, physics is not
    ax = axes[0]
    x = np.arange(len(ladder))
    ax.bar(x - 0.2, [r["r2_mean"] for r in ladder], 0.4, label="mean probe $R^2$",
           color="#1f77b4", edgecolor="k", linewidth=0.4)
    ax.bar(x + 0.2, [r["fam_sep"] for r in ladder], 0.4, label="family separability",
           color="#9467bd", edgecolor="k", linewidth=0.4)
    ax.axhline(ref["oracle (offline)"]["r2_mean"], color="#1f77b4", ls=":", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    short = {"old (as measured)": "old\n(as measured)",
             "+ non-neg weights": "+ non-neg\nweights",
             "+ residual R, matrix floor": "+ residual R\n+ matrix floor",
             "+ iterated update (FIXED)": "+ iterated\nupdate"}
    ax.set_xticks(x)
    ax.set_xticklabels([short[r["method"]] for r in ladder], fontsize=7.5)
    ax.set_ylabel("score")
    ax.set_title("(a) what the fixes recover\n(dotted = offline oracle)")
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, [r["err"] for r in ladder], "o-", color="0.25", lw=1.6, ms=5)
    ax2.set_ylabel("prediction error (grey)", color="0.25")
    ax2.grid(False)

    # (b) per-parameter probes
    ax = axes[1]
    keys = ["prior (no filter)", "old (as measured)", fixed, "oracle (offline)"]
    colors = ["0.6", "#d62728", "#1f77b4", "#2ca02c"]
    x = np.arange(len(PARAMS)); w = 0.2
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.bar(x + (i - 1.5) * w, [ref[k][f"r2_{p}"] for p in PARAMS], w,
               label=k, color=c, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(PARAMS, fontsize=8)
    ax.set_ylabel("leave-one-out probe $R^2$")
    ax.set_title("(b) which physical parameters survive")
    ax.legend(fontsize=7.5)

    # (c) per-family prediction, as a paired ratio to each object's own oracle
    ax = axes[2]
    keys = ["prior (no filter)", "gradient descent", "old (as measured)", fixed]
    colors = ["0.6", "#ff7f0e", "#d62728", "#1f77b4"]
    x = np.arange(len(FAMILIES)); w = 0.2
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.bar(x + (i - 1.5) * w, [ref[k][f"ratio_{f}"] for f in FAMILIES], w,
               label=k, color=c, edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.5)
    ax.text(0.01, 1.0, "offline oracle", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="#2ca02c", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FAMILIES], fontsize=7.5)
    ax.set_ylabel("error / that object's own oracle (log)")
    ax.set_title("(c) prediction vs the achievable ceiling")
    ax.legend(fontsize=7)

    fig.suptitle("Fixing the two measurement-path defects: 60 unseen objects, 6 families",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"  figure -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objects", type=int, default=None)
    # 400 steps does not converge, and an under-converged oracle is not a ceiling
    ap.add_argument("--oracle-steps", type=int, default=1500)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Ablating the measurement-path defects on the held-out objects")
    res = run(args.objects, args.oracle_steps)
    rows = summarise(res)
    n = rows[0]["n"]

    print("\n" + "=" * 104)
    print(f"MEASUREMENT-PATH ABLATION -- {n} unseen objects, 6 families, one change per row")
    print("=" * 104)
    print(f"  {'':30} {'pred':>7} {'|z-init|':>9} {'probe R2 (mean)':>16} "
          f"{'fam sep':>8} {'med |K|':>8} {'us/step':>8}")
    for r in rows:
        star = "  <--" if r["method"] == "+ iterated update (FIXED)" else ""
        print(f"  {r['method']:30} {r['err']:7.4f} {r['travel']:9.3f} "
              f"{r['r2_mean']:16.3f} {r['fam_sep']:8.3f} "
              f"{r['gain_med']:8.2f} {r['us_per_step']:8.0f}{star}")

    print(f"\n  per-parameter probe R2")
    print(f"  {'':30}" + "".join(f"{p:>14}" for p in PARAMS))
    for r in rows:
        print(f"  {r['method']:30}" + "".join(f"{r[f'r2_{p}']:14.3f}" for p in PARAMS))

    print(f"\n  filter health")
    print(f"  {'':30} {'% steps R floored':>18} {'max |K|':>10} {'iterations':>11}")
    for r in rows[:len(LADDER)]:
        it = dict(LADDER)[r["method"]]["n_iterations"]
        print(f"  {r['method']:30} {100 * r['frac_floored']:17.0f}% "
              f"{r['gain_max']:10.1f} {it:11d}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "ablation.json").write_text(json.dumps(rows, indent=2, default=str))
    figure(rows, res, out / "ablation.png")
    np.savez_compressed(
        out / "latents.npz", fams=res["fams"], gts=res["gts"],
        cols=np.array(res["cols"]), init=res["init"],
        **{f"z_{i}": np.stack([r["z"] for r in res["store"][m]])
           for i, m in enumerate(res["methods"])},
        methods=np.array(res["methods"]))
    print(f"  table   -> {out / 'ablation.csv'}")


if __name__ == "__main__":
    main()
