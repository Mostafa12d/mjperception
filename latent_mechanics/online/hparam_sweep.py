"""Re-validate the online-adaptation hyperparameters on a given predictor.

``configs/online_adaptation.yaml`` carries ``lr=0.03, window=32, lr_decay=3e-3,
n_inner_steps=1``. Those were measured (see ``online/README.md`` -- constant step
size made adaptation a net loss at 0.95x, and ``window=1`` degraded the belief at
``lr >= 0.03``) but against the **Stage-1 doors-only** predictor,
``runs/latent_mechanics/base/best.pt``. The geometry report's Step 6 and the
belief/UKF branch import those same values and apply them to the **all-families**
predictor, whose latent space has a four-order-of-magnitude scale axis the
doors-only space does not. Nothing had re-checked them there.

The search grid is not committed anywhere, so this reconstructs the selection
criterion rather than the script: prequential one-step error over the final
quarter of each held-out object's stream, normalised per object, expressed as

    gain = static_tail_error / adapted_tail_error

with ``> 1`` meaning adaptation helped. ``StaticLatentAdaptor`` supplies the
no-adaptation control at the same init, so every config is scored against the
same reference. This is the quantity Stage 3, Stage 5 and the geometry report all
report, so a re-tuned value is comparable to the published ones.

Read-only: loads a checkpoint, writes a CSV and a JSON summary, and touches no
config. Nothing here changes what any stage uses.

    python3.10 -m latent_mechanics.online.hparam_sweep --per-family 4
    python3.10 -m latent_mechanics.online.hparam_sweep \
        --checkpoint runs/latent_mechanics/base/best.pt \
        --data data/door_mechanics.npz --out-dir runs/.../doors_only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from latent_mechanics import provenance
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor, StaticLatentAdaptor
from latent_mechanics.online.config import load_config as load_online_config
from latent_mechanics.online.loop import episode_stream, init_strategies, run_online_adaptation

ALL_FAMILIES = "runs/latent_mechanics/geometry/runs/all_families/best.pt"
ALL_FAMILIES_DATA = "runs/latent_mechanics/geometry/data_all_families.npz"

# Centred on the shipped values, one order of magnitude either side.
GRID = {
    "lr": (0.01, 0.03, 0.1),
    "window": (8, 32, 128),
    "lr_decay": (0.0, 3.0e-3, 1.0e-2),
    "n_inner_steps": (1,),
}


def tail_nrmse(err: np.ndarray, state: np.ndarray, nxt: np.ndarray,
               tail: int, dim: int = 0) -> float:
    """Normalised angle error over the last ``tail`` steps of a stream."""
    d = nxt[-tail:] - state[-tail:]
    scale = max(float(np.sqrt(np.mean(d[:, dim] ** 2))), 1e-12)
    return float(np.sqrt(np.mean(err[-tail:, dim] ** 2))) / scale


def objects_to_score(ds: DoorTransitionDataset, families: np.ndarray,
                     per_family: int, min_len: int) -> list[tuple[int, str]]:
    """A balanced sample of held-out objects, so no family dominates the median."""
    by_fam: dict[str, list[int]] = {}
    for did in ds.door_ids:
        did = int(did)
        fam = str(families[did]) if did < len(families) else "unknown"
        by_fam.setdefault(fam, []).append(did)
    out = []
    for fam, ids in sorted(by_fam.items()):
        out.extend((d, fam) for d in ids[:per_family])
    return out


def run(
    checkpoint: str,
    data: str,
    out_dir: Path,
    per_family: int = 4,
    max_steps: int = 1200,
    min_len: int = 200,
    init_name: str = "medoid",
    grid: dict | None = None,
) -> dict:
    grid = grid or GRID
    out_dir.mkdir(parents=True, exist_ok=True)

    model, table, _, _ = load_checkpoint(checkpoint, device="cpu",
                                         stage="hparam_sweep")
    model.freeze()
    train_z = table.weight.detach().cpu().numpy()
    init = init_strategies(train_z, 0)[init_name]

    ds = DoorTransitionDataset(data, "heldout_door", exclude_near_limit=False)
    with np.load(data, allow_pickle=False) as a:
        families = (np.array([str(x) for x in a["mechanism_family"]])
                    if "mechanism_family" in a
                    else np.array(["door"] * (int(a["n_train_doors"])
                                              + int(a["n_heldout_doors"]))))

    picked = objects_to_score(ds, families, per_family, min_len)

    # Cache each object's stream once; every config reuses it.
    streams: list[tuple[int, str, list, np.ndarray, np.ndarray, int]] = []
    for did, fam in picked:
        s = episode_stream(ds, did, exclude_near_limit=False)
        if len(s) < min_len:
            continue
        s = s[:max_steps]
        st = np.stack([x for x, _, _ in s])
        nx = np.stack([x for _, _, x in s])
        streams.append((did, fam, s, st, nx, max(1, len(s) // 4)))

    print(f"  predictor : {checkpoint}")
    print(f"  data      : {data}")
    print(f"  objects   : {len(streams)} held-out, "
          f"{len(set(f for _, f, *_ in streams))} families, "
          f"init={init_name}, <= {max_steps} steps each")

    # No-adaptation control, once per object.
    static: dict[int, float] = {}
    for did, fam, s, st, nx, tail in streams:
        lg = run_online_adaptation(StaticLatentAdaptor(model, init=init), s,
                                   door_id=did, verify_frozen=False)
        static[did] = tail_nrmse(lg.error, st, nx, tail)

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"  grid      : {len(combos)} configs x {len(streams)} objects "
          f"= {len(combos)*len(streams)} runs\n")

    rows: list[dict] = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        cfgd = dict(zip(keys, combo))
        for did, fam, s, st, nx, tail in streams:
            lg = run_online_adaptation(
                GradientLatentAdaptor(model, init=init, **cfgd), s,
                door_id=did, verify_frozen=False)
            adapted = tail_nrmse(lg.error, st, nx, tail)
            rows.append({**cfgd, "door_id": did, "family": fam,
                         "static": static[did], "adapted": adapted,
                         "gain": static[did] / max(adapted, 1e-12)})
        med = float(np.median([r["gain"] for r in rows[-len(streams):]]))
        print(f"  [{i:3d}/{len(combos)}] " +
              " ".join(f"{k}={v}" for k, v in cfgd.items()) +
              f"   median gain {med:.3f}   ({time.time()-t0:.0f}s)")

    # Aggregate per config.
    summary = []
    for combo in combos:
        cfgd = dict(zip(keys, combo))
        sub = [r for r in rows if all(r[k] == v for k, v in cfgd.items())]
        gains = np.array([r["gain"] for r in sub])
        per_fam = {}
        for f in sorted(set(r["family"] for r in sub)):
            per_fam[f] = float(np.median([r["gain"] for r in sub
                                          if r["family"] == f]))
        summary.append({
            **cfgd,
            "median_gain": float(np.median(gains)),
            "mean_gain": float(gains.mean()),
            "worst_gain": float(gains.min()),
            "frac_harmful": float((gains < 1.0).mean()),
            "median_adapted": float(np.median([r["adapted"] for r in sub])),
            "per_family_median_gain": per_fam,
        })

    with open(out_dir / "hparam_sweep_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    oc = load_online_config("configs/online_adaptation.yaml").adaptor
    shipped = {"lr": oc.lr, "window": oc.window, "lr_decay": oc.lr_decay,
               "n_inner_steps": oc.n_inner_steps}
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": provenance.file_sha256(checkpoint),
        "data": data, "init": init_name,
        "n_objects": len(streams), "max_steps": max_steps,
        "grid": {k: list(v) for k, v in grid.items()},
        "shipped_config": shipped,
        "summary": summary,
    }
    (out_dir / "hparam_sweep.json").write_text(json.dumps(result, indent=2))
    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    s = sorted(result["summary"], key=lambda r: -r["median_gain"])
    shipped = result["shipped_config"]
    is_shipped = lambda r: all(abs(float(r[k]) - float(v)) < 1e-12
                               for k, v in shipped.items())

    print("\n" + "=" * 96)
    print(f"Ranked by median gain (static / adapted tail error; >1 means "
          f"adaptation helped)")
    print("=" * 96)
    print(f"{'lr':>6} {'window':>7} {'lr_decay':>9} {'inner':>6} "
          f"{'median':>8} {'mean':>7} {'worst':>7} {'%harmful':>9}  ")
    print("-" * 96)
    for r in s:
        mark = "  <-- SHIPPED" if is_shipped(r) else ""
        print(f"{r['lr']:>6} {r['window']:>7} {r['lr_decay']:>9} "
              f"{r['n_inner_steps']:>6} {r['median_gain']:>8.3f} "
              f"{r['mean_gain']:>7.3f} {r['worst_gain']:>7.3f} "
              f"{100*r['frac_harmful']:>8.0f}%{mark}")

    best = s[0]
    ship = next((r for r in s if is_shipped(r)), None)
    print("\n  best      : " + " ".join(f"{k}={best[k]}" for k in
                                        ("lr", "window", "lr_decay", "n_inner_steps"))
          + f"   median gain {best['median_gain']:.3f}")
    if ship:
        print("  shipped   : " + " ".join(f"{k}={ship[k]}" for k in
                                          ("lr", "window", "lr_decay", "n_inner_steps"))
              + f"   median gain {ship['median_gain']:.3f}"
              f"   (rank {s.index(ship)+1} of {len(s)})")
        print(f"  headroom  : {100*(best['median_gain']/max(ship['median_gain'],1e-12)-1):.1f}% "
              f"median-gain improvement available")
    print("\n  per-family median gain, best vs shipped:")
    fams = sorted(best["per_family_median_gain"])
    print(f"    {'config':22s} " + " ".join(f"{f[:9]:>10s}" for f in fams))
    for lab, r in (("best", best), ("shipped", ship)):
        if r:
            print(f"    {lab:22s} "
                  + " ".join(f"{r['per_family_median_gain'].get(f, float('nan')):>10.3f}"
                             for f in fams))
    print("\n  NOTE: nothing was written to configs/. This is a measurement only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=ALL_FAMILIES)
    ap.add_argument("--data", default=ALL_FAMILIES_DATA)
    ap.add_argument("--out-dir", default="runs/latent_mechanics/geometry/hparam_sweep")
    ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--init", default="medoid",
                    help="latent init; the geometry report uses 'medoid'")
    a = ap.parse_args()
    print("Online-adaptation hyperparameter re-validation\n")
    run(a.checkpoint, a.data, Path(a.out_dir), per_family=a.per_family,
        max_steps=a.max_steps, init_name=a.init)


if __name__ == "__main__":
    main()
