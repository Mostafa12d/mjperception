"""
Stage 4: cross-mechanism generalisation.

Does the mechanics embedding represent *doors*, or *interaction mechanics*?

Three experiments, each training the frozen architecture on a different mixture
of mechanism families and then running unchanged Stage-2 online adaptation on
families it has never seen:

  1. train on doors            -> adapt to unseen doors, drawers, laptops
  2. train on doors + drawers  -> adapt to laptops (does diversity help?)
  3. leave-one-family-out      -> adapt to the held-out family, for all six

Only the environment and the training mixture change. The dynamics predictor,
the latent representation and the adaptation algorithm are imported from earlier
stages and used as-is.

Run:
    python3.10 -m latent_mechanics.mechanisms.study
    python3.10 -m latent_mechanics.mechanisms.study --only 1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.data_gen import (
    build_dataset_npz,
    dataset_summary,
    family_of_doors,
    generate_suite,
)
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor, StaticLatentAdaptor
from latent_mechanics.online.config import load_config as load_online_config
from latent_mechanics.online.loop import (
    episode_boundaries,
    episode_stream,
    init_strategies,
    run_online_adaptation,
)
from latent_mechanics.online.rls_adaptor import RLSAdaptor
from latent_mechanics.train import train as train_stage1

METHODS = ("no-adaptation", "latent-gd", "rls-5p")


def _nrmse(err: np.ndarray, state: np.ndarray, nxt: np.ndarray) -> tuple[float, float]:
    """Error normalised by true motion, per dimension.

    Absolutely essential here: a drawer moves 0.5 m and a door 2.3 rad, so raw
    RMSE cannot be compared across families at all. Normalised error is the
    fraction of actual motion left unexplained; 1.0 means no better than
    predicting that nothing changes.
    """
    d = nxt - state
    scale = np.maximum(np.sqrt(np.mean(d**2, axis=0)), 1e-12)
    return (float(np.sqrt(np.mean(err[:, 0] ** 2)) / scale[0]),
            float(np.sqrt(np.mean(err[:, 1] ** 2)) / scale[1]))


def adapt_on_heldout(
    ckpt: Path, npz: Path, families: np.ndarray, online_cfg, device="cpu",
    init_name: str = "medoid",
) -> list[dict]:
    """Run every method on every held-out instance of a trained checkpoint."""
    model, table, stage1_cfg, extra = load_checkpoint(ckpt, device=device)
    model.freeze()
    train_latents = table.weight.detach().cpu().numpy()
    init = init_strategies(train_latents, 0)[init_name]
    ds = DoorTransitionDataset(npz, "heldout_door", exclude_near_limit=False)
    dt = ds.dt_model
    a, r = online_cfg.adaptor, online_cfg.rls

    rows = []
    for did in ds.door_ids:
        did = int(did)
        stream = episode_stream(ds, did, exclude_near_limit=False)
        bounds = episode_boundaries(ds, did, exclude_near_limit=False)
        if len(stream) < 100:
            continue
        st = np.stack([s for s, _, _ in stream])
        nx = np.stack([n for _, _, n in stream])

        for name in METHODS:
            if name == "no-adaptation":
                ad = StaticLatentAdaptor(model, init=init, device=device)
            elif name == "latent-gd":
                ad = GradientLatentAdaptor(
                    model, init=init, lr=a.lr, optimizer=a.optimizer,
                    n_inner_steps=a.n_inner_steps, window=a.window,
                    prior_weight=a.prior_weight, loss_space=a.loss_space,
                    max_grad_norm=a.max_grad_norm, lr_decay=a.lr_decay, device=device)
            else:
                ad = RLSAdaptor(dt=dt, n_substeps=r.n_substeps, n_params=5,
                                lam=r.lam, delta=r.delta, vel_thresh=r.vel_thresh)
            log = run_online_adaptation(ad, stream, door_id=did, boundaries=bounds,
                                        verify_frozen=False)
            tail = max(1, len(log) // 4)
            n_all, _ = _nrmse(log.error, st, nx)
            n_tail, v_tail = _nrmse(log.error[-tail:], st[-tail:], nx[-tail:])
            rows.append({
                "door_id": did, "family": str(families[did]), "method": name,
                "nrmse_all": n_all, "nrmse_final": n_tail, "vel_nrmse_final": v_tail,
                "n_steps": len(log),
                "belief_travel": float(np.linalg.norm(log.latents[-1] - log.latents[0])),
                **{k: ds.params_for_door(did)[k]
                   for k in ("inertia", "frictionloss", "damping", "stiffness",
                             "is_prismatic")},
            })
    return rows


def train_variant(
    name: str, pops, train_families: list[str], stage1_cfg, out: Path,
    frame_skip: int, epochs: int,
) -> tuple[Path, Path]:
    """Build the split, then train the frozen architecture on it."""
    npz = out / f"data_{name}.npz"
    build_dataset_npz(pops, train_families, npz, stage1_cfg, frame_skip)
    print(dataset_summary(npz))

    cfg = load_stage1_config(None)
    cfg.model = stage1_cfg.model
    cfg.train = stage1_cfg.train
    cfg.sim = stage1_cfg.sim
    cfg.train.epochs = epochs
    cfg.train.run_dir = str(out / "runs")
    cfg.train.run_name = name
    cfg.sim.exclude_near_limit = False  # already filtered during rollout
    ckpt = train_stage1(cfg, data_path=str(npz))
    return ckpt, npz


def summarise(rows: list[dict], title: str) -> dict:
    """Per-family table of adaptation quality."""
    fams = list(dict.fromkeys(r["family"] for r in rows))
    print(f"\n  {title}")
    print(f"  {'family':16s} {'n':>3} " + "".join(f"{m:>16}" for m in METHODS)
          + f"{'adapt gain':>12}")
    out = {}
    for f in fams:
        cells = []
        for m in METHODS:
            v = [r["nrmse_final"] for r in rows if r["family"] == f and r["method"] == m]
            cells.append(float(np.median(v)) if v else float("nan"))
        n = len({r["door_id"] for r in rows if r["family"] == f})
        gain = cells[0] / cells[1] if cells[1] > 0 else float("nan")
        out[f] = dict(zip(METHODS, cells)) | {"adapt_gain": gain, "n": n}
        print(f"  {f:16s} {n:>3d} " + "".join(f"{c:>16.3e}" for c in cells)
              + f"{gain:>11.2f}x")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"  table -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="runs/latent_mechanics/mechanisms")
    ap.add_argument("--only", default="1,2,3")
    ap.add_argument("--instances", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-analysis", action="store_true")
    args = ap.parse_args()

    which = {s.strip() for s in args.only.split(",")}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stage1_cfg = load_stage1_config("configs/latent_mechanics.yaml")
    online_cfg = load_online_config("configs/online_adaptation.yaml")
    frame_skip = stage1_cfg.sim.frame_skip
    t0 = time.perf_counter()

    print("Stage 4: cross-mechanism generalisation")
    print(f"  families: {lib.FAMILY_ORDER}")
    print(f"  {args.instances} instances/family x {args.episodes} episodes\n")
    pops = generate_suite(
        stage1_cfg, lib.FAMILY_ORDER, args.instances, args.episodes,
        stage1_cfg.sim.episode_seconds, frame_skip, seed=0,
        cache=out / "suite_cache.pkl",
    )
    print(f"  {len(pops)} usable instances\n")

    results: dict = {}
    all_rows: list[dict] = []

    def run_variant(name, train_families, title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        ckpt, npz = train_variant(name, pops, train_families, stage1_cfg, out,
                                  frame_skip, args.epochs)
        fams = family_of_doors(npz)
        rows = adapt_on_heldout(ckpt, npz, fams, online_cfg)
        for r in rows:
            r["variant"] = name
            r["train_families"] = "+".join(train_families)
        all_rows.extend(rows)
        results[name] = summarise(rows, f"held-out adaptation ({name})")
        return ckpt, npz

    if "1" in which:
        run_variant("exp1_doors_only", ["door"],
                    "EXPERIMENT 1  train on doors only")
    if "2" in which:
        run_variant("exp2_doors_drawers", ["door", "drawer"],
                    "EXPERIMENT 2  train on doors + drawers")
    if "3" in which:
        print(f"\n{'=' * 78}\nEXPERIMENT 3  leave-one-family-out\n{'=' * 78}")
        for held in lib.FAMILY_ORDER:
            train_families = [f for f in lib.FAMILY_ORDER if f != held]
            run_variant(f"exp3_no_{held}", train_families,
                        f"  hold out: {held}")

    write_csv(out / "adaptation_results.csv", all_rows)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))

    if not args.no_analysis and all_rows:
        from latent_mechanics.mechanisms import analysis
        analysis.run_all(out, results, all_rows)

    print(f"\nartefacts -> {out}")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
