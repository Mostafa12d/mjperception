"""
Stage 5: does offline mechanical diversity buy online adaptability?

Trains one model per curriculum level on a FIXED instance budget, then evaluates
every model on ONE fixed suite of unseen mechanisms. Architecture, latent
dimension, optimiser and adaptation algorithm are identical throughout; the only
variable is the mechanical diversity of the training mixture.

Per level and per test instance:

  before   normalised error with the latent held at its prior (no adaptation)
  after    normalised error over the final quarter, with online adaptation
  gain     before / after
  steps    interactions to reach 1.5x the run's own final error
  failure  gain < 1, i.e. adaptation made things worse

Run:
    python3.10 -m latent_mechanics.curriculum.study
    python3.10 -m latent_mechanics.curriculum.study --levels 1,4,7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.curriculum.levels import (
    CURRICULUM,
    EVAL_FAMILIES,
    CurriculumConfig,
    curriculum_table,
    split_budget,
)
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.data_gen import build_dataset_npz
from latent_mechanics.mechanisms.rollout import rollout_mechanism
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


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def family_seed(base_seed: int, family: str) -> int:
    """Per-family seed offset that is stable across processes.

    Python's ``hash()`` on ``str`` is salted per interpreter unless
    ``PYTHONHASHSEED`` is pinned, so the obvious ``base_seed + hash(fam)`` makes
    the population draw depend on which process drew it. That is invisible while
    a pickle is cached and silently irreproducible the moment it is not, which is
    exactly the failure mode a seed is supposed to rule out. sha256 of the family
    name is stable across processes, machines and Python versions.
    """
    digest = hashlib.sha256(family.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16) % 10_000


def build_pools(cc: CurriculumConfig, stage1_cfg, cache: Path, verbose=True) -> dict:
    """Simulate a pool of training instances per family, once.

    Levels subsample from these pools, so an instance that appears at two levels
    is the *same* instance with the same trajectories. That removes sampling
    noise from the comparison between levels.
    """
    if cache.exists():
        with open(cache, "rb") as f:
            pools = pickle.load(f)
        if verbose:
            print(f"  loaded training pools from {cache}")
        return pools

    # Pool size = the largest count any level asks of that family.
    need = {}
    for lv in CURRICULUM:
        for fam, n in split_budget(lv, cc.train_instances).items():
            need[fam] = max(need.get(fam, 0), n)

    pools: dict[str, list] = {}
    for fam, n in need.items():
        rng = np.random.default_rng(family_seed(cc.train_seed, fam))
        insts = []
        for k in range(n):
            p = lib.sample_params(fam, rng, mechanism_id=len(insts))
            ep = rollout_mechanism(p, stage1_cfg, cc.episodes_per_train_instance,
                                   cc.episode_seconds, cc.frame_skip,
                                   seed=cc.train_seed)
            if len(ep) > 50:
                insts.append(ep)
        pools[fam] = insts
        if verbose:
            print(f"    pool {fam:16s} {len(insts):3d} instances")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(pools, f)
    return pools


def build_eval_suite(cc: CurriculumConfig, stage1_cfg, cache: Path, verbose=True) -> list:
    """The fixed evaluation suite. Generated once, never regenerated per level."""
    if cache.exists():
        with open(cache, "rb") as f:
            suite = pickle.load(f)
        if verbose:
            print(f"  loaded fixed eval suite from {cache} ({len(suite)} instances)")
        return suite

    suite = []
    for fam in EVAL_FAMILIES:
        rng = np.random.default_rng(family_seed(cc.eval_seed, fam))
        for k in range(cc.eval_instances_per_family):
            p = lib.sample_params(fam, rng, mechanism_id=10_000 + len(suite))
            ep = rollout_mechanism(p, stage1_cfg, cc.episodes_per_eval_instance,
                                   cc.episode_seconds, cc.frame_skip,
                                   seed=cc.eval_seed, episode_offset=500)
            if len(ep) > 50:
                suite.append(ep)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(suite, f)
    if verbose:
        print(f"  built fixed eval suite: {len(suite)} instances")
    return suite


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _nrmse(err: np.ndarray, state: np.ndarray, nxt: np.ndarray, dim: int = 0) -> float:
    d = nxt - state
    scale = max(float(np.sqrt(np.mean(d[:, dim] ** 2))), 1e-12)
    return float(np.sqrt(np.mean(err[:, dim] ** 2)) / scale)


def evaluate_level(
    ckpt: Path, npz: Path, families: np.ndarray, online_cfg, cc: CurriculumConfig,
    with_rls: bool = False, device="cpu",
) -> list[dict]:
    """Adapt on every instance of the fixed suite and score it."""
    model, table, _, _ = load_checkpoint(ckpt, device=device,
                                         stage=f"stage5_curriculum:{ckpt.parent.name}")
    model.freeze()
    init = init_strategies(table.weight.detach().cpu().numpy(), 0)[cc.latent_init]
    ds = DoorTransitionDataset(npz, "heldout_door", exclude_near_limit=False)
    a, r = online_cfg.adaptor, online_cfg.rls

    rows = []
    for did in ds.door_ids:
        did = int(did)
        stream = episode_stream(ds, did, exclude_near_limit=False)
        if len(stream) < 100:
            continue
        bounds = episode_boundaries(ds, did, exclude_near_limit=False)
        st = np.stack([s for s, _, _ in stream])
        nx = np.stack([n for _, _, n in stream])
        tail = max(1, len(stream) // 4)

        static = run_online_adaptation(
            StaticLatentAdaptor(model, init=init, device=device), stream,
            door_id=did, boundaries=bounds, verify_frozen=False)
        adapt = run_online_adaptation(
            GradientLatentAdaptor(
                model, init=init, lr=a.lr, optimizer=a.optimizer,
                n_inner_steps=a.n_inner_steps, window=a.window,
                prior_weight=a.prior_weight, loss_space=a.loss_space,
                max_grad_norm=a.max_grad_norm, lr_decay=a.lr_decay, device=device),
            stream, door_id=did, boundaries=bounds, verify_frozen=False)

        before = _nrmse(static.error, st, nx)
        after = _nrmse(adapt.error[-tail:], st[-tail:], nx[-tail:])
        steps = adapt.steps_to(1.5 * adapt.final_rmse(0), 0, cc.rolling_window)

        row = {
            "door_id": did, "family": str(families[did]),
            "nrmse_before": before, "nrmse_after": after,
            "gain": before / after if after > 0 else np.nan,
            "failed": bool(after >= before),
            "steps_to_converge": steps if steps is not None else -1,
            "n_steps": len(stream),
            "belief_travel": float(np.linalg.norm(adapt.latents[-1] - adapt.latents[0])),
        }
        if with_rls:
            rls = run_online_adaptation(
                RLSAdaptor(dt=ds.dt_model, n_substeps=r.n_substeps, n_params=5,
                           lam=r.lam, delta=r.delta, vel_thresh=r.vel_thresh),
                stream, door_id=did, boundaries=bounds, verify_frozen=False)
            row["nrmse_rls"] = _nrmse(rls.error[-tail:], st[-tail:], nx[-tail:])
        rows.append(row)
    return rows


def summarise_level(rows: list[dict]) -> dict:
    """Level-wide aggregates. Medians, because a few families blow up."""
    g = np.array([r["gain"] for r in rows], float)
    conv = [r["steps_to_converge"] for r in rows if r["steps_to_converge"] >= 0]
    return {
        "n_instances": len(rows),
        "nrmse_before": float(np.median([r["nrmse_before"] for r in rows])),
        "nrmse_after": float(np.median([r["nrmse_after"] for r in rows])),
        "gain_median": float(np.nanmedian(g)),
        "gain_geomean": float(np.exp(np.nanmean(np.log(np.clip(g, 1e-6, None))))),
        "failure_rate": float(np.mean([r["failed"] for r in rows])),
        "steps_median": float(np.median(conv)) if conv else float("nan"),
        "per_family": {
            f: {
                "before": float(np.median([r["nrmse_before"] for r in rows if r["family"] == f])),
                "after": float(np.median([r["nrmse_after"] for r in rows if r["family"] == f])),
                "gain": float(np.nanmedian([r["gain"] for r in rows if r["family"] == f])),
                "failure_rate": float(np.mean([r["failed"] for r in rows if r["family"] == f])),
            }
            for f in dict.fromkeys(r["family"] for r in rows)
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"  table -> {path}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--levels", default=None, help="e.g. 1,4,7")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cc = CurriculumConfig()
    if args.out_dir:
        cc.out_dir = args.out_dir
    if args.epochs:
        cc.epochs = args.epochs
    if args.instances:
        cc.train_instances = args.instances
    levels = [lv for lv in CURRICULUM
              if not args.levels or str(lv.index) in args.levels.split(",")]

    out = Path(cc.out_dir); out.mkdir(parents=True, exist_ok=True)
    stage1_cfg = load_stage1_config("configs/latent_mechanics.yaml")
    online_cfg = load_online_config("configs/online_adaptation.yaml")
    t0 = time.perf_counter()

    print("Stage 5: mechanics prior scaling")
    print(f"  FIXED training budget: {cc.train_instances} instances x "
          f"{cc.episodes_per_train_instance} episodes at every level")
    print(f"  FIXED evaluation suite: {cc.eval_instances_per_family} unseen "
          f"instances of each of {len(EVAL_FAMILIES)} families\n")
    print(curriculum_table())

    print("\nBuilding populations")
    pools = build_pools(cc, stage1_cfg, out / "train_pools.pkl")
    suite = build_eval_suite(cc, stage1_cfg, out / "eval_suite.pkl")

    all_rows: list[dict] = []
    summaries: dict = {}
    rls_ref: dict = {}

    for i, lv in enumerate(levels):
        comp = split_budget(lv, cc.train_instances)
        print(f"\n{'=' * 78}\nLEVEL {lv.index}: {lv.name} -- {lv.description}\n{'=' * 78}")
        print("  composition: " + ", ".join(f"{k}x{v}" for k, v in comp.items()))

        train_pops = []
        for fam, n in comp.items():
            train_pops.extend(pools[fam][:n])
        # The eval suite is held out EXPLICITLY, not by family name. Levels 2-7
        # train on doors while the suite also contains (different, unseen)
        # doors, so a family-based split would silently promote those eval doors
        # into training and the evaluation would stop being held out at all.
        train_fams = list(dict.fromkeys(p.params.family for p in train_pops))

        npz = out / f"data_L{lv.index}.npz"
        build_dataset_npz(train_pops, train_fams, npz, stage1_cfg, cc.frame_skip,
                          heldout_pops=suite)
        n_tr = sum(len(p) for p in train_pops)
        print(f"  training transitions: {n_tr}  (budget held fixed across levels)")

        cfg = load_stage1_config(None)
        cfg.model, cfg.train, cfg.sim = stage1_cfg.model, stage1_cfg.train, stage1_cfg.sim
        cfg.train.epochs = cc.epochs
        cfg.train.run_dir = str(out / "runs")
        cfg.train.run_name = f"L{lv.index}_{lv.name}"
        cfg.sim.exclude_near_limit = False
        ckpt = train_stage1(cfg, data_path=str(npz))

        with np.load(npz, allow_pickle=False) as z:
            fams = np.array([str(x) for x in z["mechanism_family"]])
        rows = evaluate_level(ckpt, npz, fams, online_cfg, cc, with_rls=(i == 0))
        for r in rows:
            r["level"] = lv.index; r["level_name"] = lv.name
            r["n_train_families"] = lv.n_families
        if i == 0:
            rls_ref = {r["door_id"]: r.get("nrmse_rls", np.nan) for r in rows}
        all_rows.extend(rows)

        s = summarise_level(rows)
        summaries[lv.index] = s | {"name": lv.name, "families": list(lv.families),
                                   "composition": comp}
        print(f"\n  before {s['nrmse_before']:.3e}  after {s['nrmse_after']:.3e}  "
              f"gain {s['gain_median']:.2f}x  failures {100*s['failure_rate']:.0f}%  "
              f"converge {s['steps_median']:.0f} steps")
        for f, d in s["per_family"].items():
            print(f"    {f:17s} before {d['before']:.3e} after {d['after']:.3e} "
                  f"gain {d['gain']:5.2f}x  fail {100*d['failure_rate']:3.0f}%")

    write_csv(out / "curriculum_results.csv", all_rows)
    (out / "summary.json").write_text(json.dumps(
        {"config": cc.__dict__, "levels": summaries,
         "rls_reference_median": float(np.nanmedian(list(rls_ref.values())))
         if rls_ref else None}, indent=2, default=str))

    print(f"\n{'=' * 78}\nSCALING SUMMARY\n{'=' * 78}")
    print(f"  {'level':>5} {'fams':>4} {'before':>10} {'after':>10} "
          f"{'gain':>7} {'failures':>9} {'converge':>9}")
    for idx, s in summaries.items():
        print(f"  {'L'+str(idx):>5} {len(s['families']):>4} {s['nrmse_before']:>10.3e} "
              f"{s['nrmse_after']:>10.3e} {s['gain_median']:>6.2f}x "
              f"{100*s['failure_rate']:>8.0f}% {s['steps_median']:>9.0f}")
    if rls_ref:
        print(f"\n  RLS reference on the same suite (level-independent): "
              f"{np.nanmedian(list(rls_ref.values())):.3e}")

    if not args.no_figures:
        from latent_mechanics.curriculum import figures
        figures.run_all(out, summaries, all_rows, cc)

    print(f"\nartefacts -> {out}")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
