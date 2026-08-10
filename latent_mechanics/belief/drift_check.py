"""
Does the chosen adaptive-R window survive time-varying dynamics?

The d/window sweep in ``sweep.py`` ran entirely on *stationary* objects. Stage 3
built a ``ParameterDrift`` perturbation, but it was never carried into the
Stage-4 mechanism families -- ``perturbations_for`` attaches Stribeck,
position-dependent friction and soft-close damping, all time-invariant -- so no
object in that sweep had mechanics that changed during the episode.

That is exactly the gap that matters for the window choice. A long innovation
window estimates R from further into the past, and under drift the past is a
worse description of the present. Combined with a small fixed Q, the failure
mode is a filter that becomes confident about mechanics that have since moved on
and then refuses to update. A stationary benchmark cannot see this.

This script re-rolls held-out objects with ``ParameterDrift`` layered on top of
their own family physics and compares windows 20 / 50 / 100 against the
gradient-descent module and the no-adaptation control.

Nothing here is a "ceiling" comparison: under drift there is no single best
latent, so the oracle ceilings from the geometry report do not apply. Errors are
reported directly, and relative to the no-adaptation control at the *same* drift
level, which is the only fair reference.

    python3.10 -m latent_mechanics.belief.drift_check
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

import run_door_dynamics_validation as dyn
from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
from latent_mechanics.belief.basis import DEFAULT_TABLE, load_or_create
from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.data_gen import transitions_from_log
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.rollout import limit_margin_for, simulate_mechanism
from latent_mechanics.mismatch.perturbations import ParameterDrift
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor, StaticLatentAdaptor
from latent_mechanics.online.config import load_config as load_online_config
from latent_mechanics.online.loop import init_strategies, run_online_adaptation

EVAL_SUITE = Path("runs/latent_mechanics/curriculum/eval_suite.pkl")
# Stage-3 severity levels for friction drift, in units of 1/s.
DRIFT_RATES = (0.0, 0.15, 0.40)


def roll_with_drift(params, cfg, drift_rate: float, n_episodes: int,
                    episode_seconds: float, frame_skip: int, seed: int = 999):
    """One object's stream, with its own family physics PLUS parameter drift."""
    n_steps = int(round(episode_seconds / dyn.DT))
    S, A, N = [], [], []
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed * 7717 + (params.mechanism_id + 1) * 1009 + ep)
        profile = lib.scaled_profile(cfg.excitation, rng, n_steps, frame_skip, params)
        model = lib.build_model(params)
        perts = list(lib.perturbations_for(params))
        if drift_rate > 0:
            perts.append(ParameterDrift(friction_rate=drift_rate, mode="linear"))
        log = simulate_mechanism(profile.as_fn(), model, n_steps, perts)
        _, _, jid = lib.joint_info(model)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        # This object's own joint range, not the door's -- see
        # data_gen.transitions_from_log.
        tr = transitions_from_log(log.as_stage1_dict(), frame_skip,
                                  joint_range=(lo, hi),
                                  limit_margin=limit_margin_for(lo, hi))
        keep = ~tr["near_limit"]
        if keep.any():
            S.append(tr["state"][keep]); A.append(tr["action"][keep])
            N.append(tr["next_state"][keep])
    if not S:
        return None
    s, a, n = np.concatenate(S), np.concatenate(A), np.concatenate(N)
    return list(zip(s, a, n)), s, n


def _nrmse(err, st, nx) -> float:
    d = nx - st
    scale = max(float(np.sqrt(np.mean(d[:, 0] ** 2))), 1e-12)
    return float(np.sqrt(np.mean(err[:, 0] ** 2)) / scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objects", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--windows", default="20,50,100")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--out-dir", default="runs/latent_mechanics/belief")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in args.windows.split(",")]
    stage1_cfg = load_stage1_config("configs/latent_mechanics.yaml")
    oc = load_online_config("configs/online_adaptation.yaml").adaptor

    model, table, _, _ = load_checkpoint(DEFAULT_TABLE, device="cpu")
    model.freeze()
    train_z = table.weight.detach().cpu().numpy().astype(np.float64)
    basis = load_or_create(out / "latent_basis.npz", DEFAULT_TABLE, n_components=8)
    init = init_strategies(train_z, 0)["medoid"]

    with open(EVAL_SUITE, "rb") as f:
        suite = pickle.load(f)
    # Spread the sample across families rather than taking the first N.
    by_fam: dict[str, list] = {}
    for p in suite:
        by_fam.setdefault(p.params.family, []).append(p.params)
    per = max(1, args.objects // max(len(by_fam), 1))
    chosen = [q for v in by_fam.values() for q in v[:per]]

    print("Drift check: does the adaptive-R window survive time-varying mechanics?")
    print(f"  {len(chosen)} held-out objects across {len(by_fam)} families")
    print(f"  d={args.dim}, drift rates {DRIFT_RATES} (1/s, Stage-3 levels)\n")

    methods = {"no-adaptation": None, "gradient-descent": None}
    for w in windows:
        methods[f"ukf:w={w}"] = w

    rows: list[dict] = []
    for rate in DRIFT_RATES:
        for params in chosen:
            got = roll_with_drift(params, stage1_cfg, rate, args.episodes,
                                  stage1_cfg.sim.episode_seconds,
                                  stage1_cfg.sim.frame_skip)
            if got is None:
                continue
            stream, st, nx = got
            if len(stream) < 200:
                continue
            tail = max(1, len(stream) // 4)

            for name, w in methods.items():
                if name == "no-adaptation":
                    ad = StaticLatentAdaptor(model, init=init)
                elif name == "gradient-descent":
                    ad = GradientLatentAdaptor(model, init=init, lr=oc.lr,
                                               window=oc.window, lr_decay=oc.lr_decay,
                                               n_inner_steps=oc.n_inner_steps)
                else:
                    ad = UKFLatentAdaptor(
                        model, basis,
                        UKFConfig(dim=args.dim, noise_kind="adaptive", window=w),
                        init=init, prior_latents=train_z)
                log = run_online_adaptation(ad, stream, door_id=params.mechanism_id,
                                            verify_frozen=False)
                # Late-vs-mid error is the tracking signal: under drift a filter
                # that has stopped listening gets worse as the episode proceeds.
                mid = slice(len(stream) // 2 - tail // 2, len(stream) // 2 + tail // 2)
                rows.append({
                    "drift_rate": rate, "family": params.family,
                    "object": params.mechanism_id, "method": name,
                    "nrmse_final": _nrmse(log.error[-tail:], st[-tail:], nx[-tail:]),
                    "nrmse_mid": _nrmse(log.error[mid], st[mid], nx[mid]),
                    "belief_travel": float(np.linalg.norm(log.latents[-1] - log.latents[0])),
                })

    names = list(methods)
    print(f"  {'drift':>7} " + "".join(f"{m:>19}" for m in names))
    print("  " + "-" * (7 + 19 * len(names)))
    base: dict[float, float] = {}
    for rate in DRIFT_RATES:
        cells = []
        for m in names:
            v = [r["nrmse_final"] for r in rows
                 if r["method"] == m and r["drift_rate"] == rate]
            cells.append(float(np.median(v)) if v else np.nan)
        base[rate] = cells[0]
        print(f"  {rate:>7.2f} " + "".join(f"{c:>19.3e}" for c in cells))

    print(f"\n  relative to the no-adaptation control at the SAME drift level "
          f"(<1 = adaptation helps)")
    print(f"  {'drift':>7} " + "".join(f"{m:>19}" for m in names))
    for rate in DRIFT_RATES:
        cells = []
        for m in names:
            v = [r["nrmse_final"] for r in rows
                 if r["method"] == m and r["drift_rate"] == rate]
            cells.append((float(np.median(v)) / base[rate]) if v and base[rate] > 0 else np.nan)
        print(f"  {rate:>7.2f} " + "".join(f"{c:>18.2f}x" for c in cells))

    print(f"\n  degradation within the episode (final quarter / middle quarter; "
          f">1 = losing track as drift accumulates)")
    print(f"  {'drift':>7} " + "".join(f"{m:>19}" for m in names))
    for rate in DRIFT_RATES:
        cells = []
        for m in names:
            v = [r["nrmse_final"] / max(r["nrmse_mid"], 1e-12) for r in rows
                 if r["method"] == m and r["drift_rate"] == rate]
            cells.append(float(np.median(v)) if v else np.nan)
        print(f"  {rate:>7.2f} " + "".join(f"{c:>18.2f}x" for c in cells))

    with (out / "drift_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n  table -> {out / 'drift_check.csv'}")


if __name__ == "__main__":
    main()
