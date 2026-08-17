"""Dataset generation for the latent-mechanics dynamics model.

Episodes are rolled out by ``run_door_dynamics_validation.simulate`` rather than
a reimplemented integrator. This samples doors and excitation, calls simulate,
and slices the 500 Hz log into model-rate transitions tagged by door.

    python3.10 -m latent_mechanics.data_gen --config configs/latent_mechanics.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics import door_sampler
from latent_mechanics.config import ExperimentConfig, load_config
from latent_mechanics.door_sampler import DoorParams
from latent_mechanics.excitation import TorqueProfile, sample_profile

# split codes stored per transition
SPLIT_TRAIN, SPLIT_VAL, SPLIT_HELDOUT_DOOR = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_HELDOUT_DOOR: "heldout_door"}

# Door-XML defaults. Other mechanisms must pass their own range; see
# ``transitions_from_log``.
JOINT_RANGE = (-0.17, 2.09)
LIMIT_MARGIN = 0.05

# "Moving" relative to the joint's own speed scale, so m/s and rad/s compare
MOVING_FRAC_OF_P95 = 0.02


def moving_fraction(velocity: np.ndarray) -> float:
    """Fraction of samples moving, relative to this signal's own p95 speed."""
    v = np.abs(np.asarray(velocity, dtype=float))
    if v.size == 0:
        return 0.0
    return float((v > MOVING_FRAC_OF_P95 * max(float(np.percentile(v, 95)), 1e-9)).mean())


@contextlib.contextmanager
def episode_length(seconds: float):
    """Temporarily override ``dyn.T_END``/``N_STEPS``, which ``simulate`` reads from
    its module namespace. Always restored."""
    old_t_end, old_n_steps = dyn.T_END, dyn.N_STEPS
    dyn.T_END = float(seconds)
    dyn.N_STEPS = int(round(seconds / dyn.DT))
    try:
        yield dyn.N_STEPS
    finally:
        dyn.T_END, dyn.N_STEPS = old_t_end, old_n_steps


def transitions_from_log(
    log: dict,
    frame_skip: int,
    tau_key: str = "tau_ft",
    joint_range: tuple[float, float] = JOINT_RANGE,
    limit_margin: float = LIMIT_MARGIN,
) -> dict[str, np.ndarray]:
    """Slice a 500 Hz episode log into model-rate transitions.

    ``simulate`` logs the state AFTER step i, and ``tau[i]`` covers
    ``[i*dt, (i+1)*dt)``, so starting at ``j = K-1`` and striding by ``K`` makes
    each transition span exactly one zero-order-hold block.

    ``joint_range`` defaults to the door XML's limits and MUST be overridden for
    other mechanisms, whose travel is on a different scale entirely.
    """
    theta, theta_dot = log["theta"], log["theta_dot"]
    tau = log[tau_key]
    n = len(theta)
    K = frame_skip

    j = np.arange(K - 1, n - K, K)
    if j.size == 0:
        raise ValueError(f"episode too short ({n} steps) for frame_skip={K}")

    state = np.stack([theta[j], theta_dot[j]], axis=1)
    next_state = np.stack([theta[j + K], theta_dot[j + K]], axis=1)

    # Averaged over the window: the logged hinge torque wobbles ~1e-5 N*m inside
    # a hold because simulate reconstructs it from one-step-stale kinematics.
    # The check below still fires if a profile is genuinely not a ZOH.
    window = np.stack([tau[j + 1 + k] for k in range(K)], axis=1)
    action = window.mean(axis=1, keepdims=True)
    spread = np.max(np.abs(window - action), axis=1)
    tol = 1e-4 * np.maximum(np.abs(action[:, 0]), 1.0)
    if np.any(spread > tol):
        worst = int(np.argmax(spread - tol))
        raise ValueError(
            f"torque is not constant within a model timestep (max spread "
            f"{spread[worst]:.3e} N*m on a {action[worst, 0]:.3f} N*m action); the "
            "excitation profile must be a zero-order hold on the frame_skip grid"
        )

    lo, hi = joint_range
    at_limit = lambda th: (th < lo + limit_margin) | (th > hi - limit_margin)
    near_limit = at_limit(state[:, 0]) | at_limit(next_state[:, 0])

    return {
        "state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "next_state": next_state.astype(np.float32),
        "t": (log["t"][j] + dyn.DT).astype(np.float32),
        "step_in_episode": np.arange(len(j), dtype=np.int32),
        "near_limit": near_limit,
    }


def _run_episode(
    params: DoorParams,
    model,
    profile: TorqueProfile,
    frame_skip: int,
    seconds: float,
) -> tuple[dict[str, np.ndarray], dict]:
    with episode_length(seconds):
        log = dyn.simulate(profile.as_fn(), model=model)
    tr = transitions_from_log(log, frame_skip)
    stats = {
        "kind": profile.kind,
        "bias": float(profile.meta["bias"]),
        "theta_min": float(log["theta"].min()),
        "theta_max": float(log["theta"].max()),
        "thetadot_absmax": float(np.abs(log["theta_dot"]).max()),
        "frac_moving": moving_fraction(log["theta_dot"]),
        "frac_near_limit": float(tr["near_limit"].mean()),
        "max_contacts": int(log["ncon"].max()),
    }
    return tr, stats


def generate_dataset(cfg: ExperimentConfig, verbose: bool = True) -> dict:
    """Simulate the whole door population and return the packed dataset dict."""
    sim, doors_cfg = cfg.sim, cfg.doors
    n_steps = int(round(sim.episode_seconds / dyn.DT))
    if sim.val_episodes_per_door >= sim.episodes_per_door:
        raise ValueError("val_episodes_per_door must be < episodes_per_door")

    train_doors, heldout_doors = door_sampler.sample_door_population(
        doors_cfg, sim.seed
    )
    all_doors = train_doors + heldout_doors
    rng = np.random.default_rng(sim.seed + 1)

    chunks: list[dict[str, np.ndarray]] = []
    door_ids: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    episode_ptr = [0]
    episode_kind: list[str] = []
    episode_door: list[int] = []
    episode_split: list[int] = []
    gt_rows: list[dict[str, float]] = []
    ep_stats: list[dict] = []

    t0 = time.time()
    n_total_ep = len(all_doors) * sim.episodes_per_door
    ep_counter = 0

    for params in all_doors:
        model = door_sampler.build_model(params)
        gt_rows.append(door_sampler.ground_truth(model, params))
        is_heldout = params.door_id >= doors_cfg.n_train_doors

        for ep in range(sim.episodes_per_door):
            profile = sample_profile(
                cfg.excitation, rng, n_steps, sim.frame_skip, params.frictionloss
            )
            tr, stats = _run_episode(
                params, model, profile, sim.frame_skip, sim.episode_seconds
            )
            n = len(tr["state"])

            if is_heldout:
                split = SPLIT_HELDOUT_DOOR
            else:
                # split by episode; adjacent transitions are near-duplicates
                split = (
                    SPLIT_VAL if ep >= sim.episodes_per_door - sim.val_episodes_per_door
                    else SPLIT_TRAIN
                )

            chunks.append(tr)
            door_ids.append(np.full(n, params.door_id, dtype=np.int32))
            episode_ids.append(np.full(n, ep_counter, dtype=np.int32))
            splits.append(np.full(n, split, dtype=np.uint8))
            episode_ptr.append(episode_ptr[-1] + n)
            episode_kind.append(profile.kind)
            episode_door.append(params.door_id)
            episode_split.append(split)
            ep_stats.append(stats)
            ep_counter += 1

        if verbose:
            print(
                f"  {params.summary()}  "
                f"{'HELD OUT' if is_heldout else 'train'}  "
                f"[{ep_counter}/{n_total_ep} episodes]"
            )

    pack = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    pack["door_id"] = np.concatenate(door_ids)
    pack["episode_id"] = np.concatenate(episode_ids)
    pack["split"] = np.concatenate(splits)
    pack["episode_ptr"] = np.array(episode_ptr, dtype=np.int64)
    pack["episode_door_id"] = np.array(episode_door, dtype=np.int32)
    pack["episode_split"] = np.array(episode_split, dtype=np.uint8)
    pack["episode_kind"] = np.array(episode_kind)
    pack["door_params"] = door_sampler.params_table(gt_rows)
    pack["door_params_columns"] = np.array(door_sampler.PARAM_TABLE_COLUMNS)
    pack["door_model_paths"] = np.array([d.model_path for d in all_doors])
    pack["n_train_doors"] = np.int64(doors_cfg.n_train_doors)
    pack["n_heldout_doors"] = np.int64(doors_cfg.n_heldout_doors)
    pack["frame_skip"] = np.int64(sim.frame_skip)
    pack["mujoco_dt"] = np.float64(dyn.DT)
    pack["dt_model"] = np.float64(dyn.DT * sim.frame_skip)
    pack["config_yaml"] = np.array(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    pack["episode_stats_json"] = np.array(json.dumps(ep_stats))

    if verbose:
        _print_summary(pack, time.time() - t0)
    return pack


def _print_summary(pack: dict, elapsed: float) -> None:
    n = len(pack["state"])
    print(f"\nGenerated {n} transitions from {len(pack['episode_kind'])} episodes "
          f"in {elapsed:.1f}s")
    for code, name in SPLIT_NAMES.items():
        m = pack["split"] == code
        if m.any():
            print(f"  {name:13s}: {int(m.sum()):7d} transitions, "
                  f"{len(np.unique(pack['door_id'][m])):3d} doors")
    print(f"  model dt      : {float(pack['dt_model']):.3f} s "
          f"({1.0 / float(pack['dt_model']):.0f} Hz, frame_skip="
          f"{int(pack['frame_skip'])})")
    th, thd = pack["state"][:, 0], pack["state"][:, 1]
    d = pack["next_state"] - pack["state"]
    print(f"  theta         : [{th.min():+.3f}, {th.max():+.3f}] rad")
    print(f"  theta_dot     : [{thd.min():+.3f}, {thd.max():+.3f}] rad/s")
    print(f"  action (tau)  : [{pack['action'].min():+.2f}, "
          f"{pack['action'].max():+.2f}] N*m")
    print(f"  |delta theta| : mean {np.abs(d[:, 0]).mean():.5f} rad  "
          f"max {np.abs(d[:, 0]).max():.5f}")
    print(f"  |delta thdot| : mean {np.abs(d[:, 1]).mean():.5f} rad/s  "
          f"max {np.abs(d[:, 1]).max():.5f}")
    print(f"  near joint limit: {100 * pack['near_limit'].mean():.1f}% of transitions")
    print(f"  door moving     : {100 * moving_fraction(thd):.1f}% of transitions "
          f"(relative to p95 speed)")


def save_dataset(pack: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    print(f"Saved dataset -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/latent_mechanics.yaml")
    ap.add_argument("--out", default=None, help="override sim.out_path")
    ap.add_argument("--seed", type=int, default=None, help="override sim.seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg.sim.out_path = args.out
    if args.seed is not None:
        cfg.sim.seed = args.seed

    print(f"Generating door-mechanics dataset (config: {args.config})")
    print(f"  {cfg.doors.n_train_doors} training doors + "
          f"{cfg.doors.n_heldout_doors} held-out doors, "
          f"{cfg.sim.episodes_per_door} episodes each\n")
    pack = generate_dataset(cfg)
    save_dataset(pack, cfg.sim.out_path)


if __name__ == "__main__":
    main()
