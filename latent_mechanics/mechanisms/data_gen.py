"""Building multi-mechanism datasets.

The suite is simulated once and cached; each experiment repacks the same episodes
into a Stage-1-format ``.npz`` with a different split, so experiments see
identical trajectories and Stage-1 code runs unmodified.

Family and physical parameters are stored as analysis labels only; the model
still sees just ``(state, action, door_id)``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import yaml

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import ExperimentConfig
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.rollout import MechanismEpisodes, rollout_mechanism

SPLIT_TRAIN, SPLIT_VAL, SPLIT_HELDOUT = 0, 1, 2


def generate_suite(
    cfg: ExperimentConfig,
    families: list[str],
    n_per_family: int,
    n_episodes: int,
    episode_seconds: float,
    frame_skip: int,
    seed: int = 0,
    cache: str | Path | None = None,
    verbose: bool = True,
) -> list[MechanismEpisodes]:
    """Simulate every mechanism instance once. Cached, because it is the slow step."""
    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as f:
            pops = pickle.load(f)
        if verbose:
            print(f"  loaded {len(pops)} mechanism instances from {cache}")
        return pops

    pops: list[MechanismEpisodes] = []
    params = lib.sample_population(families, n_per_family, seed)
    for p in params:
        ep = rollout_mechanism(p, cfg, n_episodes, episode_seconds, frame_skip, seed=seed)
        pops.append(ep)
        if verbose and len(pops) % max(1, len(params) // 12) == 0:
            print(f"    {len(pops):3d}/{len(params)}  {p.summary()}  n={len(ep)}")

    pops = [p for p in pops if len(p) > 50]
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(pops, f)
        if verbose:
            print(f"  cached -> {cache}")
    return pops


def build_dataset_npz(
    pops: list[MechanismEpisodes],
    train_families: list[str],
    path: str | Path,
    cfg: ExperimentConfig,
    frame_skip: int,
    val_episodes: int = 1,
    heldout_pops: list[MechanismEpisodes] | None = None,
) -> Path:
    """Repack episodes into the Stage-1 ``.npz`` layout for a given split.

    Training instances take embedding rows ``0..n_train-1``; held-out ids run
    above that, so a stray lookup fails loudly.

    Passing ``heldout_pops`` splits explicitly and is REQUIRED whenever a held-out
    instance belongs to a family also being trained on. Omitting it partitions by
    family, which is valid only when the two family sets are disjoint.
    """
    if heldout_pops is not None:
        train_pops = list(pops)
        held_pops = list(heldout_pops)
    else:
        train_pops = [p for p in pops if p.params.family in train_families]
        held_pops = [p for p in pops if p.params.family not in train_families]
    ordered = train_pops + held_pops
    n_train = len(train_pops)

    S, A, N, T, DID, EID, SPL, NEAR, STEP = [], [], [], [], [], [], [], [], []
    ep_ptr, ep_door, ep_split, ep_kind = [0], [], [], []
    gt_rows, families, param_rows = [], [], []
    ep_counter = 0

    for new_id, pop in enumerate(ordered):
        is_train = new_id < n_train
        gt_rows.append([pop.gt.get(c, 0.0) for c in lib.GT_COLUMNS])
        families.append(pop.params.family)
        param_rows.append(pop.params)

        uniq = np.unique(pop.episode_id)
        # never let validation consume every episode, or a one-episode instance
        # contributes nothing to training
        n_val = min(val_episodes, max(0, len(uniq) - 1))
        val_ids = set(uniq[-n_val:].tolist()) if is_train and n_val else set()
        for e in uniq:
            m = pop.episode_id == e
            n = int(m.sum())
            if n == 0:
                continue
            split = (SPLIT_HELDOUT if not is_train
                     else (SPLIT_VAL if e in val_ids else SPLIT_TRAIN))
            S.append(pop.state[m]); A.append(pop.action[m]); N.append(pop.next_state[m])
            T.append(np.arange(n, dtype=np.float32) * frame_skip * dyn.DT)
            DID.append(np.full(n, new_id, dtype=np.int32))
            EID.append(np.full(n, ep_counter, dtype=np.int32))
            SPL.append(np.full(n, split, dtype=np.uint8))
            NEAR.append(np.zeros(n, dtype=bool))  # already filtered at rollout
            STEP.append(np.arange(n, dtype=np.int32))
            ep_ptr.append(ep_ptr[-1] + n)
            ep_door.append(new_id); ep_split.append(split)
            ep_kind.append(pop.params.family)
            ep_counter += 1

    pack = {
        "state": np.concatenate(S), "action": np.concatenate(A),
        "next_state": np.concatenate(N), "t": np.concatenate(T),
        "door_id": np.concatenate(DID), "episode_id": np.concatenate(EID),
        "split": np.concatenate(SPL), "near_limit": np.concatenate(NEAR),
        "step_in_episode": np.concatenate(STEP),
        "episode_ptr": np.array(ep_ptr, dtype=np.int64),
        "episode_door_id": np.array(ep_door, dtype=np.int32),
        "episode_split": np.array(ep_split, dtype=np.uint8),
        "episode_kind": np.array(ep_kind),
        "door_params": np.array(gt_rows, dtype=np.float64),
        "door_params_columns": np.array(lib.GT_COLUMNS),
        "door_model_paths": np.array([p.family for p in param_rows]),
        "mechanism_family": np.array(families),
        "n_train_doors": np.int64(n_train),
        "n_heldout_doors": np.int64(len(held_pops)),
        "frame_skip": np.int64(frame_skip),
        "mujoco_dt": np.float64(dyn.DT),
        "dt_model": np.float64(dyn.DT * frame_skip),
        "config_yaml": np.array(yaml.safe_dump(cfg.to_dict(), sort_keys=False)),
        "episode_stats_json": np.array("{}"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    return path


def family_of_doors(npz_path: str | Path) -> np.ndarray:
    """Family label per door id, for analysis colouring."""
    with np.load(npz_path, allow_pickle=False) as z:
        return np.array([str(x) for x in z["mechanism_family"]])


def dataset_summary(npz_path: str | Path) -> str:
    with np.load(npz_path, allow_pickle=False) as z:
        fam = np.array([str(x) for x in z["mechanism_family"]])
        n_train = int(z["n_train_doors"])
        lines = [f"  {int(len(z['state']))} transitions, {len(fam)} instances "
                 f"({n_train} trainable, {len(fam) - n_train} held out)"]
        for f in dict.fromkeys(fam):
            ids = np.nonzero(fam == f)[0]
            tr = int((ids < n_train).sum())
            lines.append(f"    {f:16s} {len(ids):3d} instances "
                         f"({tr} train / {len(ids) - tr} held out)")
    return "\n".join(lines)


__all__ = ["generate_suite", "build_dataset_npz", "family_of_doors", "dataset_summary"]
