"""
Dataset over ``(door_id, state, action, next_state)`` transitions.

The door id is carried on every sample and is the *only* link between a sample
and its mechanics -- the physical parameters are never fed to the model. During
stage 1 the id indexes the embedding table; in stage 2 there is no id at all and
a latent is optimised directly, which is why nothing downstream of the dataset
depends on ids being contiguous or even present.

Splits are episode-level. A random per-transition split would be meaningless
here: consecutive 20 ms transitions are near-duplicates, so a "validation"
sample drawn that way sits between two training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_mechanics.data_gen import (
    SPLIT_HELDOUT_DOOR,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SPLIT_VAL,
    moving_fraction,
)

SPLIT_CODES = {v: k for k, v in SPLIT_NAMES.items()}


@dataclass
class Episode:
    """One contiguous trajectory, the unit used for multi-step rollout eval."""

    episode_id: int
    door_id: int
    kind: str
    state: np.ndarray  # (T, 2)
    action: np.ndarray  # (T, 1)
    next_state: np.ndarray  # (T, 2)
    t: np.ndarray  # (T,)
    near_limit: np.ndarray  # (T,) bool

    def __len__(self) -> int:
        return len(self.state)

    def true_trajectory(self) -> np.ndarray:
        """States visited, including the final one: (T+1, 2)."""
        return np.concatenate([self.state, self.next_state[-1:]], axis=0)


class DoorTransitionDataset(Dataset):
    """Transitions from one split of a generated ``.npz``.

    Args:
        path: dataset produced by ``latent_mechanics.data_gen``.
        split: ``"train"``, ``"val"``, ``"heldout_door"`` or ``"all"``.
        exclude_near_limit: drop transitions touching a joint limit. Those carry
            a constraint torque that is not part of the action, so they are
            legitimate dynamics but a different regime; excluding them is
            sometimes the cleaner experiment. Off by default.
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        exclude_near_limit: bool = False,
    ) -> None:
        self.path = Path(path)
        self.split = split
        raw = np.load(self.path, allow_pickle=False)

        self.frame_skip = int(raw["frame_skip"])
        self.dt_model = float(raw["dt_model"])
        self.mujoco_dt = float(raw["mujoco_dt"])
        self.n_train_doors = int(raw["n_train_doors"])
        self.n_heldout_doors = int(raw["n_heldout_doors"])
        self.door_params = raw["door_params"]
        self.door_params_columns = [str(c) for c in raw["door_params_columns"]]
        self.config_yaml = str(raw["config_yaml"])

        # Full arrays kept so episode lookup can span splits consistently.
        self._all = {
            k: raw[k]
            for k in ("state", "action", "next_state", "t", "door_id",
                      "episode_id", "split", "near_limit", "step_in_episode")
        }
        self._episode_ptr = raw["episode_ptr"]
        self._episode_kind = [str(k) for k in raw["episode_kind"]]
        self._episode_door = raw["episode_door_id"]
        self._episode_split = raw["episode_split"]

        mask = self._split_mask(split)
        if exclude_near_limit:
            mask &= ~self._all["near_limit"]
        self.index = np.nonzero(mask)[0]
        if len(self.index) == 0:
            raise ValueError(f"split '{split}' selected 0 transitions from {self.path}")

        self.state = torch.from_numpy(self._all["state"][self.index])
        self.action = torch.from_numpy(self._all["action"][self.index])
        self.next_state = torch.from_numpy(self._all["next_state"][self.index])
        self.door_id = torch.from_numpy(self._all["door_id"][self.index].astype(np.int64))

    # -- indexing ---------------------------------------------------------
    def _split_mask(self, split: str) -> np.ndarray:
        if split == "all":
            return np.ones(len(self._all["state"]), dtype=bool)
        if split not in SPLIT_CODES:
            raise ValueError(
                f"unknown split '{split}'; choose from {sorted(SPLIT_CODES)} or 'all'"
            )
        return self._all["split"] == SPLIT_CODES[split]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "door_id": self.door_id[i],
            "state": self.state[i],
            "action": self.action[i],
            "next_state": self.next_state[i],
        }

    # -- metadata ---------------------------------------------------------
    @property
    def num_embedding_rows(self) -> int:
        """Rows the embedding table needs: one per *training* door.

        Held-out doors deliberately get no row. Their ids continue past this
        value, so an accidental lookup fails loudly instead of silently reusing
        another door's latent.
        """
        return self.n_train_doors

    @property
    def door_ids(self) -> np.ndarray:
        return np.unique(self._all["door_id"][self.index])

    def params_for_door(self, door_id: int) -> dict[str, float]:
        return dict(zip(self.door_params_columns, self.door_params[door_id]))

    def param_matrix(self, door_ids: np.ndarray | None = None) -> np.ndarray:
        ids = self.door_ids if door_ids is None else np.asarray(door_ids)
        return self.door_params[ids]

    # -- normalisation ----------------------------------------------------
    def norm_stats(self) -> dict[str, torch.Tensor]:
        """Per-dimension mean/std over this split, for the model's buffers.

        Always compute these on the *training* split and reuse them elsewhere;
        recomputing per split would make metrics incomparable.
        """
        state = self.state.numpy()
        action = self.action.numpy()
        delta = self.next_state.numpy() - state
        as_t = lambda a: torch.tensor(a, dtype=torch.float32)
        return {
            "state_mean": as_t(state.mean(0)),
            "state_std": as_t(state.std(0)),
            "action_mean": as_t(action.mean(0)),
            "action_std": as_t(action.std(0)),
            "delta_mean": as_t(delta.mean(0)),
            "delta_std": as_t(delta.std(0)),
        }

    # -- episodes ---------------------------------------------------------
    def episode_ids(self) -> np.ndarray:
        """Episode ids belonging to this split."""
        if self.split == "all":
            return np.arange(len(self._episode_kind))
        return np.nonzero(self._episode_split == SPLIT_CODES[self.split])[0]

    def episode(self, episode_id: int) -> Episode:
        """Full contiguous episode, ignoring any ``exclude_near_limit`` filter.

        Rollouts need every consecutive step, so filtered-out transitions cannot
        be skipped here without breaking the chain.
        """
        lo, hi = self._episode_ptr[episode_id], self._episode_ptr[episode_id + 1]
        sl = slice(lo, hi)
        return Episode(
            episode_id=int(episode_id),
            door_id=int(self._episode_door[episode_id]),
            kind=self._episode_kind[episode_id],
            state=self._all["state"][sl],
            action=self._all["action"][sl],
            next_state=self._all["next_state"][sl],
            t=self._all["t"][sl],
            near_limit=self._all["near_limit"][sl],
        )

    def episodes(self, limit: int | None = None, seed: int | None = None) -> Iterator[Episode]:
        ids = self.episode_ids()
        if limit is not None and limit < len(ids):
            rng = np.random.default_rng(seed if seed is not None else 0)
            ids = rng.choice(ids, size=limit, replace=False)
            ids.sort()
        for eid in ids:
            yield self.episode(int(eid))

    def summary(self) -> str:
        thd = self.state[:, 1].numpy()
        return (
            f"{self.path.name} [{self.split}]: {len(self)} transitions, "
            f"{len(self.door_ids)} doors, {len(self.episode_ids())} episodes, "
            f"dt_model={self.dt_model:.3f}s, "
            f"{100 * moving_fraction(thd):.0f}% moving"
        )


def load_splits(
    path: str | Path, exclude_near_limit: bool = False
) -> dict[str, DoorTransitionDataset]:
    """Convenience loader for the three splits that always exist."""
    out = {}
    for split in ("train", "val", "heldout_door"):
        try:
            out[split] = DoorTransitionDataset(path, split, exclude_near_limit)
        except ValueError:
            continue  # e.g. n_heldout_doors = 0
    return out


__all__ = [
    "DoorTransitionDataset",
    "Episode",
    "load_splits",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_HELDOUT_DOOR",
]
