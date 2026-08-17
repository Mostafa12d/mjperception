"""Turning stored rollouts into ``Transition`` streams, through an observation model.

This is where the observation model finally gets a seam. The old code had two
separate paths -- ``online.loop.episode_stream`` (clean, no sensor) and
``mismatch.streams.build_door_stream`` (door-only, sensor baked in) -- and no way
to run the mechanism suite through a sensor at all. One function now covers both.

The invariant that must not be lost: the observation model is applied ONCE to the
contiguous ``(T+1,)`` state sequence, then transitions are rebuilt from it.
Corrupting ``state`` and ``next_state`` independently would give two readings of
the same instant and halve the effective noise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from latent_mechanics.dataset import DoorTransitionDataset
from mechanics.observation import IdentityObservation, ObservationModel, apply_to_sequence
from mechanics.types import Transition


def transitions_from_dataset(
    dataset: DoorTransitionDataset,
    object_id: int,
    observation: ObservationModel | None = None,
    *,
    max_episodes: int | None = None,
    exclude_near_limit: bool = True,
    seed: int = 0,
) -> tuple[list[Transition], list[int]]:
    """All transitions for one object, in recording order, episodes concatenated.

    Returns ``(transitions, boundaries)``. ``Transition.truth`` always carries the
    clean next state, so scoring measures the estimator rather than the sensor
    even when ``observation`` corrupts the stream.
    """
    observation = observation or IdentityObservation()
    dt = dataset.dt_model
    out: list[Transition] = []
    boundaries: list[int] = []
    n_eps = 0

    for eid in dataset.episode_ids():
        ep = dataset.episode(int(eid))
        if ep.door_id != object_id:
            continue
        if max_episodes is not None and n_eps >= max_episodes:
            break

        rng = np.random.default_rng(seed * 7919 + object_id * 101 + n_eps)
        obs, next_obs = apply_to_sequence(
            observation, ep.state, ep.next_state, dt, rng)

        keep = ~ep.near_limit if exclude_near_limit else np.ones(len(ep), bool)
        if not keep.any():
            n_eps += 1
            continue

        boundaries.append(len(out))
        for i in np.nonzero(keep)[0]:
            out.append(Transition(
                obs=obs[i], action=ep.action[i], next_obs=next_obs[i],
                truth=ep.next_state[i],
            ))
        n_eps += 1

    if not out:
        raise ValueError(
            f"object {object_id} yielded no transitions from {dataset.path.name} "
            f"[{dataset.split}]")
    return out, boundaries


def as_arrays(transitions: Sequence[Transition]) -> dict[str, np.ndarray]:
    """Stack a stream into arrays, for analyses that want them."""
    return {
        "obs": np.stack([np.asarray(t.obs).reshape(-1) for t in transitions]),
        "action": np.stack([np.asarray(t.action).reshape(-1) for t in transitions]),
        "next_obs": np.stack([np.asarray(t.next_obs).reshape(-1) for t in transitions]),
        "truth": np.stack([np.asarray(t.target).reshape(-1) for t in transitions]),
    }


__all__ = ["transitions_from_dataset", "as_arrays"]
