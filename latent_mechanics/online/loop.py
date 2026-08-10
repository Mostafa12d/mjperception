"""
The online adaptation driver, plus latent initialisation strategies.

``run_online_adaptation`` is deliberately agnostic to what it is driving: it
takes anything implementing ``OnlineAdaptor`` and a stream of transitions, and
applies the same protocol to all of them --

    for each transition:
        prediction = adaptor.predict(state, action)   # belief BEFORE this data
        adaptor.observe(state, action, next_state)    # belief revised

This is *prequential* evaluation: every error reported is a one-step-ahead
prediction on data the estimator had not yet seen. It is the only protocol under
which the learned adaptor and RLS can be compared without one of them scoring
itself on its own training data, and it is also what a robot actually
experiences.

Transitions are never shuffled and never revisited in bulk. A whole-trajectory
fit would be Stage 1 again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from latent_mechanics.dataset import DoorTransitionDataset, Episode
from latent_mechanics.online.adaptor import AdaptorStep, OnlineAdaptor

Transition = tuple[np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def episode_stream(
    dataset: DoorTransitionDataset,
    door_id: int,
    max_episodes: int | None = None,
    exclude_near_limit: bool = True,
) -> list[Transition]:
    """All transitions for one door, in recording order.

    Episodes are concatenated. Each transition is scored independently, so the
    episode boundaries do not corrupt anything -- but the door does jump back to
    closed at each boundary, which is visible as a small transient in the error
    curves and is worth knowing about when reading them.

    ``exclude_near_limit`` matches the stage-1 training filter: joint-limit
    contact adds a constraint torque that is not part of the action, so neither
    estimator can be expected to predict it.
    """
    out: list[Transition] = []
    n_eps = 0
    for eid in dataset.episode_ids():
        ep = dataset.episode(int(eid))
        if ep.door_id != door_id:
            continue
        if max_episodes is not None and n_eps >= max_episodes:
            break
        keep = ~ep.near_limit if exclude_near_limit else np.ones(len(ep), bool)
        for s, a, ns in zip(ep.state[keep], ep.action[keep], ep.next_state[keep]):
            out.append((s, a, ns))
        n_eps += 1
    return out


def episode_boundaries(
    dataset: DoorTransitionDataset,
    door_id: int,
    max_episodes: int | None = None,
    exclude_near_limit: bool = True,
) -> list[int]:
    """Indices in the concatenated stream where a new episode starts."""
    bounds, total, n_eps = [], 0, 0
    for eid in dataset.episode_ids():
        ep = dataset.episode(int(eid))
        if ep.door_id != door_id:
            continue
        if max_episodes is not None and n_eps >= max_episodes:
            break
        bounds.append(total)
        keep = ~ep.near_limit if exclude_near_limit else np.ones(len(ep), bool)
        total += int(keep.sum())
        n_eps += 1
    return bounds


# ---------------------------------------------------------------------------
# Initialisation strategies (Experiment 2)
# ---------------------------------------------------------------------------

def init_strategies(train_latents: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Candidate starting points for an unseen door's latent.

    ``zero`` and ``mean`` are near each other but not identical: the trained
    table is initialised around the origin and weight-decayed toward it, so its
    centroid has small but nonzero norm.

    ``medoid`` is the extra one, and it is motivated by a measurement rather
    than symmetry: the trained latents sit on a shell whose centre contains no
    door, so the centroid is a point the network was never evaluated at, while
    the medoid is a real door that happens to be central.
    """
    rng = np.random.default_rng(seed)
    mean = train_latents.mean(axis=0)
    medoid = train_latents[int(np.argmin(np.linalg.norm(train_latents - mean, axis=1)))]
    return {
        "zero": np.zeros(train_latents.shape[1], dtype=np.float32),
        "random_trained": train_latents[int(rng.integers(len(train_latents)))].copy(),
        "mean": mean.astype(np.float32),
        "medoid": medoid.copy(),
    }


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

@dataclass
class AdaptationLog:
    """Everything recorded during one adaptation run."""

    name: str
    door_id: int
    error: np.ndarray  # (T, 2) signed one-step-ahead prediction error
    loss: np.ndarray  # (T,)
    latents: np.ndarray  # (T, d) belief after each step
    update_seconds: np.ndarray  # (T,)
    boundaries: list[int] = field(default_factory=list)
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    init_name: str = ""

    def __len__(self) -> int:
        return len(self.loss)

    # -- summary metrics ---------------------------------------------------
    def rmse(self, dim: int = 0, first: int | None = None, last: int | None = None) -> float:
        e = self.error[first:last, dim]
        return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")

    def rolling_rmse(self, dim: int = 0, window: int = 200) -> np.ndarray:
        """Rolling RMSE of the prequential error -- the learning curve."""
        sq = self.error[:, dim] ** 2
        w = min(window, len(sq))
        if w < 1:
            return np.array([])
        c = np.concatenate([[0.0], np.cumsum(sq)])
        idx = np.arange(len(sq))
        lo = np.maximum(0, idx - w + 1)
        return np.sqrt((c[idx + 1] - c[lo]) / (idx + 1 - lo))

    def final_rmse(self, dim: int = 0, frac: float = 0.25) -> float:
        """RMSE over the last ``frac`` of the stream -- the converged accuracy."""
        n = max(1, int(len(self) * frac))
        return self.rmse(dim, first=len(self) - n)

    def steps_to(self, threshold: float, dim: int = 0, window: int = 200,
                 hold: int = 100) -> int | None:
        """First step whose rolling RMSE drops below ``threshold`` and stays
        below it for ``hold`` consecutive steps. ``None`` if never."""
        r = self.rolling_rmse(dim, window)
        below = r < threshold
        if not below.any():
            return None
        for i in range(len(below)):
            if below[i] and below[i : i + hold].all():
                return int(i)
        return None

    @property
    def total_seconds(self) -> float:
        return float(self.update_seconds.sum())

    @property
    def seconds_per_update(self) -> float:
        return float(self.update_seconds.mean())


def run_online_adaptation(
    adaptor: OnlineAdaptor,
    transitions: Sequence[Transition],
    door_id: int = -1,
    boundaries: Sequence[int] | None = None,
    init_name: str = "",
    verify_frozen: bool = True,
    progress_every: int = 0,
) -> AdaptationLog:
    """Drive one adaptor over one stream, recording every step.

    ``verify_frozen`` re-checks the network checksum afterwards for latent
    adaptors. It is on by default: the frozen-network requirement is the core
    constraint of Stage 2, so it is checked on every run rather than only in
    the test suite.
    """
    steps: list[AdaptorStep] = []
    for i, (s, a, ns) in enumerate(transitions):
        steps.append(adaptor.observe(s, a, ns))
        if progress_every and (i + 1) % progress_every == 0:
            recent = np.array([st.error[0] for st in steps[-progress_every:]])
            print(f"    step {i + 1:6d}/{len(transitions)}  "
                  f"rolling angle RMSE {np.sqrt((recent**2).mean()):.3e}")

    if verify_frozen and hasattr(adaptor, "assert_network_unchanged"):
        adaptor.assert_network_unchanged()

    extra_keys = set().union(*[set(s.extras) for s in steps]) if steps else set()
    extras = {
        k: np.array([s.extras.get(k, np.nan) for s in steps], dtype=float)
        for k in extra_keys
        if all(isinstance(s.extras.get(k, 0.0), (int, float, bool, np.floating)) for s in steps)
    }

    return AdaptationLog(
        name=adaptor.name,
        door_id=door_id,
        error=np.stack([s.error for s in steps]),
        loss=np.array([s.loss for s in steps]),
        latents=np.stack([s.latent for s in steps]),
        update_seconds=np.array([s.update_seconds for s in steps]),
        boundaries=list(boundaries or []),
        extras=extras,
        init_name=init_name,
    )


__all__ = [
    "AdaptationLog",
    "run_online_adaptation",
    "episode_stream",
    "episode_boundaries",
    "init_strategies",
    "Transition",
]
