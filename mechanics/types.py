"""The vocabulary of the research loop: what flows between the four components.

Five objects, and every one of them corresponds to an arrow in the diagram:

    Transition   what the estimator is shown at one timestep
    Belief       the current estimate of the object's mechanics
    StepRecord   what happened at one timestep, including THE INNOVATION
    Trace        the whole run, plus the metrics every experiment reports

``StepRecord`` is the piece the old code did not have. Previously the residual
that drove the belief lived inside ``_update`` and never came out, so it could not
be logged, plotted, or compared between estimators -- and the three estimators do
not even use the same space. ``innovation`` plus ``innovation_space`` makes that
explicit rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Innovation spaces currently in use. Not an enum: a new estimator may legitimately
# introduce a new one, and it should be able to do so by naming it.
OBSERVATION = "observation"            # raw SI next-observation difference
NORMALISED_DELTA = "normalised_delta"  # (next_obs - obs) in the predictor's z-scored units
TORQUE = "torque"                      # generalised force residual (RLS)


@dataclass(frozen=True)
class Transition:
    """One interaction, as the estimator sees it.

    ``obs``/``next_obs`` are what the OBSERVATION MODEL produced -- possibly noisy,
    quantised or stale. ``truth`` is the clean next state and exists only so
    scoring can measure the estimator rather than the sensor. No estimator may
    read ``truth``; the driver does not pass it to them.
    """

    obs: np.ndarray        # (n_obs,)
    action: np.ndarray     # (n_act,)  zero-order held over [t, t+dt)
    next_obs: np.ndarray   # (n_obs,)
    truth: np.ndarray | None = None   # (n_obs,) clean next state, scoring only

    @property
    def target(self) -> np.ndarray:
        """What a prediction is scored against: clean truth when we have it."""
        return self.next_obs if self.truth is None else self.truth


@dataclass(frozen=True)
class Belief:
    """The mechanics belief, in the coordinates of some MechanicsRepresentation.

    ``mean`` is in REPRESENTATION coordinates (6-D for a reduced latent, 5-D for
    physical parameters), not necessarily in whatever the predictor consumes --
    ``MechanicsRepresentation.to_predictor`` makes that conversion. ``space`` names
    the representation so two beliefs are never silently compared across charts,
    which the old ``AdaptationLog.latents`` column did.
    """

    mean: np.ndarray
    cov: np.ndarray | None = None
    space: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float64).reshape(-1))

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])


@dataclass(frozen=True)
class StepRecord:
    """What happened at one timestep.

    ``prediction`` is the reported one-step-ahead prediction, made with the belief
    held BEFORE this transition -- the prequential protocol. ``innovation`` is the
    residual the estimator actually applied, which is a different quantity in a
    different space for every estimator; ``innovation_space`` says which.
    """

    prediction: np.ndarray         # (n_obs,) predicted next observation, raw units
    target: np.ndarray             # (n_obs,) what it is scored against
    error: np.ndarray              # (n_obs,) prediction - target, raw units
    innovation: np.ndarray         # the residual that drove the update
    innovation_space: str
    loss: float
    seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A complete run. Metric definitions are lifted verbatim from
    ``online.loop.AdaptationLog`` so migrated numbers stay comparable."""

    name: str
    object_id: int
    error: np.ndarray            # (T, n_obs) signed prequential error
    innovation: np.ndarray       # (T, k)
    innovation_space: str
    loss: np.ndarray             # (T,)
    beliefs: np.ndarray          # (T, d) belief mean after each step
    belief_space: str
    seconds: np.ndarray          # (T,)
    boundaries: list[int] = field(default_factory=list)
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    init_name: str = ""

    def __len__(self) -> int:
        return len(self.loss)

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
        """First step whose rolling RMSE drops below ``threshold`` and stays below
        for ``hold`` consecutive steps. ``None`` if never."""
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
        return float(self.seconds.sum())

    @property
    def seconds_per_update(self) -> float:
        return float(self.seconds.mean())

    @property
    def belief_travel(self) -> float:
        """Distance the belief moved, in its own coordinates. Only comparable
        between traces with the same ``belief_space``."""
        if len(self.beliefs) < 2:
            return 0.0
        return float(np.linalg.norm(self.beliefs[-1] - self.beliefs[0]))


__all__ = [
    "Transition", "Belief", "StepRecord", "Trace",
    "OBSERVATION", "NORMALISED_DELTA", "TORQUE",
]
