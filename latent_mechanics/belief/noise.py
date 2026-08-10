"""
Step 4: innovation-based adaptive noise estimation.

Scheme follows Mehra (1970) as reformulated for the unscented case by Zheng,
Fu, Li & Yuan, "A Robust Adaptive Unscented Kalman Filter for Nonlinear
Estimation with Uncertain Noise Covariance", Sensors 18(3):808, 2018.

The core identity: the filter's own predicted innovation covariance is

    S = Pzz + R

where ``Pzz`` is the spread of the sigma points pushed through the measurement
function -- the uncertainty the filter *already accounts for*. The empirical
covariance of the last N innovations estimates the total. Subtracting gives what
the filter is not accounting for, which is R:

    R_hat = (1/N) * sum_i nu_i nu_i^T  -  Pzz

Why this matters here specifically. The geometry investigation found that only
~47% (median) of prediction error is removable by any choice of z; the rest is
network approximation error and unmodelled physics. In filter terms that
residual *is* measurement noise: large, state-dependent, and impossible to
estimate a priori. A hand-set R would be wrong for every object. Letting the
filter measure its own noise floor is also, in effect, the principled version of
the "when not to adapt" gate that Stages 4 and 5 both identified as missing --
when residuals are dominated by model error, R grows, the Kalman gain shrinks,
and the belief stops chasing noise on its own.

EVERY NUMBER BELOW IS A CONFIGURABLE PARAMETER, NOT A TUNED CHOICE. Window
length, floor, smoothing rate and whether Q adapts at all are surfaced for the
user to select; the defaults are documented starting points, and
``sweep.py`` reports their effect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from latent_mechanics.belief.ukf import nearest_pd, symmetrize


class NoiseModel(ABC):
    """Supplies R (and Q) to the filter and optionally learns them online.

    Deliberately a swappable component: running the fixed-R ablation is then a
    one-line substitution rather than a code change, which is what makes the
    adaptive-vs-fixed comparison trustworthy.
    """

    name: str = "noise"

    @abstractmethod
    def R(self) -> np.ndarray:
        """Current measurement-noise covariance."""

    @abstractmethod
    def Q(self) -> np.ndarray:
        """Current process-noise covariance."""

    def observe(self, innovation: np.ndarray, Pzz: np.ndarray,
                K: np.ndarray | None = None) -> None:
        """Fold one step's innovation into the estimate. No-op if fixed."""

    def reset(self) -> None:
        """Forget accumulated statistics."""

    def diagnostics(self) -> dict:
        return {}


@dataclass
class FixedNoise(NoiseModel):
    """Constant R and Q. The ablation baseline."""

    dim_z: int
    dim_x: int
    r0: float = 1.0
    q0: float = 1e-4
    name: str = "fixed"

    def __post_init__(self) -> None:
        self._R = self.r0 * np.eye(self.dim_z)
        self._Q = self.q0 * np.eye(self.dim_x)

    def R(self) -> np.ndarray:
        return self._R

    def Q(self) -> np.ndarray:
        return self._Q

    def diagnostics(self) -> dict:
        return {"R_trace": float(np.trace(self._R)), "Q_trace": float(np.trace(self._Q))}


@dataclass
class InnovationAdaptiveNoise(NoiseModel):
    """Mehra / RAUKF innovation-based estimation of R, optionally also Q.

    Args:
        window: number of recent innovations in the sample covariance. Short
            windows track a changing noise level but are themselves noisy: with
            ``dim_z`` = 2 the sample covariance of N innovations has roughly
            ``N - 1`` degrees of freedom, so N below ~20 gives an R estimate
            whose own relative error exceeds 30%. Candidate values are swept.
        floor: smallest eigenvalue R is allowed to take. Guarantees positive
            definiteness so the Cholesky in the next sigma-point draw cannot
            fail, and stops R collapsing to zero during a quiet stretch (which
            would make the filter infinitely confident and then diverge on the
            next surprise).
        smoothing: exponential rate at which the new estimate is blended in,
            ``R <- (1-s) R + s R_hat``. 1.0 replaces outright.
        warmup: steps to collect before adapting at all; before this, r0 holds.
        adapt_Q: whether to also estimate Q from ``K C_nu K^T`` (Mehra's
            companion formula). Off by default -- with an identity process model
            Q is a deliberate mobility knob, and letting it adapt couples the
            filter's willingness to move to its own recent surprise, which can
            run away. Exposed so the user can test it.
        q0: fixed process-noise scale used when ``adapt_Q`` is off.
    """

    dim_z: int
    dim_x: int
    window: int = 50
    floor: float = 1e-6
    smoothing: float = 1.0
    warmup: int = 10
    adapt_Q: bool = False
    r0: float = 1.0
    q0: float = 1e-4
    name: str = "adaptive"

    _innovations: deque = field(default_factory=lambda: deque(), init=False)
    _n_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._innovations = deque(maxlen=self.window)
        self._R = self.r0 * np.eye(self.dim_z)
        self._Q = self.q0 * np.eye(self.dim_x)
        self._last_raw = None

    def reset(self) -> None:
        self._innovations.clear()
        self._n_seen = 0
        self._R = self.r0 * np.eye(self.dim_z)
        self._Q = self.q0 * np.eye(self.dim_x)

    def R(self) -> np.ndarray:
        return self._R

    def Q(self) -> np.ndarray:
        return self._Q

    def observe(self, innovation: np.ndarray, Pzz: np.ndarray,
                K: np.ndarray | None = None) -> None:
        nu = np.asarray(innovation, dtype=np.float64).reshape(-1)
        self._innovations.append(nu)
        self._n_seen += 1
        if self._n_seen < self.warmup or len(self._innovations) < 2:
            return

        V = np.stack(self._innovations)          # (N, dim_z)
        # Uncentred second moment: the innovation sequence of a consistent filter
        # is zero-mean, so subtracting a sample mean would remove real signal and
        # bias R low.
        C = (V.T @ V) / len(V)

        R_hat = symmetrize(C - np.asarray(Pzz, dtype=np.float64))
        self._last_raw = R_hat
        R_hat = nearest_pd(R_hat, floor=self.floor)

        s = float(np.clip(self.smoothing, 0.0, 1.0))
        self._R = nearest_pd((1 - s) * self._R + s * R_hat, floor=self.floor)

        if self.adapt_Q and K is not None:
            K = np.asarray(K, dtype=np.float64)
            self._Q = nearest_pd((1 - s) * self._Q + s * (K @ C @ K.T),
                                 floor=self.floor)

    def diagnostics(self) -> dict:
        d = {
            "R_trace": float(np.trace(self._R)),
            "R_min_eig": float(np.linalg.eigvalsh(self._R).min()),
            "Q_trace": float(np.trace(self._Q)),
            "n_innovations": len(self._innovations),
        }
        if self._last_raw is not None:
            # How often the raw Mehra estimate wanted to go non-PD. Frequent
            # clipping means Pzz routinely exceeds the observed innovation
            # spread, i.e. the filter is systematically over-estimating its own
            # uncertainty -- worth knowing, and invisible without this counter.
            d["R_raw_min_eig"] = float(np.linalg.eigvalsh(self._last_raw).min())
        return d


def build_noise_model(kind: str, dim_z: int, dim_x: int, **kwargs) -> NoiseModel:
    if kind == "fixed":
        allowed = {"r0", "q0"}
        return FixedNoise(dim_z=dim_z, dim_x=dim_x,
                          **{k: v for k, v in kwargs.items() if k in allowed})
    if kind == "adaptive":
        allowed = {"window", "floor", "smoothing", "warmup", "adapt_Q", "r0", "q0"}
        return InnovationAdaptiveNoise(dim_z=dim_z, dim_x=dim_x,
                                       **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown noise model {kind!r}; use 'fixed' or 'adaptive'")


__all__ = ["NoiseModel", "FixedNoise", "InnovationAdaptiveNoise", "build_noise_model"]
