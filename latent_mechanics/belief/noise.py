"""Adaptive measurement-noise estimation for the UKF.

Innovation form (Mehra 1970; Zheng et al., Sensors 18(3):808, 2018) estimates
``R_hat = mean(nu nu^T) - Pzz``; residual form (Mohamed & Schwarz, J. Geodesy
73:193-203, 1999) estimates ``R_hat = mean(eps eps^T) + Pzz_post``. All defaults
are starting points, not tuned choices; ``sweep.py`` reports their effect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from latent_mechanics.belief.ukf import nearest_pd, psd_floor, symmetrize

# Residual at the offline-optimal latent, pooled over the 120 training objects.
# No choice of z beats this, so R may never fall below it.
# Reproduce with: python3.10 -m latent_mechanics.belief.calibrate_noise
IRREDUCIBLE_R = np.array([[1.1300e-03, 1.0640e-02],
                          [1.0640e-02, 1.6809e-01]])


class NoiseModel(ABC):
    """Supplies R (and Q) to the filter and optionally learns them online."""

    name: str = "noise"

    @abstractmethod
    def R(self) -> np.ndarray:
        """Current measurement-noise covariance."""

    @abstractmethod
    def Q(self) -> np.ndarray:
        """Current process-noise covariance."""

    def observe(self, innovation: np.ndarray, Pzz: np.ndarray,
                K: np.ndarray | None = None,
                residual: np.ndarray | None = None) -> None:
        """Fold one step in. ``residual`` is the post-update residual (iterated update only)."""

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

    Legacy: ``R_hat = C_nu - Pzz`` can go indefinite (30-41% of steps here) and
    is then clipped to a scalar floor. Prefer ``ResidualAdaptiveNoise``.
    """

    dim_z: int
    dim_x: int
    window: int = 50          # below ~20 the R estimate's own error exceeds 30%
    floor: float = 1e-6       # smallest eigenvalue R may take
    smoothing: float = 1.0    # R <- (1-s) R + s R_hat; 1.0 replaces outright
    warmup: int = 10
    adapt_Q: bool = False     # Mehra's K C_nu K^T; off, can run away
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
                K: np.ndarray | None = None,
                residual: np.ndarray | None = None) -> None:
        nu = np.asarray(innovation, dtype=np.float64).reshape(-1)
        self._innovations.append(nu)
        self._n_seen += 1
        if self._n_seen < self.warmup or len(self._innovations) < 2:
            return

        V = np.stack(self._innovations)          # (N, dim_z)
        C = (V.T @ V) / len(V)   # uncentred: a consistent filter's innovations are zero-mean

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
            # negative => Pzz exceeded the observed innovation spread
            d["R_raw_min_eig"] = float(np.linalg.eigvalsh(self._last_raw).min())
        return d


@dataclass
class ResidualAdaptiveNoise(NoiseModel):
    """Residual-form adaptive R. Mohamed & Schwarz, J. Geodesy 73:193-203, 1999.

    ``R_hat = mean(eps eps^T) + Pzz_post`` is a sum of PSD terms, so unlike the
    innovation form it cannot go indefinite. ``floor_matrix`` is a matrix floor
    applied in the Loewner order: a scalar one cannot express the correlation
    between channels and collapses R onto the near-kinematic d_theta channel.
    """

    dim_z: int
    dim_x: int
    window: int = 100
    floor_matrix: np.ndarray | None = None   # None -> IRREDUCIBLE_R
    smoothing: float = 1.0
    warmup: int = 10
    r0: float = 1.0
    q0: float = 1e-4
    name: str = "residual"

    _residuals: deque = field(default_factory=lambda: deque(), init=False)
    _n_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._residuals = deque(maxlen=self.window)
        F = (IRREDUCIBLE_R if self.floor_matrix is None
             else np.asarray(self.floor_matrix, dtype=np.float64))
        if F.shape != (self.dim_z, self.dim_z):
            raise ValueError(f"floor_matrix must be {self.dim_z}x{self.dim_z}, got {F.shape}")
        self._F = symmetrize(F)
        self._R = psd_floor(self.r0 * np.eye(self.dim_z), self._F)
        self._Q = self.q0 * np.eye(self.dim_x)
        self._n_floored = 0

    def reset(self) -> None:
        self._residuals.clear()
        self._n_seen = 0
        self._n_floored = 0
        self._R = psd_floor(self.r0 * np.eye(self.dim_z), self._F)

    def R(self) -> np.ndarray:
        return self._R

    def Q(self) -> np.ndarray:
        return self._Q

    def observe(self, innovation: np.ndarray, Pzz: np.ndarray,
                K: np.ndarray | None = None,
                residual: np.ndarray | None = None) -> None:
        # fall back to the innovation so a non-iterated filter still runs
        eps = np.asarray(innovation if residual is None else residual,
                         dtype=np.float64).reshape(-1)
        self._residuals.append(eps)
        self._n_seen += 1
        if self._n_seen < self.warmup or len(self._residuals) < 2:
            return

        V = np.stack(self._residuals)
        C = (V.T @ V) / len(V)
        R_hat = symmetrize(C + np.asarray(Pzz, dtype=np.float64))

        s = float(np.clip(self.smoothing, 0.0, 1.0))
        blended = symmetrize((1 - s) * self._R + s * R_hat)
        floored = psd_floor(blended, self._F)
        if not np.allclose(floored, blended, rtol=1e-9, atol=1e-12):
            self._n_floored += 1
        self._R = floored

    def diagnostics(self) -> dict:
        return {
            "R_trace": float(np.trace(self._R)),
            "R_min_eig": float(np.linalg.eigvalsh(self._R).min()),
            "Q_trace": float(np.trace(self._Q)),
            "n_residuals": len(self._residuals),
            "frac_floored": self._n_floored / max(self._n_seen, 1),
        }


def build_noise_model(kind: str, dim_z: int, dim_x: int, **kwargs) -> NoiseModel:
    if kind == "fixed":
        allowed = {"r0", "q0"}
        return FixedNoise(dim_z=dim_z, dim_x=dim_x,
                          **{k: v for k, v in kwargs.items() if k in allowed})
    if kind == "adaptive":
        allowed = {"window", "floor", "smoothing", "warmup", "adapt_Q", "r0", "q0"}
        return InnovationAdaptiveNoise(dim_z=dim_z, dim_x=dim_x,
                                       **{k: v for k, v in kwargs.items() if k in allowed})
    if kind == "residual":
        allowed = {"window", "floor_matrix", "smoothing", "warmup", "r0", "q0"}
        return ResidualAdaptiveNoise(dim_z=dim_z, dim_x=dim_x,
                                     **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown noise model {kind!r}; use 'fixed', 'adaptive' or 'residual'")


__all__ = ["NoiseModel", "FixedNoise", "InnovationAdaptiveNoise",
           "ResidualAdaptiveNoise", "build_noise_model", "IRREDUCIBLE_R"]
