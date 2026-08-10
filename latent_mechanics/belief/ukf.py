"""
Step 3: a general Unscented Kalman Filter, independent of this project's model.

Deliberately written as a standalone filter over ``(fx, hx)`` callables so it can
be validated numerically against ``filterpy.kalman.UnscentedKalmanFilter`` before
it is ever pointed at the learned predictor. The latent-specific wiring lives in
``adaptor.py``; nothing in this file knows about mechanics, embeddings or MuJoCo.

Two conventions are copied from filterpy on purpose, because they are choices
rather than necessities and the reference test only passes if they match:

1. **Sigma-point square root.** Merwe scaling uses the *upper* Cholesky factor
   ``U`` of ``(n + lambda) P`` and takes its ROWS as the offsets. numpy's
   ``cholesky`` returns the lower factor, whose rows are a different set of
   vectors; using them silently gives a different (still valid) sigma set that
   will not reproduce the reference.

2. **Sigma points are not regenerated between predict and update.** filterpy
   propagates the sigma points through ``fx``, recombines them into the prior
   mean/covariance, and then reuses those *same* propagated points for the
   measurement update rather than drawing fresh ones from the prior. Both
   variants appear in the literature; this one is the reference's.

   That choice has a measurable consequence, verified in the reference tests:
   the propagated points carry covariance ``F P F^T``, not ``F P F^T + Q``, so
   the update's output covariance and Kalman gain are computed as though the
   process noise had not been added -- while ``P_prior`` used in the covariance
   update *does* include it. On a linear system the filterpy variant therefore
   does NOT reproduce the exact Kalman filter unless ``Q = 0``; the regenerating
   variant does, to 1e-12.

   ``regenerate_sigma_points`` selects between them. Default ``False`` for
   reference compatibility. For this application ``fx`` is the identity and
   ``Q`` is the only thing keeping the belief mobile, so ``True`` is arguably
   the better choice and is flagged as an open decision.

For this application ``fx`` is the identity -- the latent is a constant
parameter, not a moving state -- so the prediction step reduces to inflating the
covariance by ``Q``. ``Q`` is what lets the belief keep moving at all, and is the
knob that decides how much the filter is willing to revise a settled estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg


@dataclass
class MerweSigmaPoints:
    """Van der Merwe scaled sigma points.

    Args:
        n: state dimension.
        alpha: spread of the points about the mean. Small (1e-3 .. 1) keeps them
            close, which matters here because the geometry report measured the
            predictor to be locally linear in z only out to ``|dz| ~ 0.25``;
            sigma points scattered further than that sample a regime where the
            unscented approximation is no longer capturing a near-linear map.
        beta: prior knowledge of the distribution; 2 is optimal for Gaussian.
        kappa: secondary scaling, conventionally 0 or ``3 - n``.
    """

    n: int
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    @property
    def num_sigmas(self) -> int:
        return 2 * self.n + 1

    @property
    def lambda_(self) -> float:
        return self.alpha**2 * (self.n + self.kappa) - self.n

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        n, lam = self.n, self.lambda_
        c = 0.5 / (n + lam)
        Wm = np.full(2 * n + 1, c)
        Wc = np.full(2 * n + 1, c)
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)
        return Wm, Wc

    def compute(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """(2n+1, n) sigma points. Row 0 is the mean."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        P = np.asarray(P, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError(f"expected state of size {self.n}, got {x.shape[0]}")
        U = scipy.linalg.cholesky((self.n + self.lambda_) * P)  # upper, rows used
        sigmas = np.empty((2 * self.n + 1, self.n))
        sigmas[0] = x
        for k in range(self.n):
            sigmas[k + 1] = x + U[k]
            sigmas[self.n + k + 1] = x - U[k]
        return sigmas


def unscented_transform(
    sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray,
    noise_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recombine sigma points into a mean and covariance."""
    x = Wm @ sigmas
    y = sigmas - x
    P = (y.T * Wc) @ y
    if noise_cov is not None:
        P = P + noise_cov
    return x, P


@dataclass
class UKFState:
    x: np.ndarray
    P: np.ndarray
    x_prior: np.ndarray | None = None
    P_prior: np.ndarray | None = None
    K: np.ndarray | None = None       # Kalman gain from the last update
    y: np.ndarray | None = None       # innovation from the last update
    S: np.ndarray | None = None       # innovation covariance (Pyy + R)
    Pzz: np.ndarray | None = None     # sigma-point output covariance, WITHOUT R
    sigmas_f: np.ndarray | None = None
    sigmas_used: np.ndarray | None = None   # points actually fed to hx
    sigmas_h: np.ndarray | None = None


class UnscentedKalmanFilter:
    """A UKF over ``dim_x`` states and ``dim_z`` measurements.

    ``hx`` may be given per-sigma-point (``hx(x) -> (dim_z,)``) or batched
    (``hx_batch(sigmas) -> (2n+1, dim_z)``). The batched form exists because the
    real measurement function is a neural network: evaluating all sigma points in
    one forward pass turns 2n+1 tiny GPU/CPU calls into one, which is the
    difference between the filter being slower and faster than gradient descent.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        points: MerweSigmaPoints,
        fx: Callable[[np.ndarray], np.ndarray] | None = None,
        hx: Callable[[np.ndarray], np.ndarray] | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        inv: Callable[[np.ndarray], np.ndarray] = np.linalg.inv,
        regenerate_sigma_points: bool = False,
    ) -> None:
        if points.n != dim_x:
            raise ValueError("sigma-point dimension must equal dim_x")
        self.dim_x, self.dim_z = dim_x, dim_z
        self.points = points
        self.Wm, self.Wc = points.weights()
        self.fx, self.hx = fx, hx
        self.Q = np.eye(dim_x) if Q is None else np.asarray(Q, dtype=np.float64)
        self.R = np.eye(dim_z) if R is None else np.asarray(R, dtype=np.float64)
        self.inv = inv
        self.regenerate_sigma_points = regenerate_sigma_points
        self.state = UKFState(
            x=np.zeros(dim_x) if x0 is None else np.asarray(x0, float).reshape(-1),
            P=np.eye(dim_x) if P0 is None else np.asarray(P0, float).copy(),
        )

    # -- convenience accessors -------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return self.state.x

    @property
    def P(self) -> np.ndarray:
        return self.state.P

    # -- cycle -------------------------------------------------------------
    def predict(self, fx: Callable | None = None, Q: np.ndarray | None = None) -> None:
        fx = fx or self.fx
        Q = self.Q if Q is None else Q
        st = self.state
        sigmas = self.points.compute(st.x, st.P)
        st.sigmas_f = (np.array([fx(s) for s in sigmas]) if fx is not None
                       else sigmas.copy())
        st.x, st.P = unscented_transform(st.sigmas_f, self.Wm, self.Wc, Q)
        st.x_prior, st.P_prior = st.x.copy(), st.P.copy()

    def update(
        self, z: np.ndarray, R: np.ndarray | None = None,
        hx: Callable | None = None,
        hx_batch: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        R = self.R if R is None else np.asarray(R, dtype=np.float64)
        st = self.state
        if st.sigmas_f is None:
            raise RuntimeError("call predict() before update()")

        # Which points feed the measurement update: the propagated ones
        # (filterpy) or fresh ones drawn from the prior including Q.
        if self.regenerate_sigma_points:
            base = self.points.compute(st.x_prior, nearest_pd(st.P_prior))
        else:
            base = st.sigmas_f
        st.sigmas_used = base

        if hx_batch is not None:
            st.sigmas_h = np.asarray(hx_batch(base), dtype=np.float64)
        else:
            hx = hx or self.hx
            st.sigmas_h = np.atleast_2d([hx(s) for s in base]).astype(np.float64)

        # Output covariance WITHOUT R is kept: the adaptive-R estimator in
        # noise.py needs exactly this quantity to separate "spread the filter
        # already expects" from "spread the measurements actually show".
        zp, Pzz = unscented_transform(st.sigmas_h, self.Wm, self.Wc, None)
        st.Pzz = Pzz
        S = Pzz + R

        dx = base - st.x_prior
        dz = st.sigmas_h - zp
        Pxz = (dx.T * self.Wc) @ dz

        SI = self.inv(S)
        K = Pxz @ SI
        y = np.asarray(z, dtype=np.float64).reshape(-1) - zp

        st.x = st.x_prior + K @ y
        st.P = st.P_prior - K @ S @ K.T
        st.K, st.y, st.S = K, y, S

    def step(self, z, hx_batch=None, R=None, Q=None) -> UKFState:
        self.predict(Q=Q)
        self.update(z, R=R, hx_batch=hx_batch)
        return self.state


def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def nearest_pd(A: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Symmetrise and clip eigenvalues up to ``floor``.

    Repeated Joseph-free covariance updates (``P - K S K^T``) can drift slightly
    indefinite in floating point; this keeps the Cholesky in the next sigma-point
    draw from failing without materially changing the covariance.
    """
    A = symmetrize(np.asarray(A, dtype=np.float64))
    w, V = np.linalg.eigh(A)
    if w.min() >= floor:
        return A
    return V @ np.diag(np.maximum(w, floor)) @ V.T


__all__ = ["MerweSigmaPoints", "UnscentedKalmanFilter", "UKFState",
           "unscented_transform", "symmetrize", "nearest_pd"]
