"""Standalone UKF over ``(fx, hx)`` callables. Latent-specific wiring is in adaptor.py.

Two filterpy conventions are matched so the reference test passes: sigma offsets
are the ROWS of the UPPER Cholesky factor, and sigma points are not regenerated
between predict and update (see ``regenerate_sigma_points``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg


@dataclass
class MerweSigmaPoints:
    """Van der Merwe scaled sigma points. Small alpha keeps points inside the
    predictor's locally-linear region (|dz| ~ 0.25)."""

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
    y_post: np.ndarray | None = None  # post-update residual; iterated_update only
    n_iterations: int = 0


class UnscentedKalmanFilter:
    """UKF over ``dim_x`` states and ``dim_z`` measurements.

    ``hx`` is per-point (``hx(x) -> (dim_z,)``) or batched
    (``hx_batch(sigmas) -> (2n+1, dim_z)``); batched matters when hx is a network.
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

        # propagated points (filterpy) vs fresh ones drawn from the prior incl. Q
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

        # Pzz is kept without R; the adaptive-R estimator in noise.py needs it
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

    # -- iterated update ---------------------------------------------------
    def iterated_update(
        self, z: np.ndarray, R: np.ndarray | None = None,
        hx: Callable | None = None,
        hx_batch: Callable[[np.ndarray], np.ndarray] | None = None,
        n_iterations: int = 3, tol: float = 1e-4,
    ) -> UKFState:
        """Iterated posterior-linearisation update (IPLF; Garcia-Fernandez et al.,
        IEEE TSP 63(20):5561-5573, 2015).

        Each iteration re-linearises ``h`` around the current posterior but redoes
        the update from the prior, so the measurement is used exactly once.
        ``n_iterations=1`` reduces to the standard sigma-point update.
        """
        R = self.R if R is None else np.asarray(R, dtype=np.float64)
        st = self.state
        if st.x_prior is None:
            raise RuntimeError("call predict() before iterated_update()")
        x_prior = st.x_prior
        P_prior = nearest_pd(st.P_prior)
        z = np.asarray(z, dtype=np.float64).reshape(-1)

        def measure(sigmas: np.ndarray) -> np.ndarray:
            if hx_batch is not None:
                return np.asarray(hx_batch(sigmas), dtype=np.float64)
            f = hx or self.hx
            return np.atleast_2d([f(s) for s in sigmas]).astype(np.float64)

        x_j, P_j = x_prior.copy(), P_prior.copy()
        A = b = Omega = S = K = None
        it = 0
        for it in range(max(1, n_iterations)):
            sigmas = self.points.compute(x_j, P_j)
            h = measure(sigmas)
            zbar, Pzz = unscented_transform(h, self.Wm, self.Wc, None)
            Pxz = ((sigmas - x_j).T * self.Wc) @ (h - zbar)

            # SLR of h over N(x_j, P_j):  h(x) ~= A x + b + noise(Omega)
            A = np.linalg.solve(P_j, Pxz).T
            b = zbar - A @ x_j
            Omega = nearest_pd(Pzz - A @ P_j @ A.T, floor=1e-12)

            # ...but the update always starts from the prior
            S = symmetrize(A @ P_prior @ A.T + Omega + R)
            K = P_prior @ A.T @ self.inv(S)
            x_new = x_prior + K @ (z - (A @ x_prior + b))
            P_new = nearest_pd(P_prior - K @ S @ K.T, floor=1e-12)

            shift = float(np.linalg.norm(x_new - x_j))
            x_j, P_j = x_new, P_new
            if shift < tol:
                break

        st.x, st.P = x_j, P_j
        st.K, st.S = K, S
        st.y = z - (A @ x_prior + b)      # prior innovation
        st.y_post = z - (A @ x_j + b)     # post-update residual
        st.Pzz = symmetrize(A @ P_j @ A.T + Omega)  # posterior spread, PSD by construction
        st.n_iterations = it + 1
        st.sigmas_used, st.sigmas_h = None, None
        return st

    def step(self, z, hx_batch=None, R=None, Q=None) -> UKFState:
        self.predict(Q=Q)
        self.update(z, R=R, hx_batch=hx_batch)
        return self.state


def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def nearest_pd(A: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Symmetrise and clip eigenvalues up to ``floor``, so the next Cholesky holds."""
    A = symmetrize(np.asarray(A, dtype=np.float64))
    w, V = np.linalg.eigh(A)
    if w.min() >= floor:
        return A
    return V @ np.diag(np.maximum(w, floor)) @ V.T


def psd_floor(A: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Smallest matrix ``>= A`` (Loewner order) that is also ``>= F``.

    An anisotropic floor: clip the eigenvalues of ``F^-1/2 A F^-1/2`` up at 1 and
    map back. A scalar floor would leave the quiet channel effectively noiseless.
    """
    A = symmetrize(np.asarray(A, dtype=np.float64))
    F = symmetrize(np.asarray(F, dtype=np.float64))
    L = np.linalg.cholesky(nearest_pd(F, floor=1e-15))
    Li = np.linalg.inv(L)
    w, V = np.linalg.eigh(symmetrize(Li @ A @ Li.T))
    M = V @ np.diag(np.maximum(w, 1.0)) @ V.T
    return symmetrize(L @ M @ L.T)


__all__ = ["MerweSigmaPoints", "UnscentedKalmanFilter", "UKFState",
           "unscented_transform", "symmetrize", "nearest_pd", "psd_floor"]
