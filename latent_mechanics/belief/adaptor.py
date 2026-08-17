"""UKF as a drop-in replacement for the gradient-descent belief update.

Two design points: the filter works in normalised measurement space (``h(z)`` is
``model.raw_output``), which is the only space where one R is meaningful across
families; and the latent is a constant, so ``fx`` is the identity and Q alone
decides how much a settled belief may still be revised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from latent_mechanics.belief.basis import LatentBasis
from latent_mechanics.belief.noise import NoiseModel, build_noise_model
from latent_mechanics.belief.ukf import (
    MerweSigmaPoints,
    UnscentedKalmanFilter,
    nearest_pd,
)
from latent_mechanics.model import MechanicsDynamicsModel
from latent_mechanics.online.adaptor import ArrayLike, OnlineLatentAdaptor


@dataclass
class UKFConfig:
    """Filter settings. Fields marked CHOSEN were swept; the rest are starting points."""

    dim: int = 6                     # CHOSEN: best ratio to the oracle ceiling. sweep.py
    # alpha=1.0/kappa=0 keeps all sigma weights non-negative; 0.3 at d=6 gives
    # Wm[0]=-10.1 and the transform error swamps the innovation.
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    n_iterations: int = 3            # 1 = plain one-shot update
    iter_tol: float = 1e-4
    regenerate_sigma_points: bool = True   # CHOSEN: exact KF on linear systems when Q > 0
    p0_mode: str = "empirical"       # empirical (training-latent cov) | scalar
    p0_scale: float = 1.0
    # residual form; the innovation form's C_nu - Pzz went indefinite on 30-41% of steps
    noise_kind: str = "residual"     # residual | adaptive | fixed
    r0: float = 1.0
    q0: float = 1e-4
    window: int = 100                # CHOSEN: only setting that survives drift. drift_check.py
    floor: float = 1e-6              # scalar floor, legacy innovation model only
    floor_matrix: np.ndarray | None = None   # None -> noise.IRREDUCIBLE_R
    smoothing: float = 1.0           # unswept
    warmup: int = 10
    adapt_Q: bool = False            # off: couples mobility to surprise and can run away


class UKFLatentAdaptor(OnlineLatentAdaptor):
    """Unscented Kalman filtering of the latent, in a reduced fixed basis."""

    name = "ukf"

    def __init__(
        self,
        model: MechanicsDynamicsModel,
        basis: LatentBasis,
        cfg: UKFConfig | None = None,
        init: ArrayLike | None = None,
        prior_latents: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or UKFConfig()
        self.basis = basis.truncate(self.cfg.dim) if basis.dim > self.cfg.dim else basis
        if self.basis.dim != self.cfg.dim:
            raise ValueError(
                f"basis has {self.basis.dim} components but cfg.dim={self.cfg.dim}; "
                "recompute the basis with at least that many components")
        self._prior_latents = prior_latents
        super().__init__(model, init=init, device=device)

    # -- setup -------------------------------------------------------------
    def _initial_covariance(self) -> np.ndarray:
        d = self.cfg.dim
        if self.cfg.p0_mode == "empirical" and self._prior_latents is not None:
            zr = self.basis.encode(np.asarray(self._prior_latents, dtype=np.float64))
            P = np.cov(np.atleast_2d(zr), rowvar=False)
            P = np.atleast_2d(P) * self.cfg.p0_scale
            return nearest_pd(P, floor=1e-9)
        return self.cfg.p0_scale * np.eye(d)

    def _on_reset(self) -> None:
        d = self.cfg.dim
        c = self.cfg
        self.noise: NoiseModel = build_noise_model(
            c.noise_kind, dim_z=self.model.state_dim, dim_x=d,
            r0=c.r0, q0=c.q0, window=c.window, floor=c.floor,
            floor_matrix=c.floor_matrix,
            smoothing=c.smoothing, warmup=c.warmup, adapt_Q=c.adapt_Q)

        z0_full = self._z.detach().cpu().numpy().astype(np.float64)
        x0 = np.atleast_1d(self.basis.encode(z0_full)).reshape(-1)

        self.ukf = UnscentedKalmanFilter(
            dim_x=d, dim_z=self.model.state_dim,
            points=MerweSigmaPoints(d, c.alpha, c.beta, c.kappa),
            fx=None,                                   # identity: z is constant
            Q=self.noise.Q(), R=self.noise.R(),
            x0=x0, P0=self._initial_covariance(),
            regenerate_sigma_points=c.regenerate_sigma_points,
        )
        self._sync_latent()   # keep the base class's 16-D latent consistent

    def _sync_latent(self) -> None:
        z_full = self.basis.decode(self.ukf.x)
        with torch.no_grad():
            self._z.copy_(torch.as_tensor(z_full, dtype=torch.float32,
                                          device=self.device).reshape(-1))

    # -- measurement function ---------------------------------------------
    def _make_hx_batch(self, s: torch.Tensor, a: torch.Tensor):
        """All sigma points through the predictor in one batched forward pass."""
        def hx_batch(sigmas_r: np.ndarray) -> np.ndarray:
            z_full = self.basis.decode(np.asarray(sigmas_r, dtype=np.float64))
            n = z_full.shape[0]
            zt = torch.as_tensor(z_full, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out = self.model.raw_output(s.expand(n, -1), a.expand(n, -1), zt)
            return out.cpu().numpy().astype(np.float64)

        return hx_batch

    # -- the interface method ---------------------------------------------
    def _update(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[float, dict[str, Any]]:
        with torch.no_grad():
            y = self.model.target(state, next_state)[0].cpu().numpy().astype(np.float64)

        self.ukf.Q = self.noise.Q()
        self.ukf.predict()
        hx_batch = self._make_hx_batch(state, action)
        try:
            if self.cfg.n_iterations > 1:
                self.ukf.iterated_update(
                    y, R=self.noise.R(), hx_batch=hx_batch,
                    n_iterations=self.cfg.n_iterations, tol=self.cfg.iter_tol)
            else:
                self.ukf.update(y, R=self.noise.R(), hx_batch=hx_batch)
        except np.linalg.LinAlgError:
            # P drifted indefinite: repair and skip this measurement
            self.ukf.state.P = nearest_pd(self.ukf.state.P, floor=1e-9)
            return float("nan"), {"filter_reset": 1.0}

        st = self.ukf.state
        self.ukf.state.P = nearest_pd(st.P, floor=1e-12)
        self.noise.observe(st.y, st.Pzz, st.K, residual=st.y_post)
        self._sync_latent()

        diag = self.noise.diagnostics()
        return float(np.mean(st.y**2)), {
            "innovation_norm": float(np.linalg.norm(st.y)),
            "gain_norm": float(np.linalg.norm(st.K)),
            "P_trace": float(np.trace(st.P)),
            "S_trace": float(np.trace(st.S)),
            "n_iterations": float(st.n_iterations),
            **diag,
        }

    # -- belief ------------------------------------------------------------
    def belief(self) -> dict[str, Any]:
        """Mean and covariance in full 16-D coordinates; the covariance is rank ``dim``."""
        return {
            "mean": self.latent,
            "cov": self.basis.decode_covariance(self.ukf.P),
            "mean_reduced": self.ukf.x.copy(),
            "cov_reduced": self.ukf.P.copy(),
            "R": self.noise.R().copy(),
            "Q": self.noise.Q().copy(),
        }

    @property
    def covariance(self) -> np.ndarray:
        return self.basis.decode_covariance(self.ukf.P)


__all__ = ["UKFLatentAdaptor", "UKFConfig"]
