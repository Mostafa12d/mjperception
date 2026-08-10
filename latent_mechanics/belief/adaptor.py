"""
Step 5: the UKF as a drop-in replacement for the gradient-descent belief update.

``UKFLatentAdaptor`` subclasses the same ``OnlineLatentAdaptor`` base as
``GradientLatentAdaptor``, implements the same single abstract method
``_update(state, action, next_state)``, and therefore satisfies the existing
interface exactly: ``predict``, ``observe``, ``reset``, ``latent``, ``belief``
and the frozen-network guarantees all come from the base class unchanged. Any
driver that runs the gradient adaptor runs this one with no edits.

What is new is that ``belief()`` now returns a real covariance instead of
``None``. The slot has been there since Stage 2 and was never filled.

Two design points worth stating.

**The filter works in normalised measurement space.** ``h(z)`` is
``model.raw_output`` -- the normalised state delta -- and the measurement is
``model.target``, not the raw next state. This is the space the network was
trained in, and it is the only one in which a single R is meaningful across
families: raw next-state units are radians for a door and metres for a drawer,
differing by orders of magnitude, so a shared R in raw units would be
simultaneously far too tight for one family and far too loose for another.

**The state is a constant, not a trajectory.** The mechanics of an object do not
evolve, so ``fx`` is the identity and the entire prediction step is the
covariance inflation ``P <- P + Q``. Q is therefore not a physical process noise
but an explicit statement of how much the filter is willing to keep revising a
settled belief -- the direct analogue of the gradient adaptor's learning rate,
and the reason Stage 5's "adaptation never stops when it should" failure has a
principled fix here.
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
    """Filter settings.

    ``dim``, ``window``, ``adapt_Q`` and ``regenerate_sigma_points`` were chosen
    by the user on the evidence in ``sweep.py`` and ``drift_check.py``; the rest
    remain unswept starting points. Provenance is noted per field.
    """

    # CHOSEN. Best ratio to the oracle ceiling (0.97x vs 1.12x at d=5, 1.17x at
    # d=4) at equal cost and equal stability. sweep.py.
    dim: int = 6
    alpha: float = 0.3               # sigma-point spread
    beta: float = 2.0
    kappa: float = 0.0
    # CHOSEN. Reproduces the exact Kalman filter on linear systems when Q > 0;
    # filterpy's convention does not. See ukf.py.
    regenerate_sigma_points: bool = True
    # Initial covariance. "empirical" uses the covariance of the training
    # latents projected into the reduced basis, which is the honest prior: it
    # says the unseen object is about as far from the mean as a training object.
    p0_mode: str = "empirical"       # empirical | scalar
    p0_scale: float = 1.0
    noise_kind: str = "adaptive"     # adaptive | fixed
    r0: float = 1.0
    q0: float = 1e-4
    # CHOSEN. Stationary objects marginally prefer 50 (0.54x vs 0.59x relative
    # to no-adaptation), but under Stage-3 friction drift at 0.40/s window 50
    # becomes actively harmful (1.42x) while 100 is the only setting still
    # helping (0.94x). The ~9% given up on stationary objects buys the filter
    # not breaking on time-varying ones. drift_check.py.
    window: int = 100
    floor: float = 1e-6           # unswept; follow-up
    smoothing: float = 1.0        # unswept; follow-up
    warmup: int = 10
    # LEFT OFF by decision: couples mobility to recent surprise and can run away.
    adapt_Q: bool = False


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
        # Keep the base class's 16-D latent exactly consistent with the filter.
        self._sync_latent()

    def _sync_latent(self) -> None:
        z_full = self.basis.decode(self.ukf.x)
        with torch.no_grad():
            self._z.copy_(torch.as_tensor(z_full, dtype=torch.float32,
                                          device=self.device).reshape(-1))

    # -- measurement function ---------------------------------------------
    def _make_hx_batch(self, s: torch.Tensor, a: torch.Tensor):
        """All sigma points through the predictor in ONE batched forward pass.

        Cost per update is therefore one network call, not 2d+1 -- the same order
        as a single gradient step, which is what keeps the filter competitive on
        compute with the module it replaces.
        """
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
        try:
            self.ukf.update(y, R=self.noise.R(),
                            hx_batch=self._make_hx_batch(state, action))
        except np.linalg.LinAlgError:
            # A failed Cholesky means P drifted indefinite. Repair and skip this
            # measurement rather than propagating NaNs through the whole run.
            self.ukf.state.P = nearest_pd(self.ukf.state.P, floor=1e-9)
            return float("nan"), {"filter_reset": 1.0}

        st = self.ukf.state
        self.ukf.state.P = nearest_pd(st.P, floor=1e-12)
        self.noise.observe(st.y, st.Pzz, st.K)
        self._sync_latent()

        diag = self.noise.diagnostics()
        return float(np.mean(st.y**2)), {
            "innovation_norm": float(np.linalg.norm(st.y)),
            "gain_norm": float(np.linalg.norm(st.K)),
            "P_trace": float(np.trace(st.P)),
            "S_trace": float(np.trace(st.S)),
            **diag,
        }

    # -- belief ------------------------------------------------------------
    def belief(self) -> dict[str, Any]:
        """Mean and covariance in FULL 16-D coordinates.

        The covariance is ``V^T P_r V``: rank ``d`` and therefore singular in
        16-D. That is the honest representation of what the filter believes --
        zero uncertainty in the directions the reduced basis discards.
        """
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
