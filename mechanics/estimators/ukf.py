"""Unscented Kalman filtering of the latent in a reduced chart.

Delegates to ``UKFLatentAdaptor``, which in turn uses ``belief/ukf.py`` -- the
implementation validated to 1e-10 against ``filterpy`` on every intermediate
quantity. Nothing here re-derives any of that.

What this adapter adds is that the innovation stops being an internal variable.
``UKFState.y`` is the SLR-linearised residual the gain is actually applied to; it
is now recorded in the trace like any other quantity, in the normalised-delta
space the filter deliberately works in.
"""

from __future__ import annotations

import numpy as np

from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
from mechanics.predictor import LatentNetworkPredictor
from mechanics.representation import ReducedLatent
from mechanics.types import Belief, StepRecord, Transition


class UKFEstimator:
    """UKF over the mechanics belief, in a frozen PCA chart of the latent."""

    name = "ukf"

    def __init__(
        self,
        predictor: LatentNetworkPredictor,
        cfg: UKFConfig | None = None,
    ) -> None:
        rep = predictor.representation
        if not isinstance(rep, ReducedLatent):
            raise TypeError(
                "UKFEstimator filters a reduced chart; pass a ReducedLatent "
                f"representation, got {type(rep).__name__}.")
        self.predictor = predictor
        self.representation = rep
        self.cfg = cfg or UKFConfig(dim=rep.dim)
        if self.cfg.dim != rep.dim:
            raise ValueError(
                f"UKFConfig.dim={self.cfg.dim} but the representation has "
                f"dim={rep.dim}; they must agree.")
        self._legacy = UKFLatentAdaptor(
            predictor.model,
            basis=rep.basis,
            cfg=self.cfg,
            init=rep.init,                       # full-latent coordinates
            prior_latents=rep.prior_latents,
            device=predictor.device,
        )

    def initialize(self) -> Belief:
        st = self._legacy.ukf
        return Belief(mean=st.x.copy(), cov=st.P.copy(),
                      space=self.representation.name)

    def update(self, belief: Belief, transition: Transition) -> tuple[Belief, StepRecord]:
        step = self._legacy.observe(transition.obs, transition.action,
                                    transition.next_obs)
        st = self._legacy.ukf.state

        # the residual the gain was applied to, not a recomputation of it
        innovation = (np.asarray(st.y, dtype=np.float64).reshape(-1)
                      if st.y is not None else np.full(self.predictor.obs_dim, np.nan))

        new = Belief(
            mean=self._legacy.ukf.x.copy(),
            cov=self._legacy.ukf.P.copy(),
            space=self.representation.name,
            extras={"R": self._legacy.noise.R().copy(),
                    "Q": self._legacy.noise.Q().copy()},
        )
        return new, StepRecord(
            prediction=np.zeros(0), target=np.zeros(0), error=np.zeros(0),
            innovation=innovation,
            innovation_space=self.predictor.measurement_space,
            loss=step.loss,
            seconds=step.update_seconds,
            extras=step.extras,
        )

    def full_latent(self) -> np.ndarray:
        """The belief decoded into full 16-D latent coordinates."""
        return self.representation.to_predictor(self._legacy.ukf.x)


__all__ = ["UKFEstimator", "UKFConfig"]
