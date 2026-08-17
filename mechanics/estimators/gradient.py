"""Online gradient descent on the latent -- Stage 2's belief update.

Delegates every arithmetic operation to ``GradientLatentAdaptor``, so the Adam
state, the Robbins-Monro step decay and the sliding window behave exactly as they
did when the Stage-2 numbers were produced.

One honesty note that the old code hid: the reported innovation is the residual
for the CURRENT sample, but the update applied is a gradient step on the mean loss
over the last ``window`` samples (32 by default). The two are different objects.
``extras["window"]`` records that, so a reader is not misled into thinking this is
a one-sample recursive filter.
"""

from __future__ import annotations

import numpy as np
import torch

from latent_mechanics.online.adaptor import GradientLatentAdaptor
from mechanics.predictor import LatentNetworkPredictor
from mechanics.representation import FullLatent
from mechanics.types import Belief, StepRecord, Transition


class GradientEstimator:
    """Adam/SGD on the latent, over a bounded sliding window."""

    name = "gradient"

    def __init__(
        self,
        predictor: LatentNetworkPredictor,
        lr: float = 0.03,
        optimizer: str = "adam",
        n_inner_steps: int = 1,
        window: int = 32,
        prior_weight: float = 0.0,
        prior_center: np.ndarray | None = None,
        loss_space: str = "normalized",
        max_grad_norm: float = 0.0,
        lr_decay: float = 3e-3,
    ) -> None:
        rep = predictor.representation
        if not isinstance(rep, FullLatent):
            raise TypeError(
                "GradientEstimator differentiates w.r.t. the full latent; pass a "
                f"FullLatent representation, got {type(rep).__name__}. A reduced "
                "chart would need the gradient projected, which is an algorithmic "
                "change, not a wiring change.")
        self.predictor = predictor
        self.representation = rep
        self.window = window
        self._legacy = GradientLatentAdaptor(
            predictor.model,
            init=rep.initial(),
            lr=lr, optimizer=optimizer, n_inner_steps=n_inner_steps,
            window=window, prior_weight=prior_weight, prior_center=prior_center,
            loss_space=loss_space, max_grad_norm=max_grad_norm,
            lr_decay=lr_decay, device=predictor.device,
        )

    def initialize(self) -> Belief:
        return Belief(mean=self.representation.initial(),
                      space=self.representation.name)

    def update(self, belief: Belief, transition: Transition) -> tuple[Belief, StepRecord]:
        # innovation for THIS sample, under the belief held BEFORE the update
        y = self.predictor.measurement(transition.obs, transition.next_obs)
        y_hat = self.predictor.predict_measurement(
            transition.obs, transition.action, belief.mean[None, :])[0]
        innovation = y - y_hat

        step = self._legacy.observe(transition.obs, transition.action,
                                    transition.next_obs)

        new = Belief(mean=np.asarray(step.latent, dtype=np.float64),
                     space=self.representation.name)
        return new, StepRecord(
            prediction=np.zeros(0), target=np.zeros(0), error=np.zeros(0),
            innovation=innovation,
            innovation_space=self.predictor.measurement_space,
            loss=step.loss,
            seconds=step.update_seconds,
            extras={**step.extras, "window": float(self.window)},
        )


__all__ = ["GradientEstimator"]
