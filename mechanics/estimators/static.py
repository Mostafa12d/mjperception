"""The no-adaptation control.

Never updates. Without it, "the error went down" could just mean the stream got
easier -- this control is what revealed Stage 5's central negative result, so it
is not optional and every experiment spec includes it by default.

The only estimator here that does not delegate to a legacy class, because
"return the belief unchanged" has nothing to delegate. Its residual is the same
normalised-delta quantity ``StaticLatentAdaptor._loss_value`` computed, and
``mechanics/tests.py`` checks that against the legacy class anyway.
"""

from __future__ import annotations

import time

import numpy as np

from mechanics.predictor import MeasurementPredictor
from mechanics.types import Belief, StepRecord, Transition


class StaticEstimator:
    """Holds the prior belief fixed. Reports the residual it would have used."""

    name = "no-adaptation"

    def __init__(self, predictor: MeasurementPredictor) -> None:
        self.predictor = predictor
        self.representation = predictor.representation

    def initialize(self) -> Belief:
        return Belief(mean=self.representation.initial(),
                      space=self.representation.name)

    def update(self, belief: Belief, transition: Transition) -> tuple[Belief, StepRecord]:
        t0 = time.perf_counter()
        y = self.predictor.measurement(transition.obs, transition.next_obs)
        y_hat = self.predictor.predict_measurement(
            transition.obs, transition.action, belief.mean[None, :])[0]
        innovation = y - y_hat
        elapsed = time.perf_counter() - t0

        return belief, StepRecord(
            prediction=np.zeros(0), target=np.zeros(0), error=np.zeros(0),
            innovation=innovation,
            innovation_space=self.predictor.measurement_space,
            loss=float(np.mean(innovation ** 2)),
            seconds=elapsed,
            extras={},
        )


__all__ = ["StaticEstimator"]
