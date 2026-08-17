"""Recursive least squares on interpretable hinge parameters -- the baseline.

This is the one the brief singles out: *"Existing working baselines, especially
RLS, should remain intact."* It therefore delegates to ``RLSAdaptor``, which
delegates to ``dyn.rls_init`` / ``dyn.rls_step`` unmodified. Not one arithmetic
operation is reimplemented here, and ``mechanics/tests.py`` asserts exact float64
agreement with the legacy path.

The three fairness choices that made the original comparison honest are inherited
untouched: the 5-parameter spring-aware regressor, velocity gating, and
sub-stepped integration at MuJoCo's own 0.002 s.

Innovation: ``tau - phi^T theta``, in TORQUE units -- a different space from the
latent estimators, which is exactly why ``innovation_space`` exists. Unlike the
legacy ``loss`` field (which used the POST-update parameters), this is computed
with the parameters held before the update, so it is a genuine innovation. The
legacy value is preserved unchanged in ``loss``.
"""

from __future__ import annotations

import numpy as np

from latent_mechanics.online.rls_adaptor import RLSAdaptor
from mechanics.predictor import AnalyticalPredictor
from mechanics.representation import PhysicalParameters
from mechanics.types import TORQUE, Belief, StepRecord, Transition


class RLSEstimator:
    """RLS over ``[I, mu, b, k, c]``."""

    def __init__(
        self,
        predictor: AnalyticalPredictor,
        n_params: int = 5,
        lam: float = 0.995,
        delta: float = 1e3,
        vel_thresh: float = 0.02,
        n_substeps: int = 10,
    ) -> None:
        rep = predictor.representation
        if not isinstance(rep, PhysicalParameters):
            raise TypeError(
                "RLSEstimator estimates physical parameters; pass a "
                f"PhysicalParameters representation, got {type(rep).__name__}.")
        self.predictor = predictor
        self.representation = rep
        self.n_params = n_params
        self.name = f"rls-{n_params}p"
        self._legacy = RLSAdaptor(
            dt=predictor.dt, n_substeps=n_substeps, n_params=n_params,
            lam=lam, delta=delta, vel_thresh=vel_thresh,
            init=rep.initial() if rep.dim == n_params else None,
        )

    def initialize(self) -> Belief:
        return Belief(mean=self._legacy.params, cov=self._legacy._rls.P.copy(),
                      space=self.representation.name)

    def _regressor(self, obs, next_obs) -> np.ndarray:
        th, thd = (float(v) for v in np.asarray(obs).reshape(-1)[:2])
        thd_next = float(np.asarray(next_obs).reshape(-1)[1])
        thdd = (thd_next - thd) / self._legacy.dt
        phi = [thdd, np.sign(thd), thd]
        if self.n_params == 5:
            phi += [th, 1.0]
        return np.array(phi, dtype=np.float64)

    def update(self, belief: Belief, transition: Transition) -> tuple[Belief, StepRecord]:
        # innovation under the PRE-update parameters
        phi = self._regressor(transition.obs, transition.next_obs)
        tau = float(np.asarray(transition.action).reshape(-1)[0])
        innovation = np.array([tau - float(phi @ belief.mean)], dtype=np.float64)

        step = self._legacy.observe(transition.obs, transition.action,
                                    transition.next_obs)

        new = Belief(mean=self._legacy.params,
                     cov=self._legacy._rls.P.copy(),
                     space=self.representation.name,
                     extras={"names": self.representation.names})
        return new, StepRecord(
            prediction=np.zeros(0), target=np.zeros(0), error=np.zeros(0),
            innovation=innovation,
            innovation_space=TORQUE,
            loss=step.loss,               # legacy value, post-update residual^2
            seconds=step.update_seconds,
            extras=step.extras,
        )


__all__ = ["RLSEstimator"]
