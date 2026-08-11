"""
The RLS baseline, wrapped in the ``OnlineAdaptor`` interface.

This calls ``run_door_dynamics_validation.rls_init`` / ``rls_step`` unmodified.
Only the surrounding scaffolding is new: a regressor, a forward integrator so
the identified parameters can produce a *next-state prediction*, and the same
predict-then-update protocol the latent adaptor follows. That symmetry is the
whole point -- Experiment 3 drives both through identical code on identical data.

Making the comparison fair required three decisions, all of which favour RLS:

1. **A spring-aware regressor.** The baseline's own regressor is
   ``tau = I*thdd + mu*sign(thd) + b*thd``, which has no term for a torsional
   spring. 70% of the randomised doors have a door-closer spring, so that
   regressor is structurally misspecified for them through no fault of RLS. The
   default here is the 5-parameter version
   ``tau = I*thdd + mu*sign(thd) + b*thd + k*th + c``  (``c = -k*springref``),
   which spans the true dynamics exactly. ``n_params=3`` reproduces the
   baseline's own regressor for reference, and both are reported.

2. **Velocity gating.** Updates are skipped when ``|thd| <= vel_thresh``. At rest
   the equation of motion is an inequality (static friction absorbs whatever it
   needs), so those samples are not valid regression rows. The baseline gates the
   same way in ``moving_mask`` / ``run_door_adaptive_impedance``. The latent
   adaptor gets no equivalent help -- it updates on every transition.

3. **Sub-stepped integration.** Predictions integrate the identified ODE with the
   same 0.002 s substeps MuJoCo uses, not one coarse 0.02 s Euler step, so RLS is
   not charged for discretisation error the learned model never pays.

The remaining asymmetry is stated rather than hidden: acceleration is not
observable, so the regressor uses a finite difference of the observed velocities.
That is the honest choice for a 50 Hz stream, but it is noisier than the exact
``qacc`` the offline baseline script reads out of MuJoCo.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.online.adaptor import AdaptorStep, ArrayLike, OnlineAdaptor

# Priors matching the baseline's own initialisation in run_door_adaptive_impedance.
DEFAULT_I = 5.0
DEFAULT_MU = 3.0
DEFAULT_B = 0.2
I_MIN = 0.1  # guard: the baseline clamps I_hat the same way


class RLSAdaptor(OnlineAdaptor):
    """Recursive least squares on interpretable hinge parameters.

    Args:
        dt: model timestep of the observation stream (s).
        n_substeps: integration substeps used when predicting one model step.
        n_params: 5 for the spring-aware regressor, 3 for the baseline's own.
        lam: forgetting factor, passed straight to ``dyn.rls_init``.
        delta: initial covariance scale, passed to ``dyn.rls_init``.
        vel_thresh: skip updates below this |velocity|.
    """

    name = "rls"

    def __init__(
        self,
        dt: float,
        n_substeps: int = 10,
        n_params: int = 5,
        lam: float = 0.995,
        delta: float = 1e3,
        vel_thresh: float = 0.02,
        init: np.ndarray | None = None,
    ) -> None:
        if n_params not in (3, 5):
            raise ValueError("n_params must be 3 (baseline regressor) or 5 (spring-aware)")
        self.dt = float(dt)
        self.n_substeps = int(n_substeps)
        self.n_params = n_params
        self.lam = lam
        self.delta = delta
        self.vel_thresh = vel_thresh
        self._init = init
        self.name = f"rls{n_params}"
        self.reset(init)

    # -- belief ------------------------------------------------------------
    def reset(self, init: np.ndarray | None = None) -> None:
        self._rls = dyn.rls_init(self.n_params, delta=self.delta, lam=self.lam)
        prior = np.zeros(self.n_params)
        prior[:3] = [DEFAULT_I, DEFAULT_MU, DEFAULT_B]
        if init is not None:
            prior = np.asarray(init, dtype=float).reshape(self.n_params)
        self._rls.theta[:] = prior
        self._n_updates = 0
        self._n_skipped = 0

    @property
    def params(self) -> np.ndarray:
        return self._rls.theta.copy()

    @property
    def latent(self) -> np.ndarray:
        """The belief, for logging symmetry with the latent adaptor.

        Note these are physical parameters, not a latent -- they are not
        comparable to ``z`` and are never projected into the embedding PCA.
        """
        return self.params

    def belief(self) -> dict[str, Any]:
        names = ["I", "mu", "b", "k", "c"][: self.n_params]
        return {
            "mean": self.params,
            "cov": self._rls.P.copy(),
            "names": names,
            "trace_P": float(np.trace(self._rls.P)),
        }

    # -- forward model -----------------------------------------------------
    def _unpack(self) -> tuple[float, float, float, float, float]:
        th = self._rls.theta
        I = max(float(th[0]), I_MIN)  # guard against a non-physical early estimate
        mu = max(float(th[1]), 0.0)
        b = float(th[2])
        k = float(th[3]) if self.n_params == 5 else 0.0
        c = float(th[4]) if self.n_params == 5 else 0.0
        return I, mu, b, k, c

    def predict(self, state: ArrayLike, action: ArrayLike) -> np.ndarray:
        """Integrate the identified ODE forward one model step."""
        th, thd = (float(v) for v in np.asarray(state).reshape(-1)[:2])
        tau = float(np.asarray(action).reshape(-1)[0])
        I, mu, b, k, c = self._unpack()
        h = self.dt / self.n_substeps

        for _ in range(self.n_substeps):
            # Everything except inertia and Coulomb friction.
            net = tau - b * thd - k * th - c
            if abs(thd) <= 1e-9:
                # At rest: static friction holds unless the net torque exceeds it.
                if abs(net) <= mu:
                    acc = 0.0
                else:
                    acc = (net - mu * np.sign(net)) / I
            else:
                acc = (net - mu * np.sign(thd)) / I
            thd_new = thd + acc * h
            # Coulomb friction decelerates but must not reverse the motion
            # within a substep; without this guard the model chatters at rest.
            if thd != 0.0 and np.sign(thd_new) != np.sign(thd) and abs(net) <= mu:
                thd_new = 0.0
            thd = thd_new
            th = th + thd * h
        return np.array([th, thd], dtype=np.float32)

    # -- update ------------------------------------------------------------
    def observe(
        self, state: ArrayLike, action: ArrayLike, next_state: ArrayLike
    ) -> AdaptorStep:
        prediction = self.predict(state, action)
        s = np.asarray(state, dtype=float).reshape(-1)
        ns = np.asarray(next_state, dtype=float).reshape(-1)
        tau = float(np.asarray(action).reshape(-1)[0])
        th, thd = float(s[0]), float(s[1])
        thd_next = float(ns[1])

        t0 = time.perf_counter()
        skipped = abs(thd) <= self.vel_thresh
        if not skipped:
            # Acceleration is not observed; finite-difference the velocities.
            thdd = (thd_next - thd) / self.dt
            phi = [thdd, np.sign(thd), thd]
            if self.n_params == 5:
                phi += [th, 1.0]
            self._rls = dyn.rls_step(self._rls, np.array(phi), tau)
        else:
            self._n_skipped += 1
        dt_update = time.perf_counter() - t0

        self._n_updates += 1
        target = ns.astype(np.float32)
        resid = float(tau - np.dot(self._regressor(th, thd, thd_next), self._rls.theta))
        return AdaptorStep(
            prediction=prediction,
            target=target,
            error=prediction - target,
            loss=resid**2,
            latent=self.params,
            update_seconds=dt_update,
            extras={
                "skipped": bool(skipped),
                "trace_P": float(np.trace(self._rls.P)),
                "I_hat": float(self._rls.theta[0]),
            },
        )

    def _regressor(self, th: float, thd: float, thd_next: float) -> np.ndarray:
        thdd = (thd_next - thd) / self.dt
        phi = [thdd, np.sign(thd), thd]
        if self.n_params == 5:
            phi += [th, 1.0]
        return np.array(phi)


__all__ = ["RLSAdaptor"]
