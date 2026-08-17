"""The dynamics predictor: ``(observation, action, belief) -> predicted next observation``.

Three implementations, covering the three predictor experiments the brief names:

    LatentNetworkPredictor    learned neural dynamics (the frozen MLP)
    AnalyticalPredictor       the identified ODE -- what RLS integrates
    MisspecifiedPredictor     a deliberately wrong predictor, for control experiments

A predictor OWNS its mechanics representation, because converting belief
coordinates into "whatever I consume" is the predictor's business and nobody
else's. That is what keeps the driver ignorant of latents, charts and parameter
vectors alike.

Latent estimators additionally need the predictor as a MEASUREMENT FUNCTION -- the
UKF pushes sigma points through it. That is the ``MeasurementPredictor`` protocol
below. It is an extra capability, not a requirement: an estimator that only needs
one-step predictions works with any ``Predictor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from latent_mechanics.model import MechanicsDynamicsModel
from mechanics.representation import MechanicsRepresentation
from mechanics.types import NORMALISED_DELTA, OBSERVATION, Belief


@runtime_checkable
class Predictor(Protocol):
    """One-step-ahead prediction under a mechanics belief."""

    name: str
    representation: MechanicsRepresentation

    def predict(self, obs: np.ndarray, action: np.ndarray,
                belief: Belief) -> np.ndarray:
        """Predicted next observation, raw SI units. Must not mutate the belief."""


@runtime_checkable
class MeasurementPredictor(Predictor, Protocol):
    """A predictor usable as a filter's measurement function ``h``.

    ``measurement_space`` names the space ``measurement``/``predict_measurement``
    work in. It need not be the observation space -- the UKF deliberately works in
    normalised-delta space, because that is the only space in which one ``R`` is
    meaningful across mechanism families.
    """

    measurement_space: str

    def measurement(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """The measurement ``y`` extracted from an observed transition."""

    def predict_measurement(self, obs: np.ndarray, action: np.ndarray,
                            xs: np.ndarray) -> np.ndarray:
        """``h(x)`` for a BATCH of belief coordinates ``(N, d) -> (N, k)``.
        Batched because ``h`` is a network and per-point calls dominate the cost."""


# --------------------------------------------------------------------------
# learned neural dynamics
# --------------------------------------------------------------------------

@dataclass
class LatentNetworkPredictor:
    """The frozen Stage-1 MLP, ``f(s, a, z) -> s'``.

    The network is frozen at construction and the freeze is verified, not assumed:
    a parameter checksum is taken and ``assert_unchanged()`` re-checks it, catching
    what ``requires_grad=False`` does not (stray optimisers, in-place writes).
    """

    model: MechanicsDynamicsModel
    representation: MechanicsRepresentation
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    name: str = "latent_network"
    measurement_space: str = NORMALISED_DELTA

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.model = self.model.to(self.device)
        self.model.freeze()
        live = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(
                f"predictor network is not frozen; {len(live)} tensor(s) still "
                f"require grad, e.g. {live[:3]}")
        self._checksum0 = self._checksum()

    # -- frozen-network guarantee (preserved from OnlineLatentAdaptor) -----
    def _checksum(self) -> float:
        with torch.no_grad():
            return float(sum(p.double().abs().sum() for p in self.model.parameters()))

    def assert_unchanged(self, tol: float = 0.0) -> None:
        now = self._checksum()
        if abs(now - self._checksum0) > tol:
            raise RuntimeError(
                f"network weights changed during adaptation "
                f"({self._checksum0!r} -> {now!r})")
        leaked = [n for n, p in self.model.named_parameters() if p.grad is not None]
        if leaked:
            raise RuntimeError(f"gradients accumulated on frozen weights: {leaked[:3]}")

    # -- tensor plumbing ---------------------------------------------------
    def _t(self, x, dim: int) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(x, dtype=np.float32), device=self.device
        ).reshape(1, dim)

    def _z(self, x: np.ndarray, n: int = 1) -> torch.Tensor:
        z = np.atleast_2d(self.representation.to_predictor(x))
        return torch.as_tensor(np.asarray(z, dtype=np.float32),
                               device=self.device).reshape(n, -1)

    # -- Predictor ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs: np.ndarray, action: np.ndarray,
                belief: Belief) -> np.ndarray:
        s = self._t(obs, self.model.state_dim)
        a = self._t(action, self.model.action_dim)
        return self.model(s, a, self._z(belief.mean))[0].cpu().numpy()

    # -- MeasurementPredictor ----------------------------------------------
    @torch.no_grad()
    def measurement(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """``normalise(next_obs - obs)`` -- the target the network regresses."""
        s = self._t(obs, self.model.state_dim)
        ns = self._t(next_obs, self.model.state_dim)
        return self.model.target(s, ns)[0].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_measurement(self, obs: np.ndarray, action: np.ndarray,
                            xs: np.ndarray) -> np.ndarray:
        """All sigma points through the network in one batched forward pass."""
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        n = xs.shape[0]
        s = self._t(obs, self.model.state_dim).expand(n, -1)
        a = self._t(action, self.model.action_dim).expand(n, -1)
        out = self.model.raw_output(s, a, self._z(xs, n))
        return out.cpu().numpy().astype(np.float64)

    @property
    def obs_dim(self) -> int:
        return int(self.model.state_dim)


# --------------------------------------------------------------------------
# analytical dynamics
# --------------------------------------------------------------------------

@dataclass
class AnalyticalPredictor:
    """The identified hinge ODE, integrated forward one model step.

        tau = I*qddot + mu*sign(qdot) + b*qdot + k*q + c

    Extracted verbatim from ``RLSAdaptor.predict`` -- same sub-stepping, same
    at-rest stiction branch, same anti-chatter guard -- so RLS keeps behaving
    exactly as it did, and so "analytical dynamics" becomes a predictor you can
    hand to any estimator rather than something welded inside one.
    """

    dt: float
    representation: MechanicsRepresentation
    n_substeps: int = 10
    i_min: float = 0.1            # guard a non-physical early inertia estimate
    name: str = "analytical"
    measurement_space: str = OBSERVATION

    def _unpack(self, x: np.ndarray) -> tuple[float, float, float, float, float]:
        p = np.asarray(self.representation.to_predictor(x), dtype=float).reshape(-1)
        I = max(float(p[0]), self.i_min)
        mu = max(float(p[1]), 0.0)
        b = float(p[2])
        k = float(p[3]) if len(p) >= 5 else 0.0
        c = float(p[4]) if len(p) >= 5 else 0.0
        return I, mu, b, k, c

    def predict(self, obs: np.ndarray, action: np.ndarray,
                belief: Belief) -> np.ndarray:
        th, thd = (float(v) for v in np.asarray(obs).reshape(-1)[:2])
        tau = float(np.asarray(action).reshape(-1)[0])
        I, mu, b, k, c = self._unpack(belief.mean)
        h = self.dt / self.n_substeps

        for _ in range(self.n_substeps):
            net = tau - b * thd - k * th - c
            if abs(thd) <= 1e-9:
                # at rest, static friction holds unless net torque exceeds it
                acc = 0.0 if abs(net) <= mu else (net - mu * np.sign(net)) / I
            else:
                acc = (net - mu * np.sign(thd)) / I
            thd_new = thd + acc * h
            # friction may decelerate but not reverse within a substep
            if thd != 0.0 and np.sign(thd_new) != np.sign(thd) and abs(net) <= mu:
                thd_new = 0.0
            thd = thd_new
            th = th + thd * h
        return np.array([th, thd], dtype=np.float32)

    def measurement(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        return np.asarray(next_obs, dtype=np.float64).reshape(-1)

    def predict_measurement(self, obs: np.ndarray, action: np.ndarray,
                            xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        return np.stack([
            self.predict(obs, action, Belief(mean=x)) for x in xs
        ]).astype(np.float64)


# --------------------------------------------------------------------------
# deliberately wrong dynamics
# --------------------------------------------------------------------------

@dataclass
class MisspecifiedPredictor:
    """A predictor distorted on purpose, for "what if the model is wrong?" runs.

    The distortion is applied to the predicted DELTA, so ``gain=1, bias=0`` is
    exactly the wrapped predictor and any deviation is attributable.
    """

    inner: Predictor
    gain: np.ndarray | float = 1.0
    bias: np.ndarray | float = 0.0
    name: str = "misspecified"

    @property
    def representation(self) -> MechanicsRepresentation:
        return self.inner.representation

    @property
    def measurement_space(self) -> str:
        return getattr(self.inner, "measurement_space", OBSERVATION)

    def predict(self, obs: np.ndarray, action: np.ndarray,
                belief: Belief) -> np.ndarray:
        base = self.inner.predict(obs, action, belief)
        o = np.asarray(obs, dtype=base.dtype).reshape(-1)[: len(base)]
        return (o + self.gain * (base - o) + self.bias).astype(base.dtype)

    def measurement(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        return self.inner.measurement(obs, next_obs)

    def predict_measurement(self, obs: np.ndarray, action: np.ndarray,
                            xs: np.ndarray) -> np.ndarray:
        return self.inner.predict_measurement(obs, action, xs)


__all__ = ["Predictor", "MeasurementPredictor", "LatentNetworkPredictor",
           "AnalyticalPredictor", "MisspecifiedPredictor"]
