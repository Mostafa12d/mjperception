"""What the robot can actually measure.

This is the component the old code did not have. Observations were produced by
slicing MuJoCo's ``qpos``/``qvel`` straight out of the simulator log, so there was
no seam at which to insert noise, quantisation, a camera, or a force sensor.
``SensorPipeline`` existed but lived inside ``mismatch/``, applied only to the
door, and hardcoded the door's joint span into its quantiser.

An observation model maps a clean STATE SEQUENCE to an observation sequence. It
is applied to the sequence rather than to transitions, because consecutive
transitions share a state: perturbing ``state`` and ``next_state`` independently
would give two readings of the same instant and halve the effective noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from latent_mechanics.mismatch.sensors import SensorPipeline


@runtime_checkable
class ObservationModel(Protocol):
    """Clean state sequence -> what the estimator gets to see."""

    name: str

    def observe(self, states: np.ndarray, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        """``(T, n_state)`` -> ``(T, n_obs)``. Must not mutate ``states``."""

    def describe(self) -> dict:
        """Everything needed to reproduce this channel."""


@dataclass
class IdentityObservation:
    """Perfect proprioception: the observation IS the state.

    What every experiment outside ``mismatch/`` currently assumes, now stated
    rather than implied.
    """

    name: str = "identity"

    def observe(self, states: np.ndarray, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        return np.asarray(states).copy()

    def describe(self) -> dict:
        return {"name": self.name}


@dataclass
class JointSensor:
    """Noisy / quantised / dropped / delayed joint readings.

    Wraps the existing ``mismatch.sensors.SensorPipeline`` so Stage-3 numbers are
    reproduced exactly, with one fix: the quantiser's full-scale span is a
    parameter instead of the door's hardcoded ``2.26 rad``. Leave ``joint_span``
    at ``None`` to keep the historical door behaviour bit-for-bit.

        true state -> additive noise -> quantisation -> dropout (hold) -> latency
    """

    theta_sigma: float = 0.0
    theta_dot_sigma: float | None = None
    quantize_bits: int | None = None
    dropout_prob: float = 0.0
    latency_steps: int = 0
    joint_span: float | None = None   # None -> the door's span, as Stage 3 used
    name: str = "joint_sensor"

    def _pipeline(self) -> SensorPipeline:
        return SensorPipeline(
            theta_sigma=self.theta_sigma,
            theta_dot_sigma=self.theta_dot_sigma,
            quantize_bits=self.quantize_bits,
            dropout_prob=self.dropout_prob,
            latency_steps=self.latency_steps,
        )

    def observe(self, states: np.ndarray, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        states = np.asarray(states)
        if self.joint_span is None or self.quantize_bits is None:
            return self._pipeline().apply(states, dt, rng)

        # Same pipeline, but quantise against this mechanism's own travel. Done
        # here rather than by editing SensorPipeline so Stage-3 stays untouched.
        noise_only = SensorPipeline(
            theta_sigma=self.theta_sigma,
            theta_dot_sigma=self.theta_dot_sigma,
            quantize_bits=None,
            dropout_prob=0.0,
            latency_steps=0,
        )
        obs = noise_only.apply(states, dt, rng).astype(np.float64)
        step = self.joint_span / (2 ** self.quantize_bits)
        obs[:, 0] = np.round(obs[:, 0] / step) * step
        obs[:, 1] = np.round(obs[:, 1] / (step / dt)) * (step / dt)

        tail = SensorPipeline(
            theta_sigma=0.0, theta_dot_sigma=0.0, quantize_bits=None,
            dropout_prob=self.dropout_prob, latency_steps=self.latency_steps,
        )
        return tail.apply(obs.astype(np.float32), dt, rng)

    def describe(self) -> dict:
        return {"name": self.name, "joint_span": self.joint_span,
                **self._pipeline().describe()}


@dataclass
class PartialObservation:
    """Keep only some observation channels -- e.g. angle without velocity.

    The honest way to ask "what if we cannot measure velocity?". Dropped channels
    are removed, not zeroed, so a predictor expecting them fails loudly.
    """

    keep: tuple[int, ...] = (0,)
    inner: ObservationModel = field(default_factory=IdentityObservation)
    name: str = "partial"

    def observe(self, states: np.ndarray, dt: float,
                rng: np.random.Generator) -> np.ndarray:
        return self.inner.observe(states, dt, rng)[:, list(self.keep)]

    def describe(self) -> dict:
        return {"name": self.name, "keep": list(self.keep),
                "inner": self.inner.describe()}


def apply_to_sequence(
    model: ObservationModel,
    states: np.ndarray,
    next_states: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a transition list correctly: rebuild the ``(T+1,)`` state sequence,
    observe it once, then re-split. Returns ``(obs, next_obs)``.

    This is the invariant from ``mismatch/streams.py`` promoted to a shared helper,
    because getting it wrong halves the effective noise and is invisible in the
    output.
    """
    seq = np.concatenate([states, next_states[-1:]], axis=0)
    obs_seq = model.observe(seq, dt, rng)
    return obs_seq[:-1], obs_seq[1:]


__all__ = ["ObservationModel", "IdentityObservation", "JointSensor",
           "PartialObservation", "apply_to_sequence"]
