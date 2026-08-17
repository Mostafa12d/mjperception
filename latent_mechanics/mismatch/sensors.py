"""Sensor-level model mismatch: the plant is ideal, the measurements are not.

Applied to a STATE SEQUENCE, never to transitions directly: consecutive
transitions share a state, so perturbing both fields independently would give two
readings of the same instant and halve the effective noise.

    true state -> additive noise -> quantisation -> dropout (hold) -> latency

Only the state is corrupted; torque sensing stays ideal. Every estimator gets the
same stream and all are scored against the true next state, since scoring against
the noisy reading would measure the sensor instead of the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Joint travel of the door XMLs, used to size the quantiser.
JOINT_RANGE = (-0.17, 2.09)
JOINT_SPAN = JOINT_RANGE[1] - JOINT_RANGE[0]


@dataclass
class SensorPipeline:
    """A configurable observation channel."""

    theta_sigma: float = 0.0            # rad
    # rad/s; None derives sqrt(2)*sigma/dt, as differencing a noisy encoder would
    theta_dot_sigma: float | None = None
    quantize_bits: int | None = None    # encoder resolution over the joint span
    dropout_prob: float = 0.0           # lost samples hold the last good reading
    latency_steps: int = 0              # stale state, current action
    seed: int = 0

    def is_identity(self) -> bool:
        return (
            self.theta_sigma == 0.0
            and not self.theta_dot_sigma
            and self.quantize_bits is None
            and self.dropout_prob == 0.0
            and self.latency_steps == 0
        )

    def velocity_sigma(self, dt: float) -> float:
        if self.theta_dot_sigma is not None:
            return self.theta_dot_sigma
        return self.theta_sigma * np.sqrt(2.0) / dt

    def apply(self, states: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Corrupt a ``(T, 2)`` state sequence. Returns a new array."""
        if self.is_identity():
            return states.copy()
        obs = states.astype(np.float64).copy()

        if self.theta_sigma > 0:
            obs[:, 0] += rng.normal(0.0, self.theta_sigma, size=len(obs))
        v_sigma = self.velocity_sigma(dt)
        if v_sigma > 0:
            obs[:, 1] += rng.normal(0.0, v_sigma, size=len(obs))

        if self.quantize_bits is not None:
            step = JOINT_SPAN / (2**self.quantize_bits)
            obs[:, 0] = np.round(obs[:, 0] / step) * step
            # Velocity from a quantised encoder inherits the same grid, scaled
            # by the differencing interval.
            v_step = step / dt
            obs[:, 1] = np.round(obs[:, 1] / v_step) * v_step

        if self.dropout_prob > 0:
            lost = rng.random(len(obs)) < self.dropout_prob
            lost[0] = False  # never drop the first sample; nothing to hold
            last = obs[0].copy()
            for i in range(len(obs)):
                if lost[i]:
                    obs[i] = last
                else:
                    last = obs[i].copy()

        if self.latency_steps > 0:
            k = self.latency_steps
            obs = np.concatenate([np.repeat(obs[:1], k, axis=0), obs[:-k]], axis=0)

        return obs.astype(np.float32)

    def describe(self) -> dict:
        return {
            "theta_sigma": self.theta_sigma,
            "theta_dot_sigma": self.theta_dot_sigma,
            "quantize_bits": self.quantize_bits,
            "dropout_prob": self.dropout_prob,
            "latency_steps": self.latency_steps,
        }


def states_from_transitions(
    state: np.ndarray, next_state: np.ndarray
) -> np.ndarray:
    """Recover the ``(T+1, 2)`` state sequence a transition list came from. Only
    valid within a contiguous episode."""
    return np.concatenate([state, next_state[-1:]], axis=0)


def transitions_from_states(
    states: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``states_from_transitions``: ``(T+1, 2)`` -> state/next_state."""
    return states[:-1], states[1:]


__all__ = ["SensorPipeline", "states_from_transitions", "transitions_from_states"]
