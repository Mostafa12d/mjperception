"""
Sensor-level model mismatch: the plant is ideal, the measurements are not.

Applied to a *state sequence*, never to transitions directly. This matters and
is easy to get wrong: consecutive transitions share a state -- the ``next_state``
of transition ``t`` is the ``state`` of transition ``t+1`` -- so perturbing the
two fields independently would give the robot two different readings of the same
instant and halve the effective noise through averaging. Corrupting the
trajectory once and then rebuilding transitions from it is what a real encoder
does.

Pipeline order, matching a real acquisition chain:

    true state -> additive noise -> quantisation -> dropout (hold) -> latency

Only the *state* is corrupted. Torque sensing is left ideal so that each
experiment isolates one mechanism; adding actuation noise would be a separate
sweep.

Both estimators receive exactly the same corrupted stream, and both are scored
against the true next state rather than the noisy one. Scoring against the noisy
reading would impose a floor of sigma on everybody and measure the sensor
instead of the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Joint travel of the door XMLs, used to size the quantiser.
JOINT_RANGE = (-0.17, 2.09)
JOINT_SPAN = JOINT_RANGE[1] - JOINT_RANGE[0]


@dataclass
class SensorPipeline:
    """A configurable observation channel.

    Args:
        theta_sigma: Gaussian noise on the angle [rad].
        theta_dot_sigma: Gaussian noise on the velocity [rad/s]. If ``None``,
            it is *derived* from ``theta_sigma`` as ``sqrt(2)*sigma/dt``, which
            is what you get when velocity is obtained by differencing a noisy
            encoder -- the realistic coupling, and a punishing one: at dt=0.02
            it amplifies position noise about 70x.
        quantize_bits: encoder resolution over the joint span. ``None`` disables.
        dropout_prob: probability that a sample is lost; the last good reading is
            held, as a real driver would.
        latency_steps: whole-model-step delay on the state, with the action
            remaining current -- the estimator reasons about a stale pose.
        seed: per-stream RNG seed.
    """

    theta_sigma: float = 0.0
    theta_dot_sigma: float | None = None
    quantize_bits: int | None = None
    dropout_prob: float = 0.0
    latency_steps: int = 0
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
    """Recover the ``(T+1, 2)`` state sequence a transition list came from.

    Only valid within a contiguous episode, where ``next_state[i] == state[i+1]``.
    """
    return np.concatenate([state, next_state[-1:]], axis=0)


def transitions_from_states(
    states: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``states_from_transitions``: ``(T+1, 2)`` -> state/next_state."""
    return states[:-1], states[1:]


__all__ = ["SensorPipeline", "states_from_transitions", "transitions_from_states"]
