"""Plant-level model mismatch: physics the estimators do not know about.

Each perturbation contributes an extra hinge torque and/or mutates MuJoCo model
parameters during a rollout; an empty list reproduces the ideal plant exactly.
The recorded action never includes the perturbation, and each class holds one
mechanism, so an effect attributes to one violated assumption.

Against the RLS regressor ``tau = I*thdd + mu*sign(thd) + b*thd + k*th + c``:

  StribeckFriction           friction nonlinear in |thd|
  PositionDependentFriction  friction depends on th, which no term captures
  NonlinearCompliance        elastic torque cubic in th, not linear
  ParameterDrift             I, mu, b, k are no longer constants
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco
import numpy as np


class PlantPerturbation(ABC):
    """One source of model mismatch."""

    name: str = "perturbation"

    def reset(self, model: mujoco.MjModel) -> None:
        """Capture nominal model values before a rollout begins."""

    def extra_torque(self, t: float, theta: float, theta_dot: float) -> float:
        """Unmodelled hinge torque applied on top of the commanded one."""
        return 0.0

    def update_model(self, t: float, model: mujoco.MjModel) -> None:
        """Mutate model parameters in place (for time-varying dynamics)."""

    def describe(self) -> dict:
        return {"name": self.name}


def smooth_sign(x: float, width: float) -> float:
    """``tanh(x/width)``: a smooth ``sign``, so an external torque does not make the
    solver chatter near rest. Agrees with ``sign`` wherever the door is moving."""
    return float(np.tanh(x / max(width, 1e-12)))


@dataclass
class StribeckFriction(PlantPerturbation):
    """Extra static friction that decays as the joint starts to slide.

        tau_extra = -(mu_s - mu_c) * exp(-(|thd| / v_s)^delta) * sign(thd)

    MuJoCo's ``frictionloss`` already gives the Coulomb level, so this adds only
    the breakaway excess near zero velocity. The dependence on ``v_s`` cannot be
    a fixed regressor column, so no RLS reparameterisation absorbs it.
    """

    excess: float = 1.5  # mu_s - mu_c, N*m
    v_stribeck: float = 0.05  # rad/s
    exponent: float = 2.0
    smooth_width: float = 1e-3
    name: str = "stribeck"

    def extra_torque(self, t: float, theta: float, theta_dot: float) -> float:
        decay = np.exp(-((abs(theta_dot) / self.v_stribeck) ** self.exponent))
        return -self.excess * decay * smooth_sign(theta_dot, self.smooth_width)

    def describe(self) -> dict:
        return {"name": self.name, "excess": self.excess, "v_stribeck": self.v_stribeck}


@dataclass
class PositionDependentFriction(PlantPerturbation):
    """A worn or contaminated bearing: friction varying with door angle.

        tau_extra = -amplitude * sin(2*pi*theta/period + phase) * sign(thd)

    Invisible to any constant-times-``sign(thd)`` friction term, and unlike
    Stribeck it persists at all speeds.
    """

    amplitude: float = 1.0  # N*m
    period: float = 0.8  # rad
    phase: float = 0.0
    smooth_width: float = 1e-3
    name: str = "position_friction"

    def extra_torque(self, t: float, theta: float, theta_dot: float) -> float:
        mod = np.sin(2 * np.pi * theta / self.period + self.phase)
        return -self.amplitude * mod * smooth_sign(theta_dot, self.smooth_width)

    def describe(self) -> dict:
        return {"name": self.name, "amplitude": self.amplitude, "period": self.period}


@dataclass
class NonlinearCompliance(PlantPerturbation):
    """Hinge elasticity that is not a linear spring.

        tau_extra = -k3 * theta^3  -  k_seal * theta * exp(-(theta/width)^2)

    A hardening element plus a seal or latch detent near closed. Both are 1-DOF,
    so the state stays Markov and the learned model could represent them. The
    linear part is left to MuJoCo's ``jnt_stiffness``, which RLS already models,
    so only the nonlinear excess is mismatch.
    """

    k_cubic: float = 2.0  # N*m/rad^3
    seal_gain: float = 0.0  # N*m/rad
    seal_width: float = 0.15  # rad
    name: str = "compliance"

    def extra_torque(self, t: float, theta: float, theta_dot: float) -> float:
        tau = -self.k_cubic * theta**3
        if self.seal_gain:
            tau -= self.seal_gain * theta * np.exp(-((theta / self.seal_width) ** 2))
        return float(tau)

    def describe(self) -> dict:
        return {"name": self.name, "k_cubic": self.k_cubic, "seal_gain": self.seal_gain}


@dataclass
class ParameterDrift(PlantPerturbation):
    """Slowly time-varying mechanics: the door warms up, dries out, or wears.

    Scales friction / damping / stiffness over the episode. ``mode`` is
    ``linear`` (1 + rate*t), ``ramp`` (held after ``t_hold``) or ``sine``.

    The one perturbation violating stationarity rather than functional form. RLS
    is not defenceless: its forgetting factor exists to track drift.
    """

    friction_rate: float = 0.0
    damping_rate: float = 0.0
    stiffness_rate: float = 0.0
    mode: str = "linear"
    frequency: float = 0.1  # Hz, for mode="sine"
    t_hold: float = np.inf
    name: str = "drift"

    def __post_init__(self) -> None:
        self._nominal: dict[str, float] = {}

    def reset(self, model: mujoco.MjModel) -> None:
        jid = model.joint("hinge").id
        self._jid = jid
        self._nominal = {
            "frictionloss": float(model.dof_frictionloss[0]),
            "damping": float(model.dof_damping[0]),
            "stiffness": float(model.jnt_stiffness[jid]),
        }

    def _factor(self, t: float, rate: float) -> float:
        if rate == 0.0:
            return 1.0
        if self.mode == "sine":
            return 1.0 + rate * float(np.sin(2 * np.pi * self.frequency * t))
        tt = min(t, self.t_hold)
        return 1.0 + rate * tt

    def update_model(self, t: float, model: mujoco.MjModel) -> None:
        if not self._nominal:
            self.reset(model)
        n = self._nominal
        # Clamped at zero: negative friction or damping would inject energy and
        # make the rollout diverge rather than merely drift.
        model.dof_frictionloss[0] = max(
            0.0, n["frictionloss"] * self._factor(t, self.friction_rate)
        )
        model.dof_damping[0] = max(0.0, n["damping"] * self._factor(t, self.damping_rate))
        model.jnt_stiffness[self._jid] = max(
            0.0, n["stiffness"] * self._factor(t, self.stiffness_rate)
        )

    def current_params(self, t: float) -> dict[str, float]:
        """True parameter values at time ``t`` -- ground truth for belief plots."""
        n = self._nominal
        return {
            "frictionloss": n["frictionloss"] * self._factor(t, self.friction_rate),
            "damping": n["damping"] * self._factor(t, self.damping_rate),
            "stiffness": n["stiffness"] * self._factor(t, self.stiffness_rate),
        }

    def describe(self) -> dict:
        return {
            "name": self.name, "mode": self.mode,
            "friction_rate": self.friction_rate,
            "damping_rate": self.damping_rate,
            "stiffness_rate": self.stiffness_rate,
        }


PERTURBATION_TYPES = {
    "stribeck": StribeckFriction,
    "position_friction": PositionDependentFriction,
    "compliance": NonlinearCompliance,
    "drift": ParameterDrift,
}


def build_perturbation(kind: str, **kwargs) -> PlantPerturbation:
    if kind not in PERTURBATION_TYPES:
        raise ValueError(
            f"unknown perturbation {kind!r}; choose from {sorted(PERTURBATION_TYPES)}"
        )
    return PERTURBATION_TYPES[kind](**kwargs)


__all__ = [
    "PlantPerturbation",
    "StribeckFriction",
    "PositionDependentFriction",
    "NonlinearCompliance",
    "ParameterDrift",
    "build_perturbation",
    "PERTURBATION_TYPES",
    "smooth_sign",
]
