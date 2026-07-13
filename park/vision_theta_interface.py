"""
Parked vision interface for door hinge angle θ̂(t).

DO NOT wire this into the dynamics estimator yet. Dynamics validation uses
oracle MuJoCo qpos/qvel/qacc (see run_door_dynamics_validation.py).

When revisiting vision, implement VisionThetaEstimator so that
run_door_dynamics_validation (or a sibling) can swap:

    theta      <- estimator.theta(t)      # instead of data.qpos
    theta_dot  <- filtered derivative      # NOT raw finite diff without care
    theta_ddot <- filtered second deriv

and keep τ from the F/T path unchanged. Ablate:
    oracle θ  |  vision θ̂  |  vision θ̂ + filter
on the same physical trial.

Recommended first implementation: ArUco/AprilTag on the door panel in MuJoCo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VisionThetaEstimator(ABC):
    """Drop-in source of hinge angle from camera features."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def update(self, rgb: np.ndarray, t: float) -> float:
        """Return θ̂ [rad] about the known/estimated hinge axis."""
        ...

    def available(self) -> bool:
        """False until the first successful feature lock."""
        return True


class OracleThetaEstimator(VisionThetaEstimator):
    """Placeholder that mirrors MuJoCo qpos — used only for API smoke tests."""

    def __init__(self):
        self._theta = 0.0

    def reset(self) -> None:
        self._theta = 0.0

    def update(self, rgb: np.ndarray, t: float) -> float:
        # rgb ignored; caller should set_theta from sim if using this stub
        return self._theta

    def set_theta(self, theta: float) -> None:
        self._theta = float(theta)


# Future: class ArucoDoorThetaEstimator(VisionThetaEstimator): ...
# Future: class DepthPlaneThetaEstimator(VisionThetaEstimator): ...
