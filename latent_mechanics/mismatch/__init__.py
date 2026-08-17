"""Stage 3: robustness to model mismatch.

Violations of the assumed dynamics, applied one mechanism at a time. The learned
model is not retrained and RLS keeps its regressor: both hold their original
assumptions while the plant stops obeying them.

    python3.10 -m latent_mechanics.mismatch.study
    python3.10 -m latent_mechanics.mismatch.tests
"""

from latent_mechanics.mismatch.config import StudyConfig, Sweep, default_sweeps
from latent_mechanics.mismatch.perturbations import (
    NonlinearCompliance,
    ParameterDrift,
    PlantPerturbation,
    PositionDependentFriction,
    StribeckFriction,
    build_perturbation,
)
from latent_mechanics.mismatch.sensors import SensorPipeline
from latent_mechanics.mismatch.simulate import simulate_perturbed, verify_matches_baseline
from latent_mechanics.mismatch.streams import DoorStream, build_door_stream

__all__ = [
    "PlantPerturbation", "StribeckFriction", "PositionDependentFriction",
    "NonlinearCompliance", "ParameterDrift", "build_perturbation",
    "SensorPipeline", "simulate_perturbed", "verify_matches_baseline",
    "DoorStream", "build_door_stream", "Sweep", "StudyConfig", "default_sweeps",
]
