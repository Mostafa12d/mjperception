"""The minimal experimental core for online mechanics estimation.

    Plant -> ObservationModel -> Transition
                                    |
              Predictor.predict(obs, action, belief) -> predicted next observation
                                    |
              Estimator.update(belief, transition)   -> Belief + innovation

Four things are swappable independently, which is the whole point:

    predictor        mechanics/predictor.py       learned | analytical | misspecified
    observation      mechanics/observation.py     identity | noisy | partial
    representation   mechanics/representation.py  full latent | reduced chart | physical
    estimator        mechanics/estimators/        static | gradient | UKF | RLS

See CURRENT_SYSTEM.md for what this replaces and REFACTOR_PROPOSAL.md for why.
"""

from mechanics.build import Method, MethodConfig, Workspace, build_method
from mechanics.data import transitions_from_dataset
from mechanics.estimator import Estimator
from mechanics.loop import run
from mechanics.observation import (
    IdentityObservation,
    JointSensor,
    ObservationModel,
    PartialObservation,
)
from mechanics.predictor import (
    AnalyticalPredictor,
    LatentNetworkPredictor,
    MisspecifiedPredictor,
    Predictor,
)
from mechanics.representation import (
    FullLatent,
    Hybrid,
    MechanicsRepresentation,
    PhysicalParameters,
    ReducedLatent,
)
from mechanics.types import Belief, StepRecord, Trace, Transition

__all__ = [
    # loop
    "run",
    # types
    "Transition", "Belief", "StepRecord", "Trace",
    # components
    "Predictor", "LatentNetworkPredictor", "AnalyticalPredictor", "MisspecifiedPredictor",
    "ObservationModel", "IdentityObservation", "JointSensor", "PartialObservation",
    "MechanicsRepresentation", "FullLatent", "ReducedLatent", "PhysicalParameters", "Hybrid",
    "Estimator",
    # wiring
    "Workspace", "MethodConfig", "Method", "build_method", "transitions_from_dataset",
]
