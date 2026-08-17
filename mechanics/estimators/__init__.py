"""Estimators, as adapters over the existing algorithms.

Every class here DELEGATES to the implementation that produced the published
results -- ``GradientLatentAdaptor``, ``UKFLatentAdaptor``, ``RLSAdaptor``,
``dyn.rls_step``. Nothing is reimplemented. That is deliberate: it is what makes
the equivalence tests in ``mechanics/tests.py`` a real check rather than a hope,
and it keeps the refactor a refactor.

The adapters add exactly one thing: they surface the INNOVATION, which the legacy
classes computed internally and threw away.
"""

from mechanics.estimators.gradient import GradientEstimator
from mechanics.estimators.rls import RLSEstimator
from mechanics.estimators.static import StaticEstimator
from mechanics.estimators.ukf import UKFEstimator

__all__ = ["StaticEstimator", "GradientEstimator", "UKFEstimator", "RLSEstimator"]
