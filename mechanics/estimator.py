"""The online estimator: what turns an innovation into a revised belief.

One protocol, two methods. Everything the brief lists as a candidate -- gradient
descent, RLS, EKF, UKF, particle filter, a learned GRU update -- fits behind it,
because it says nothing about how the belief is represented or what space the
residual lives in.

    initialize()                 -> Belief
    update(belief, transition)   -> (Belief, StepRecord)

``update`` is deliberately NOT handed the reported prediction. The estimator forms
its own residual, in its own space, and declares that space in the ``StepRecord``.
That is the honest description of what the existing three estimators do (see
CURRENT_SYSTEM.md B.6) and pretending otherwise would be an algorithmic change
disguised as a refactor.

Estimators are stateful across a run and are constructed per object.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mechanics.types import Belief, StepRecord, Transition


@runtime_checkable
class Estimator(Protocol):
    """A streaming estimator of an object's mechanics."""

    name: str

    def initialize(self) -> Belief:
        """The prior. Called once per object, before any data."""

    def update(self, belief: Belief, transition: Transition) -> tuple[Belief, StepRecord]:
        """Fold one observed transition into the belief.

        Must not read ``transition.truth`` -- that is the clean state, kept for
        scoring only. Must not modify the predictor's parameters.
        """


__all__ = ["Estimator"]
