"""The research loop. This file is the point of the whole refactor.

    belief = estimator.initialize()

    for transition in transitions:
        prediction    = predictor.predict(transition.obs, transition.action, belief)
        belief, record = estimator.update(belief, transition)

Five questions, answerable by reading the twelve lines of ``run`` below:

    What is observed?        transition.obs / next_obs, from the ObservationModel
    What is predicted?       predictor.predict(...) -> next observation
    What makes the residual? record.innovation, in record.innovation_space
    What is estimated?       belief.mean, in belief.space
    What updates it?         estimator.update(...)

Two protocol guarantees, both carried over from ``online.loop`` because they are
what make the numbers mean anything:

  * PREQUENTIAL. The prediction at step t uses the belief held BEFORE step t.
    Transitions are never shuffled, batched or revisited.
  * FROZEN PREDICTOR. Verified by checksum after the run, not assumed.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from mechanics.estimator import Estimator
from mechanics.predictor import Predictor
from mechanics.types import Belief, StepRecord, Trace, Transition


def run(
    estimator: Estimator,
    predictor: Predictor,
    transitions: Sequence[Transition],
    *,
    object_id: int = -1,
    boundaries: Sequence[int] | None = None,
    init_name: str = "",
    verify_frozen: bool = True,
    progress_every: int = 0,
) -> Trace:
    """Drive one estimator over one stream of transitions."""
    if not transitions:
        raise ValueError("empty transition stream")

    belief: Belief = estimator.initialize()
    records: list[StepRecord] = []
    beliefs: list[np.ndarray] = []

    for i, tr in enumerate(transitions):
        # 1. predict, using the belief held BEFORE this transition
        prediction = np.asarray(predictor.predict(tr.obs, tr.action, belief))

        # 2. update. The estimator forms its own residual, in its own space.
        t0 = time.perf_counter()
        belief, rec = estimator.update(belief, tr)
        elapsed = time.perf_counter() - t0

        # 3. record. Scored against clean truth when the stream carries it, so
        #    the metric measures the estimator and not the sensor.
        target = np.asarray(tr.target, dtype=prediction.dtype).reshape(-1)
        records.append(StepRecord(
            prediction=prediction,
            target=target,
            error=prediction - target,
            innovation=np.asarray(rec.innovation).reshape(-1),
            innovation_space=rec.innovation_space,
            loss=rec.loss,
            seconds=rec.seconds if rec.seconds else elapsed,
            extras=rec.extras,
        ))
        beliefs.append(belief.mean.copy())

        if progress_every and (i + 1) % progress_every == 0:
            recent = np.array([r.error[0] for r in records[-progress_every:]])
            print(f"    step {i + 1:6d}/{len(transitions)}  "
                  f"rolling angle RMSE {np.sqrt((recent ** 2).mean()):.3e}")

    if verify_frozen and hasattr(predictor, "assert_unchanged"):
        predictor.assert_unchanged()

    return Trace(
        name=estimator.name,
        object_id=object_id,
        error=np.stack([r.error for r in records]),
        innovation=np.stack([r.innovation for r in records]),
        innovation_space=records[0].innovation_space,
        loss=np.array([r.loss for r in records], dtype=float),
        beliefs=np.stack(beliefs),
        belief_space=belief.space,
        seconds=np.array([r.seconds for r in records], dtype=float),
        boundaries=list(boundaries or []),
        extras=_collect_extras(records),
        init_name=init_name,
    )


def _collect_extras(records: list[StepRecord]) -> dict[str, np.ndarray]:
    """Gather scalar diagnostics into per-key arrays.

    Unlike the old ``AdaptationLog``, a non-scalar extra is reported rather than
    silently dropped -- a filter that hides data is worse than one that complains.
    """
    keys = set().union(*[set(r.extras) for r in records]) if records else set()
    out: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for k in sorted(keys):
        vals = [r.extras.get(k, np.nan) for r in records]
        if all(isinstance(v, (int, float, bool, np.floating, np.integer)) for v in vals):
            out[k] = np.array(vals, dtype=float)
        else:
            skipped.append(k)
    if skipped:
        out["_non_scalar_keys"] = np.array([len(skipped)], dtype=float)
        print(f"  note: non-scalar diagnostics not aggregated: {skipped}")
    return out


__all__ = ["run"]
