"""One scoring implementation, replacing four near-identical copies.

The audit found tail-quarter normalised RMSE implemented separately in
``mechanisms/study.py``, ``mismatch/study.py``, ``curriculum/study.py`` and
``belief/sweep.py``. The definitions agreed, but nothing guaranteed they would
stay agreeing, and two of them differed in how they computed the motion scale.

The two conventions worth stating out loud, both inherited unchanged:

  * NORMALISE BY TRUE MOTION. Raw RMSE is meaningless across families, and a
    perturbation that merely slows the mechanism down would otherwise look like an
    improvement. ``nrmse = 1.0`` means "no better than predicting nothing changes".
  * SCORE THE TAIL. The converged belief is what the experiment is about, so the
    headline number is over the final quarter of the stream.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from mechanics.types import Trace, Transition

TAIL_FRAC = 0.25


def motion_scale(transitions: Sequence[Transition], first: int | None = None) -> np.ndarray:
    """RMS true one-step change per observation dimension."""
    tgt = np.stack([t.target for t in transitions[first:]])
    obs = np.stack([np.asarray(t.obs).reshape(-1) for t in transitions[first:]])
    d = tgt - obs[:, : tgt.shape[1]]
    return np.maximum(np.sqrt(np.mean(d ** 2, axis=0)), 1e-12) if len(d) else np.ones(tgt.shape[1])


def nrmse(trace: Trace, transitions: Sequence[Transition], dim: int = 0,
          tail_frac: float | None = None) -> float:
    """Normalised RMSE over the tail. ``tail_frac=None`` scores the whole stream."""
    n = len(trace)
    first = 0 if tail_frac is None else n - max(1, int(n * tail_frac))
    err = trace.error[first:, dim]
    scale = motion_scale(transitions, first)[dim]
    return float(np.sqrt(np.mean(err ** 2)) / scale)


def score(trace: Trace, transitions: Sequence[Transition],
          reference: float | None = None) -> dict[str, float]:
    """The standard metric block every experiment reports."""
    out = {
        "n_steps": len(trace),
        "angle_rmse": trace.rmse(0),
        "angle_rmse_final": trace.final_rmse(0, TAIL_FRAC),
        "vel_rmse_final": trace.final_rmse(1, TAIL_FRAC) if trace.error.shape[1] > 1 else float("nan"),
        "angle_nrmse": nrmse(trace, transitions, 0, None),
        "angle_nrmse_final": nrmse(trace, transitions, 0, TAIL_FRAC),
        "us_per_update": 1e6 * trace.seconds_per_update,
        "belief_travel": trace.belief_travel,
        "belief_space": trace.belief_space,
        "innovation_space": trace.innovation_space,
        "innovation_rms": float(np.sqrt(np.mean(trace.innovation ** 2))),
    }
    if reference is not None and reference > 0:
        out["ratio_to_reference"] = out["angle_nrmse_final"] / reference
    return out


def steps_to_converge(trace: Trace, threshold: float, window: int = 200) -> int:
    """Steps until the rolling RMSE settles below ``threshold``; ``-1`` if never.
    ``-1`` rather than ``None`` so the column survives a CSV round-trip."""
    s = trace.steps_to(threshold, 0, window)
    return -1 if s is None else s


def belief_drift(trace: Trace, tail_frac: float = TAIL_FRAC) -> float:
    """Mean per-step belief movement over the tail, relative to total travel.
    High means the estimate is chattering rather than converging."""
    n = max(1, int(len(trace) * tail_frac))
    tail = trace.beliefs[-n:]
    if len(tail) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(tail, axis=0), axis=1).mean()
    total = max(trace.belief_travel, 1e-12)
    return float(steps / total)


def write_csv(path: str | Path, rows: list[dict]) -> Path | None:
    """The single CSV writer. Four copies of this existed."""
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


__all__ = ["score", "nrmse", "motion_scale", "steps_to_converge",
           "belief_drift", "write_csv", "TAIL_FRAC"]
