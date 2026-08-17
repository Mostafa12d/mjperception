"""Randomised door instances.

A door is one draw of hidden mechanics (inertia, friction, damping, spring),
fixed across all of that door's episodes. Models come from the baseline's
``dyn.load_model``; only the torsional spring is layered on here, set through the
MuJoCo model fields rather than by editing XML.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import mujoco
import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import DoorSamplingConfig

# order defines the column order of the saved ``door_params`` table
PARAM_FIELDS: tuple[str, ...] = (
    "density",
    "frictionloss",
    "damping",
    "stiffness",
    "springref",
)

DERIVED_FIELDS: tuple[str, ...] = ("mass", "I_hinge")


@dataclass(frozen=True)
class DoorParams:
    """Hidden mechanics of a single door instance."""

    door_id: int
    model_path: str
    density: float
    frictionloss: float
    damping: float
    stiffness: float
    springref: float

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in PARAM_FIELDS], dtype=np.float64)

    def summary(self) -> str:
        return (
            f"door {self.door_id:3d} | rho={self.density:7.1f} "
            f"mu={self.frictionloss:5.2f} b={self.damping:5.2f} "
            f"k={self.stiffness:5.2f} ref={self.springref:+.2f} "
            f"[{self.model_path}]"
        )


def sample_door_params(
    cfg: DoorSamplingConfig, rng: np.random.Generator, door_id: int
) -> DoorParams:
    """Draw one door. Density is log-uniform, the scale that matters for inertia."""
    lo, hi = cfg.density_range
    density = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    stiffness = (
        0.0
        if rng.random() < cfg.frac_no_spring
        else float(rng.uniform(*cfg.stiffness_range))
    )
    return DoorParams(
        door_id=door_id,
        model_path=str(rng.choice(cfg.model_paths)),
        density=density,
        frictionloss=float(rng.uniform(*cfg.frictionloss_range)),
        damping=float(rng.uniform(*cfg.damping_range)),
        stiffness=stiffness,
        springref=float(rng.uniform(*cfg.springref_range)) if stiffness > 0 else 0.0,
    )


def sample_door_population(
    cfg: DoorSamplingConfig, seed: int
) -> tuple[list[DoorParams], list[DoorParams]]:
    """Training and held-out populations. Held-out ids continue after the training
    ids, so a training id is always a valid embedding-table row."""
    rng = np.random.default_rng(seed)
    train = [sample_door_params(cfg, rng, i) for i in range(cfg.n_train_doors)]
    heldout = [
        sample_door_params(cfg, rng, cfg.n_train_doors + i)
        for i in range(cfg.n_heldout_doors)
    ]
    return train, heldout


def build_model(params: DoorParams) -> mujoco.MjModel:
    """MuJoCo model for one door: ``dyn.load_model`` for density/friction/damping,
    then the torsional spring on top."""
    model = dyn.load_model(
        density=params.density,
        frictionloss=params.frictionloss,
        damping=params.damping,
        model_path=params.model_path,
    )
    jid = model.joint("hinge").id
    qadr = model.jnt_qposadr[jid]
    model.jnt_stiffness[jid] = params.stiffness
    model.qpos_spring[qadr] = params.springref
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model


def ground_truth(model: mujoco.MjModel, params: DoorParams) -> dict[str, float]:
    """True hinge-frame mechanics, via the baseline's ``true_hinge_inertia`` so the
    numbers stay comparable to the RLS estimates."""
    gt = dyn.true_hinge_inertia(model)
    return {
        "density": params.density,
        "frictionloss": float(gt["frictionloss"]),
        "damping": float(gt["damping"]),
        "stiffness": float(model.jnt_stiffness[model.joint("hinge").id]),
        "springref": float(model.qpos_spring[model.jnt_qposadr[model.joint("hinge").id]]),
        "mass": float(gt["mass"]),
        "I_hinge": float(gt["I_hinge"]),
    }


def params_table(rows: Sequence[dict[str, float]]) -> np.ndarray:
    """Stack ground-truth dicts into a (n_doors, n_fields) array."""
    cols = PARAM_FIELDS + DERIVED_FIELDS
    return np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)


PARAM_TABLE_COLUMNS: tuple[str, ...] = PARAM_FIELDS + DERIVED_FIELDS

assert set(PARAM_FIELDS) <= {f.name for f in fields(DoorParams)}
