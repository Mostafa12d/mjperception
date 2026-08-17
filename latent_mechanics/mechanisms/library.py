"""A family of articulated mechanisms behind one interaction interface.

Every mechanism exposes ``state = [q, qdot]`` and ``action = tau`` for ONE joint.
Units are NOT rescaled between revolute (rad, N*m) and prismatic (m, N): whether
a representation trained on radians transfers to metres is the question this
stage asks. Actions go straight to ``qfrc_applied[dof]``, the only formulation
identical across joint types, so Stage-4 data is regenerated rather than mixed
with the earlier handle-site datasets.

The six families, in increasing distance from the training distribution:

  door              the original revolute door
  nonlinear_hinge   door + Stribeck and position-dependent friction
  soft_close        door + a damper that engages near closed
  drawer            prismatic, metres, ~10x the force scale
  laptop            revolute but tiny inertia and friction-dominated
  bifold            two-link cabinet; only the first joint is observed, so the
                    observed state is non-Markov by construction
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from latent_mechanics.config import ExcitationConfig
from latent_mechanics.mismatch.perturbations import (
    PlantPerturbation,
    PositionDependentFriction,
    StribeckFriction,
    smooth_sign,
)
from scenes import scene_path

ASSETS = Path(__file__).parent / "assets"
DOOR_XML = "door.xml"  # lives in scenes/, resolved by FamilySpec.resolve_xml


@dataclass
class SoftCloseDamper(PlantPerturbation):
    """A soft-close hinge: free through most of the range, heavily damped over the
    last few degrees.

        tau_extra = -gain * qdot * exp(-(q / width)^2)

    A position-dependent damping coefficient, which no constant-`b` model can
    express, and unlike Stribeck it depends on position rather than speed.
    """

    gain: float = 6.0
    width: float = 0.25
    name: str = "soft_close"

    def extra_torque(self, t: float, theta: float, theta_dot: float) -> float:
        return -self.gain * theta_dot * float(np.exp(-((theta / self.width) ** 2)))

    def describe(self) -> dict:
        return {"name": self.name, "gain": self.gain, "width": self.width}


@dataclass(frozen=True)
class MechanismParams:
    """One sampled instance. All of this is analysis metadata, never model input."""

    mechanism_id: int
    family: str
    xml: str
    joint_type: str  # "revolute" | "prismatic"
    density_scale: float
    frictionloss: float
    damping: float
    stiffness: float
    springref: float
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        e = " ".join(f"{k}={v:.2f}" for k, v in self.extra.items())
        return (f"{self.family:16s} id={self.mechanism_id:3d} "
                f"rho={self.density_scale:5.2f} mu={self.frictionloss:6.2f} "
                f"b={self.damping:5.2f} k={self.stiffness:5.2f} {e}")


@dataclass
class FamilySpec:
    """How to sample and build one family."""

    name: str
    xml: str
    joint_type: str
    density_scale_range: tuple[float, float]
    friction_range: tuple[float, float]
    damping_range: tuple[float, float]
    stiffness_range: tuple[float, float] = (0.0, 0.0)
    springref_range: tuple[float, float] = (0.0, 0.0)
    frac_no_spring: float = 1.0
    # Natural force unit of this family, in door units. Excitation is generated
    # in door units and multiplied by this, so sample_profile's door-calibrated
    # bias floor does not hand a 0.1 N*m laptop hinge an oversized push.
    force_unit: float = 1.0
    # override for families a door-sized push would saturate against their stop
    bias_range: tuple[float, float] | None = None
    extra_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    observed_joint: str = "hinge"

    def resolve_xml(self) -> str:
        """Absolute path to this family's XML: an asset beside this module, or the
        shared ``scenes/door.xml``. Absolute, so cached params stay loadable."""
        p = ASSETS / self.xml
        return str(p) if p.exists() else scene_path(self.xml)


FAMILIES: dict[str, FamilySpec] = {
    # same mechanism as "door", ranges collapsed toward their centres, so the
    # curriculum can vary parameter diversity before introducing a new mechanism
    "door_narrow": FamilySpec(
        name="door_narrow", xml=DOOR_XML, joint_type="revolute",
        density_scale_range=(0.85, 1.20),
        friction_range=(2.5, 3.5), damping_range=(0.15, 0.35),
        stiffness_range=(0.0, 0.0), springref_range=(0.0, 0.0),
        frac_no_spring=1.0, force_unit=1.0,
    ),
    "door": FamilySpec(
        name="door", xml=DOOR_XML, joint_type="revolute",
        density_scale_range=(0.33, 2.33),  # matches Stage-1 density 200-1400
        friction_range=(0.5, 6.0), damping_range=(0.02, 1.5),
        stiffness_range=(0.0, 8.0), springref_range=(-0.1, 0.6),
        frac_no_spring=0.3, force_unit=1.0,
    ),
    "nonlinear_hinge": FamilySpec(
        name="nonlinear_hinge", xml=DOOR_XML, joint_type="revolute",
        density_scale_range=(0.33, 2.33),
        friction_range=(0.5, 6.0), damping_range=(0.02, 1.5),
        stiffness_range=(0.0, 8.0), springref_range=(-0.1, 0.6),
        frac_no_spring=0.3, force_unit=1.0,
        extra_ranges={"stribeck_excess": (0.5, 3.0), "stribeck_v": (0.03, 0.12),
                      "pos_friction_amp": (0.3, 2.0), "pos_friction_period": (0.4, 1.2)},
    ),
    "soft_close": FamilySpec(
        name="soft_close", xml=DOOR_XML, joint_type="revolute",
        density_scale_range=(0.33, 2.33),
        friction_range=(0.5, 4.0), damping_range=(0.02, 1.0),
        stiffness_range=(1.0, 8.0), springref_range=(-0.05, 0.05),
        frac_no_spring=0.0,  # a soft-close door always has a closer spring
        force_unit=1.0,
        extra_ranges={"soft_close_gain": (3.0, 20.0), "soft_close_width": (0.15, 0.45)},
    ),
    "drawer": FamilySpec(
        name="drawer", xml="drawer.xml", joint_type="prismatic",
        density_scale_range=(0.4, 2.5),
        friction_range=(3.0, 16.0),  # newtons
        damping_range=(1.0, 12.0),   # N*s/m
        force_unit=3.0,
    ),
    "laptop": FamilySpec(
        name="laptop", xml="laptop.xml", joint_type="revolute",
        density_scale_range=(0.5, 2.0),
        friction_range=(0.15, 0.8),
        # heavy damping is what stops the screen falling open
        damping_range=(0.30, 1.50),
        force_unit=0.2, bias_range=(0.7, 1.2),
    ),
    "bifold": FamilySpec(
        name="bifold", xml="bifold.xml", joint_type="revolute",
        density_scale_range=(0.4, 1.6),
        friction_range=(0.5, 4.0), damping_range=(0.05, 1.0),
        force_unit=1.0,
        extra_ranges={"leaf_friction": (0.2, 2.5), "leaf_damping": (0.05, 0.8)},
    ),
}

FAMILY_ORDER = list(FAMILIES)


def sample_params(
    family: str, rng: np.random.Generator, mechanism_id: int
) -> MechanismParams:
    spec = FAMILIES[family]
    lo, hi = spec.density_scale_range
    density_scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    stiffness = (
        0.0 if rng.random() < spec.frac_no_spring
        else float(rng.uniform(*spec.stiffness_range))
    )
    extra = {k: float(rng.uniform(*v)) for k, v in spec.extra_ranges.items()}
    return MechanismParams(
        mechanism_id=mechanism_id, family=family, xml=spec.resolve_xml(),
        joint_type=spec.joint_type, density_scale=density_scale,
        frictionloss=float(rng.uniform(*spec.friction_range)),
        damping=float(rng.uniform(*spec.damping_range)),
        stiffness=stiffness,
        springref=float(rng.uniform(*spec.springref_range)) if stiffness > 0 else 0.0,
        extra=extra,
    )


def build_model(params: MechanismParams) -> mujoco.MjModel:
    """Instantiate one mechanism with its sampled mechanics."""
    # Resolved again rather than trusted: caches written before the scene files
    # moved into scenes/ hold a bare "door.xml" in params.xml.
    model = mujoco.MjModel.from_xml_path(scene_path(params.xml))
    bid = model.body("door").id
    model.body_mass[bid] *= params.density_scale
    model.body_inertia[bid] *= params.density_scale

    jid = model.joint("hinge").id
    dof = model.jnt_dofadr[jid]
    qadr = model.jnt_qposadr[jid]
    model.dof_frictionloss[dof] = params.frictionloss
    model.dof_damping[dof] = params.damping
    model.jnt_stiffness[jid] = params.stiffness
    model.qpos_spring[qadr] = params.springref

    if params.family == "bifold":
        # Second link scales with the first so the pair stays physically sane.
        lb = model.body("leaf").id
        model.body_mass[lb] *= params.density_scale
        model.body_inertia[lb] *= params.density_scale
        ljid = model.joint("leaf_hinge").id
        model.dof_frictionloss[model.jnt_dofadr[ljid]] = params.extra["leaf_friction"]
        model.dof_damping[model.jnt_dofadr[ljid]] = params.extra["leaf_damping"]

    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model


def perturbations_for(params: MechanismParams) -> list[PlantPerturbation]:
    """Family-specific physics layered on top of the MuJoCo model."""
    if params.family == "nonlinear_hinge":
        return [
            StribeckFriction(excess=params.extra["stribeck_excess"],
                             v_stribeck=params.extra["stribeck_v"]),
            PositionDependentFriction(amplitude=params.extra["pos_friction_amp"],
                                      period=params.extra["pos_friction_period"]),
        ]
    if params.family == "soft_close":
        return [SoftCloseDamper(gain=params.extra["soft_close_gain"],
                                width=params.extra["soft_close_width"])]
    return []


def excitation_for(base: ExcitationConfig, params: MechanismParams) -> ExcitationConfig:
    """Excitation config for one family, in DOOR units; ``scaled_profile`` converts
    via ``force_unit``. The shape is identical for every family, so none gets a
    richer information diet than another."""
    spec = FAMILIES[params.family]
    cfg = copy.deepcopy(base)
    if spec.bias_range is not None:
        cfg.bias_over_friction_range = spec.bias_range
        cfg.swing_over_friction_range = (1.05, 1.5)
    return cfg


def scaled_profile(base: ExcitationConfig, rng, n_steps: int, frame_skip: int,
                   params: MechanismParams):
    """Draw an excitation profile in door units, then convert to family units."""
    from dataclasses import replace as _replace

    from latent_mechanics.excitation import sample_profile

    u = FAMILIES[params.family].force_unit
    prof = sample_profile(excitation_for(base, params), rng, n_steps, frame_skip,
                          params.frictionloss / u)
    return _replace(prof, values=prof.values * u)


def joint_info(model: mujoco.MjModel) -> tuple[int, int, int]:
    """(qpos address, dof address, joint id) of the single observed joint."""
    jid = model.joint("hinge").id
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), int(jid)


def ground_truth(model: mujoco.MjModel, params: MechanismParams) -> dict[str, float]:
    """True mechanics of the observed joint. Analysis only, never a model input.

    "inertia" is the hinge-frame moment for a revolute joint and the moving mass
    for a prismatic one: different quantities, one column, same role in the EOM.
    """
    bid = model.body("door").id
    _, dof, jid = joint_info(model)
    m = float(model.body_mass[bid])

    if params.joint_type == "prismatic":
        inertia = m
    else:
        I_prin = np.array(model.body_inertia[bid], dtype=float)
        com = np.array(model.body_ipos[bid], dtype=float)
        R = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R.ravel(), np.array(model.body_iquat[bid], dtype=float))
        I_com = R @ np.diag(I_prin) @ R.T
        axis = np.array(model.jnt_axis[jid], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # Parallel-axis about the joint axis through the joint origin.
        perp = com - np.dot(com, axis) * axis
        inertia = float(axis @ I_com @ axis) + m * float(perp @ perp)

    return {
        "inertia": float(inertia),
        "mass": m,
        "frictionloss": float(model.dof_frictionloss[dof]),
        "damping": float(model.dof_damping[dof]),
        "stiffness": float(model.jnt_stiffness[jid]),
        "springref": float(model.qpos_spring[model.jnt_qposadr[jid]]),
        "is_prismatic": 1.0 if params.joint_type == "prismatic" else 0.0,
        "n_dof": float(model.nv),
        "range_lo": float(model.jnt_range[jid][0]),
        "range_hi": float(model.jnt_range[jid][1]),
        **{k: float(v) for k, v in params.extra.items()},
    }


GT_COLUMNS = ("inertia", "mass", "frictionloss", "damping", "stiffness",
              "springref", "is_prismatic", "n_dof", "range_lo", "range_hi")


def sample_population(
    families: Sequence[str], n_per_family: int, seed: int, id_offset: int = 0
) -> list[MechanismParams]:
    rng = np.random.default_rng(seed)
    out = []
    for fam in families:
        for _ in range(n_per_family):
            out.append(sample_params(fam, rng, id_offset + len(out)))
    return out


__all__ = [
    "MechanismParams", "FamilySpec", "FAMILIES", "FAMILY_ORDER", "SoftCloseDamper",
    "sample_params", "sample_population", "build_model", "perturbations_for",
    "excitation_for", "ground_truth", "joint_info", "GT_COLUMNS",
]
