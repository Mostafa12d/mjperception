"""
Generic rollout and dataset generation for the mechanism suite.

One simulator serves every family. It applies the action directly to the
observed degree of freedom, so revolute and prismatic joints go through exactly
the same code path -- there is no per-family branch anywhere in the loop, which
is what makes "the interface is unchanged" a checkable claim rather than a
description.

Transitions are sliced by Stage 1's ``transitions_from_log``, unchanged, so the
zero-order-hold alignment and the frame-skip convention are identical to every
earlier stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mujoco
import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import ExperimentConfig
from latent_mechanics.data_gen import moving_fraction, transitions_from_log
from latent_mechanics.excitation import sample_profile
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.library import MechanismParams
from latent_mechanics.mismatch.perturbations import PlantPerturbation

LIMIT_MARGIN_FRAC = 0.03  # of the joint's travel


@dataclass
class MechanismLog:
    t: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray
    action: np.ndarray
    tau_perturb: np.ndarray
    ncon: np.ndarray

    def as_stage1_dict(self) -> dict:
        """Shaped like ``dyn.simulate``'s output so Stage-1 slicing applies."""
        return {"t": self.t, "theta": self.q, "theta_dot": self.qdot,
                "theta_ddot": self.qddot, "tau_ft": self.action,
                "tau_oracle": self.action, "ncon": self.ncon, "gt": {}}


def simulate_mechanism(
    tau_fn: Callable[[float], float],
    model: mujoco.MjModel,
    n_steps: int,
    perturbations: Sequence[PlantPerturbation] = (),
) -> MechanismLog:
    """Roll out one episode, applying the action as a generalised force.

    Unobserved degrees of freedom (the bi-fold leaf) evolve freely and are never
    logged -- that is the point of including such a mechanism.
    """
    data = mujoco.MjData(model)
    qadr, dof, _ = lib.joint_info(model)
    for p in perturbations:
        p.reset(model)

    t = np.arange(n_steps) * dyn.DT
    q = np.zeros(n_steps); qdot = np.zeros(n_steps); qddot = np.zeros(n_steps)
    action = np.zeros(n_steps); tau_p = np.zeros(n_steps)
    ncon = np.zeros(n_steps, dtype=int)

    for i in range(n_steps):
        qi = float(data.qpos[qadr])
        vi = float(data.qvel[dof])
        tau = float(tau_fn(t[i]))

        extra = 0.0
        for p in perturbations:
            p.update_model(t[i], model)
            extra += p.extra_torque(t[i], qi, vi)

        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = tau + extra
        mujoco.mj_step(model, data)

        q[i] = float(data.qpos[qadr])
        qdot[i] = float(data.qvel[dof])
        qddot[i] = float(data.qacc[dof])
        action[i] = tau          # the ACTION excludes unmodelled physics
        tau_p[i] = extra
        ncon[i] = int(data.ncon)

    return MechanismLog(t, q, qdot, qddot, action, tau_p, ncon)


def limit_margin_for(lo: float, hi: float) -> float:
    """Absolute limit margin for a joint spanning ``[lo, hi]``.

    Expressed as a fraction of travel so it means the same thing for a 0.5 m
    drawer and a 2.26 rad door.
    """
    return LIMIT_MARGIN_FRAC * (hi - lo)


def near_limit_mask(
    state: np.ndarray, next_state: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Limit proximity as a fraction of travel, so it means the same thing for a
    0.5 m drawer and a 2.26 rad door."""
    margin = limit_margin_for(lo, hi)
    at = lambda x: (x < lo + margin) | (x > hi - margin)
    return at(state[:, 0]) | at(next_state[:, 0])


@dataclass
class MechanismEpisodes:
    """All transitions for one mechanism instance."""

    params: MechanismParams
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    episode_id: np.ndarray
    gt: dict[str, float]

    def __len__(self) -> int:
        return len(self.state)


def rollout_mechanism(
    params: MechanismParams,
    cfg: ExperimentConfig,
    n_episodes: int,
    episode_seconds: float,
    frame_skip: int,
    seed: int = 0,
    episode_offset: int = 0,
    exclude_near_limit: bool = True,
) -> MechanismEpisodes:
    """Simulate one mechanism instance and return its transitions."""
    n_steps = int(round(episode_seconds / dyn.DT))
    model = lib.build_model(params)
    gt = lib.ground_truth(model, params)
    _, _, jid = lib.joint_info(model)
    lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])

    S, A, N, E = [], [], [], []
    for ep in range(episode_offset, episode_offset + n_episodes):
        rng = np.random.default_rng(seed * 7717 + (params.mechanism_id + 1) * 1009 + ep)
        profile = lib.scaled_profile(cfg.excitation, rng, n_steps, frame_skip, params)
        model = lib.build_model(params)
        log = simulate_mechanism(profile.as_fn(), model, n_steps,
                                 lib.perturbations_for(params))
        # This mechanism's own joint range, not the door's -- see
        # data_gen.transitions_from_log. With the range and margin passed
        # through, tr["near_limit"] is correct here and is the single source of
        # truth instead of being recomputed below.
        tr = transitions_from_log(log.as_stage1_dict(), frame_skip,
                                  joint_range=(lo, hi),
                                  limit_margin=limit_margin_for(lo, hi))
        s, a, ns = tr["state"], tr["action"], tr["next_state"]
        keep = (~tr["near_limit"] if exclude_near_limit
                else np.ones(len(s), bool))
        if not keep.any():
            continue
        S.append(s[keep]); A.append(a[keep]); N.append(ns[keep])
        E.append(np.full(int(keep.sum()), ep, dtype=np.int32))

    empty = np.zeros((0, 2), np.float32)
    return MechanismEpisodes(
        params=params,
        state=np.concatenate(S) if S else empty,
        action=np.concatenate(A) if A else np.zeros((0, 1), np.float32),
        next_state=np.concatenate(N) if N else empty,
        episode_id=np.concatenate(E) if E else np.zeros(0, np.int32),
        gt=gt,
    )


def describe_population(pops: list[MechanismEpisodes]) -> str:
    """Per-family coverage summary, for sanity-checking the excitation scales."""
    lines = [f"  {'family':16s} {'n':>7} {'|q| p95':>9} {'|qd| p95':>9} "
             f"{'|a| rms':>9} {'moving':>7}"]
    fams = dict.fromkeys(p.params.family for p in pops)
    for fam in fams:
        sub = [p for p in pops if p.params.family == fam and len(p)]
        if not sub:
            continue
        q = np.concatenate([p.state[:, 0] for p in sub])
        v = np.concatenate([p.state[:, 1] for p in sub])
        a = np.concatenate([p.action[:, 0] for p in sub])
        d = np.concatenate([(p.next_state - p.state)[:, 1] for p in sub])
        moving = moving_fraction(v)
        lines.append(f"  {fam:16s} {len(q):>7d} {np.percentile(np.abs(q), 95):>9.3f} "
                     f"{np.percentile(np.abs(v), 95):>9.3f} "
                     f"{np.sqrt((a**2).mean()):>9.3f} {moving:>6.0%}")
    return "\n".join(lines)


__all__ = ["simulate_mechanism", "rollout_mechanism", "MechanismEpisodes",
           "MechanismLog", "near_limit_mask", "describe_population"]
