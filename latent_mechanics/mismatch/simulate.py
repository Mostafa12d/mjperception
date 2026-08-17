"""The perturbed simulator.

``dyn.simulate``'s torque callback receives only the time, but Stage-3 physics is
state dependent, so this needs its own loop. It mirrors ``dyn.simulate`` line for
line, calling the same helpers in the same order, plus two hooks: one to mutate
model parameters and one to add an unmodelled torque.

``verify_matches_baseline`` asserts the two agree to machine precision with no
perturbations, since a silent divergence would look like a real effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mujoco
import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.mismatch.perturbations import PlantPerturbation


@dataclass
class RolloutLog:
    """Same keys as ``dyn.simulate`` returns, plus the perturbation record."""

    t: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    theta_ddot: np.ndarray
    tau_oracle: np.ndarray
    tau_ft: np.ndarray
    tau_perturb: np.ndarray  # unmodelled torque actually applied
    ncon: np.ndarray
    gt: dict

    def as_dict(self) -> dict:
        """Dict shaped like ``dyn.simulate``'s output, so Stage-1 helpers slice
        perturbed and ideal rollouts with identical code."""
        return {
            "t": self.t, "theta": self.theta, "theta_dot": self.theta_dot,
            "theta_ddot": self.theta_ddot, "tau_oracle": self.tau_oracle,
            "tau_ft": self.tau_ft, "ncon": self.ncon, "gt": self.gt,
        }


def simulate_perturbed(
    tau_fn: Callable[[float], float],
    model: mujoco.MjModel,
    n_steps: int,
    perturbations: Sequence[PlantPerturbation] = (),
) -> RolloutLog:
    """Roll out one episode with optional unmodelled physics. ``tau_ft``, the
    recorded action, is reconstructed from the commanded force alone; the
    perturbation torque is added afterwards, so it never enters the action."""
    data = mujoco.MjData(model)
    assert abs(model.opt.timestep - dyn.DT) < 1e-12

    hinge_qpos = model.joint("hinge").qposadr[0]
    hinge_dof = model.joint("hinge").dofadr[0]
    handle_sid = model.site("handle").id
    door_bid = model.body("door").id
    gt = dyn.true_hinge_inertia(model)

    for p in perturbations:
        p.reset(model)

    t = np.arange(n_steps) * dyn.DT
    theta = np.zeros(n_steps)
    theta_dot = np.zeros(n_steps)
    theta_ddot = np.zeros(n_steps)
    tau_oracle = np.zeros(n_steps)
    tau_ft = np.zeros(n_steps)
    tau_perturb = np.zeros(n_steps)
    ncon = np.zeros(n_steps, dtype=int)

    for i in range(n_steps):
        th = float(data.qpos[hinge_qpos])
        thd = float(data.qvel[hinge_dof])
        tau_des = float(tau_fn(t[i]))
        force = (tau_des / dyn.HANDLE_DIST) * dyn.tangential_direction(th)

        hinge_pos = data.xpos[door_bid].copy()
        hinge_axis = data.xmat[door_bid].reshape(3, 3)[:, 2].copy()
        handle_pos = data.site_xpos[handle_sid].copy()

        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data, force, np.zeros(3), handle_pos, door_bid, data.qfrc_applied
        )
        tau_h_oracle = float(data.qfrc_applied[hinge_dof])
        # The ACTION: reconstructed from the commanded force only.
        tau_h_ft = dyn.hinge_torque_from_handle_force(
            handle_pos, hinge_pos, hinge_axis, force
        )

        # Unmodelled physics, applied after the action is recorded.
        extra = 0.0
        for p in perturbations:
            p.update_model(t[i], model)
            extra += p.extra_torque(t[i], th, thd)
        if extra:
            data.qfrc_applied[hinge_dof] += extra

        mujoco.mj_step(model, data)

        theta[i] = float(data.qpos[hinge_qpos])
        theta_dot[i] = float(data.qvel[hinge_dof])
        theta_ddot[i] = float(data.qacc[hinge_dof])
        tau_oracle[i] = tau_h_oracle
        tau_ft[i] = tau_h_ft
        tau_perturb[i] = extra
        ncon[i] = int(data.ncon)

    return RolloutLog(t, theta, theta_dot, theta_ddot, tau_oracle, tau_ft,
                      tau_perturb, ncon, gt)


def verify_matches_baseline(
    tau_fn: Callable[[float], float], model: mujoco.MjModel, n_steps: int,
    tol: float = 0.0,
) -> dict[str, float]:
    """With no perturbations, must equal ``dyn.simulate`` exactly. The caller must
    have set ``N_STEPS`` (``data_gen.episode_length``). Raises past ``tol``."""
    import copy

    mine = simulate_perturbed(tau_fn, copy.deepcopy(model), n_steps, perturbations=())
    theirs = dyn.simulate(tau_fn, model=copy.deepcopy(model))

    diffs = {}
    for key in ("theta", "theta_dot", "theta_ddot", "tau_oracle", "tau_ft"):
        a = getattr(mine, key)
        b = theirs[key]
        diffs[key] = float(np.max(np.abs(a[: len(b)] - b[: len(a)])))
    worst = max(diffs.values())
    if worst > tol:
        raise AssertionError(
            f"perturbed simulator diverges from dyn.simulate with no perturbations "
            f"(max deviation {worst:.3e}): {diffs}"
        )
    return diffs


__all__ = ["simulate_perturbed", "verify_matches_baseline", "RolloutLog"]
