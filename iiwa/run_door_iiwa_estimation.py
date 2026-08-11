"""
Door + KUKA iiwa 14: proprioceptive θ + simulated wrist F/T sensor.

Sensor pipeline (hardware-realizable):
  1. Wrist F/T site on link7 reads the wrench transmitted through joint7.
     In simulation this is extracted from the weld equality-constraint rows
     of MuJoCo's efc arrays: τ_weld = Σ J_efc[eq,0] · f_efc[eq].
  2. Optional Gaussian noise models a real ATI-style F/T sensor.
  3. τ_weld is the torque the arm applies to the door hinge — it does NOT
     include the door's own frictionloss, so the estimator correctly
     recovers both I_hinge and μ (friction).

This replaces the previous qfrc_constraint[0] shortcut, which is internal
to the MuJoCo solver and has no hardware equivalent.

Angle (θ) is still proprioceptive: atan2(r_y, r_x) from arm FK.

Run:
    python3.10 run_door_iiwa_estimation.py
"""

from __future__ import annotations

import numpy as np
import mujoco

from baseline.run_door_dynamics_validation import (
    DT,
    HANDLE_DIST,
    true_hinge_inertia,
    tangential_direction,
    hinge_torque_from_handle_force,
    fit_params,
)
from scenes import scene_path

SCENE_PATH = scene_path("door_iiwa_scene.xml")
# Keep EE kinematics; reduce dynamic loading through the weld on the door DOF.
ARM_INERTIA_SCALE = 0.025
ARM_JOINT_ARMATURE = 0.24
T_END = 6.0
N_STEPS = int(T_END / DT)
STICTION_GUESS = 3.5
EXCITATION_AMP = 6.0
EXCITATION_FREQ = 0.5
VEL_THRESH = 0.02

# Wrist F/T sensor noise (std dev in Nm, typical ATI Mini45 level).
# Set to 0.0 for noise-free simulation.
WRIST_FT_NOISE_STD = 0.0


def tau_quasistatic(t: float) -> float:
    if t < 0.3:
        return STICTION_GUESS * (t / 0.3)
    return STICTION_GUESS


def tau_excited(t: float) -> float:
    return STICTION_GUESS + EXCITATION_AMP * np.sin(2 * np.pi * EXCITATION_FREQ * t)


def load_iiwa_door_model(
    scale: float = ARM_INERTIA_SCALE,
    armature: float = ARM_JOINT_ARMATURE,
) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    for b in range(model.nbody):
        name = model.body(b).name
        if name == "base" or name.startswith("link"):
            model.body_mass[b] *= scale
            model.body_inertia[b] *= scale
    # Arm joints only (skip door hinge at dof 0)
    for j in range(1, model.njnt):
        model.dof_armature[model.jnt_dofadr[j]] = armature
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model


def door_angle_from_proprio(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Proprioceptive θ from arm FK: hinge → attachment_site in world XY."""
    hinge = data.xpos[model.body("door").id]
    tip = data.site_xpos[model.site("attachment_site").id]
    r = tip - hinge
    return float(np.arctan2(r[1], r[0]))


def arm_torques_for_hinge_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_hinge: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map desired hinge torque → handle force → arm joint torques (+ gravity)."""
    force = (tau_hinge / HANDLE_DIST) * tangential_direction(theta)
    att_id = model.site("attachment_site").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, att_id)

    # Arm DOFs are everything except the door hinge (dof 0)
    arm_dofs = np.arange(1, model.nv)
    tau_arm = jacp[:, arm_dofs].T @ force
    # Gravity / bias compensation so the arm holds itself
    tau_arm = tau_arm + data.qfrc_bias[arm_dofs]
    return tau_arm, force, tau_hinge


def wrist_ft_hinge_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    noise_std: float = WRIST_FT_NOISE_STD,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Simulated wrist F/T sensor → hinge torque (Nm).

    Physical interpretation
    -----------------------
    A 6-axis F/T sensor mounted between link6 and link7 measures the wrench
    the distal system (link7 + door via weld) exerts on the proximal arm.
    The hinge torque is obtained by projecting that wrench via the handle
    geometry: τ = (p_handle − p_hinge) × F · ẑ.

    MuJoCo implementation
    ---------------------
    The weld equality constraint populates MuJoCo's efc arrays.  The scalar
    torque the weld applies to the door's hinge DOF is:

        τ_weld = Σ_{equality rows i} J_efc[i, 0] · f_efc[i]

    where column 0 of J_efc corresponds to the hinge DOF.  This is the
    hardware-equivalent of projecting the wrist F/T reading onto the hinge
    axis — without the door's own frictionloss, so the estimator sees:

        I · θ̈ + μ = τ_weld        (μ captures frictionloss correctly)

    Noise
    -----
    Optional Gaussian noise (noise_std ≠ 0) models sensor noise at the level
    of a real ATI Mini45 wrist sensor (~0.05–0.1 Nm RMS on torque).
    """
    nefc = data.nefc
    nv = model.nv
    J_efc = data.efc_J.reshape(nefc, nv)
    f_efc = data.efc_force[:nefc]
    # mjCNSTR_EQUALITY = 0 in MuJoCo enum; frictionloss = 1 (excluded)
    eq_mask = data.efc_type[:nefc] == 0
    tau_weld = float((J_efc[eq_mask, 0] * f_efc[eq_mask]).sum())

    if noise_std > 0.0:
        _rng = rng if rng is not None else np.random.default_rng()
        tau_weld += _rng.standard_normal() * noise_std

    return tau_weld


def simulate(
    tau_fn,
    noise_std: float = WRIST_FT_NOISE_STD,
    seed: int | None = None,
) -> dict:
    model = load_iiwa_door_model()
    data = mujoco.MjData(model)
    assert abs(model.opt.timestep - DT) < 1e-12

    # Start at welded grasp keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    gt = true_hinge_inertia(model)
    hinge_dof = model.joint("hinge").dofadr[0]
    assert hinge_dof == 0

    rng = np.random.default_rng(seed) if noise_std > 0.0 else None

    t = np.arange(N_STEPS) * DT
    theta = np.zeros(N_STEPS)
    theta_dot = np.zeros(N_STEPS)
    theta_ddot = np.zeros(N_STEPS)
    tau_cmd = np.zeros(N_STEPS)
    tau_ft = np.zeros(N_STEPS)   # wrist F/T sensor observation
    ncon = np.zeros(N_STEPS, dtype=int)

    th_prev = None
    thd_prev = None

    for i in range(N_STEPS):
        # Proprioceptive angle from arm FK (attachment site), not hinge qpos
        th = door_angle_from_proprio(model, data)
        tau_des = float(tau_fn(t[i]))

        tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau_des, th)
        data.ctrl[:] = tau_arm
        mujoco.mj_step(model, data)

        # Wrist F/T sensor: equality-constraint (weld) contribution to hinge DOF.
        # Excludes door frictionloss — same as what a physical F/T sensor at
        # the wrist would see after tool compensation and hinge projection.
        tau_h = wrist_ft_hinge_torque(model, data, noise_std=noise_std, rng=rng)

        th_new = door_angle_from_proprio(model, data)
        # Differentiate proprioceptive θ (no oracle qvel)
        thd = 0.0 if th_prev is None else (th_new - th_prev) / DT
        thdd = 0.0 if thd_prev is None else (thd - thd_prev) / DT
        th_prev, thd_prev = th_new, thd

        theta[i] = th_new
        theta_dot[i] = thd
        theta_ddot[i] = thdd
        tau_cmd[i] = tau_des
        tau_ft[i] = tau_h
        ncon[i] = int(data.ncon)

    return dict(
        t=t,
        theta=theta,
        theta_dot=theta_dot,
        theta_ddot=theta_ddot,
        tau_oracle=tau_cmd,
        tau_ft=tau_ft,
        ncon=ncon,
        gt=gt,
        hinge_qpos_final=float(data.qpos[model.joint("hinge").qposadr[0]]),
    )


def evaluate(log: dict, tau_key: str = "tau_ft") -> dict:
    I_true = log["gt"]["I_hinge"]
    moving = np.abs(log["theta_dot"]) > VEL_THRESH
    moving[:20] = False
    # stay off joint limits
    moving &= (log["theta"] > -0.17 + 0.05) & (log["theta"] < 2.09 - 0.05)
    fit = fit_params(
        log["theta_dot"], log["theta_ddot"], log[tau_key], moving, include_damping=False
    )
    I_hat = float(fit["params"][0]) if np.isfinite(fit["params"][0]) else np.nan
    rel_err = abs(I_hat - I_true) / I_true * 100.0 if np.isfinite(I_hat) else np.nan
    thdd_rms = (
        float(np.sqrt(np.mean(log["theta_ddot"][moving] ** 2))) if moving.any() else 0.0
    )
    return dict(
        I_true=I_true,
        I_hat=I_hat,
        mu_hat=float(fit["params"][1]) if len(fit["params"]) > 1 else np.nan,
        rel_err=rel_err,
        cond=fit["cond"],
        n_moving=int(moving.sum()),
        thdd_rms=thdd_rms,
        theta_end_deg=float(np.degrees(log["theta"][-1])),
        hinge_end_deg=float(np.degrees(log["hinge_qpos_final"])),
        max_ncon=int(log["ncon"].max()),
    )


def main() -> None:
    print("Door + iiwa 14 estimation — wrist F/T sensor, proprioceptive θ\n")

    # Sanity: model loads and weld holds at keyframe
    model = load_iiwa_door_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)
    att = data.site_xpos[model.site("attachment_site").id]
    han = data.site_xpos[model.site("handle").id]
    print(f"  attachment={att}")
    print(f"  handle    ={han}")
    print(f"  |att-handle|={np.linalg.norm(att - han):.4e} m")
    print(f"  proprio θ={np.degrees(door_angle_from_proprio(model, data)):.2f}°  "
          f"hinge qpos={np.degrees(data.qpos[0]):.2f}°")
    print(f"  arm inertia scale={ARM_INERTIA_SCALE}  armature={ARM_JOINT_ARMATURE}")
    gt = true_hinge_inertia(model)
    print(f"  I_hinge true={gt['I_hinge']:.4f} kg·m²  "
          f"μ_true={model.dof_frictionloss[model.joint('hinge').dofadr[0]]:.3f} Nm")
    print(f"  wrist F/T noise σ={WRIST_FT_NOISE_STD:.3f} Nm\n")

    results = {}
    for name, fn in [("quasi-static", tau_quasistatic), ("excited", tau_excited)]:
        log = simulate(fn)
        r = evaluate(log)
        results[name] = r
        mu_true = model.dof_frictionloss[model.joint("hinge").dofadr[0]]
        mu_err = abs(r["mu_hat"] - mu_true) / mu_true * 100 if mu_true > 0 else float("nan")
        print(f"=== {name} ===")
        print(f"  I_true={r['I_true']:.4f}  I_hat={r['I_hat']:.4f}  I_err={r['rel_err']:.1f}%")
        print(f"  μ_true={mu_true:.3f}  μ_hat={r['mu_hat']:.3f}  μ_err={mu_err:.1f}%")
        print(f"  cond={r['cond']:.1f}  thdd_rms={r['thdd_rms']:.4f}")
        print(f"  proprio θ_end={r['theta_end_deg']:.1f}°  "
              f"hinge θ_end={r['hinge_end_deg']:.1f}°  "
              f"moving={r['n_moving']}  max_ncon={r['max_ncon']}")
        print()

    qs, ex = results["quasi-static"], results["excited"]
    print("--- Pass criteria ---")
    print(f"  excited I_err < 5%:       {ex['rel_err']:.1f}%  "
          f"{'PASS' if ex['rel_err'] < 5 else 'FAIL'}")
    print(f"  quasi-static I_err > 50%: {qs['rel_err']:.1f}%  "
          f"{'PASS' if qs['rel_err'] > 50 else 'FAIL'}")

    # --- noise sweep ---
    print("\n--- Wrist F/T noise sweep (excited only) ---")
    print(f"  {'σ (Nm)':<10} {'I_err %':<10} {'μ_err %':<10} {'cond':<8}")
    mu_true = model.dof_frictionloss[model.joint("hinge").dofadr[0]]
    for sigma in [0.0, 0.05, 0.10, 0.20, 0.50]:
        log_n = simulate(tau_excited, noise_std=sigma, seed=42)
        r_n = evaluate(log_n)
        mu_err_n = abs(r_n["mu_hat"] - mu_true) / mu_true * 100
        print(f"  {sigma:<10.2f} {r_n['rel_err']:<10.1f} {mu_err_n:<10.1f} {r_n['cond']:<8.1f}")


if __name__ == "__main__":
    main()
