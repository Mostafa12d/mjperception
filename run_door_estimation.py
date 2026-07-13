"""
MuJoCo Phase 0: reproduce the quasi-static vs. excited observability effect
on a real physics-engine door (geometry-derived mass/inertia), not a toy ODE.

Same regressor as phase0_observability_demo.py:

    tau_applied = I_hinge * theta_ddot + mu_k * sign(theta_dot)

Differences from the NumPy toy (reported honestly, not tuned away):
  - MuJoCo joint also has viscous damping (damping="0.2"), which is NOT in
    the 2-parameter regressor. That is a model mismatch.
  - frictionloss is MuJoCo's smooth Coulomb approximation, not ideal sign().
  - I_hinge ground truth is extracted from the compiled model (parallel axis),
    not a hand-typed constant.

Run:
    python3.10 run_door_estimation.py

For F/T torque reconstruction, ablation, parameter sweeps, and online RLS
(vision still deferred), see run_door_dynamics_validation.py.
"""

from __future__ import annotations

import numpy as np
import mujoco

MODEL_PATH = "door.xml"
DT = 0.002
T_END = 6.0
N_STEPS = int(T_END / DT)
HANDLE_DIST = 0.85  # must match handle site local x in door.xml

# Same force profiles as run_door_sim.py / Phase 0 spirit
STICTION_TORQUE_GUESS = 3.5  # just above frictionloss=3.0


def tau_quasistatic(t: float) -> float:
    if t < 0.3:
        return STICTION_TORQUE_GUESS * (t / 0.3)
    return STICTION_TORQUE_GUESS


def tau_excited(t: float) -> float:
    return STICTION_TORQUE_GUESS + 6.0 * np.sin(2 * np.pi * 0.5 * t)


def tangential_direction(theta: float) -> np.ndarray:
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


def true_hinge_inertia(model: mujoco.MjModel) -> dict:
    """I_hinge about the vertical hinge axis from MuJoCo body mass properties.

    body_inertia is principal moments about the CoM. For this door the hinge
    is at the body origin with axis = body z, so:

        I_hinge = e_z^T * R_prin * diag(I_prin) * R_prin^T * e_z
                  + m * (x_com^2 + y_com^2)

    where R_prin comes from body_iquat (principal-frame orientation in the body).
    """
    bid = model.body("door").id
    m = float(model.body_mass[bid])
    I_prin = np.array(model.body_inertia[bid], dtype=float)  # [Ixx, Iyy, Izz]
    com = np.array(model.body_ipos[bid], dtype=float)        # CoM in body frame

    # principal axes orientation relative to body frame
    iquat = np.array(model.body_iquat[bid], dtype=float)  # [w, x, y, z]
    R = np.zeros((3, 3))
    mujoco.mju_quat2Mat(R.ravel(), iquat)

    I_com_body = R @ np.diag(I_prin) @ R.T
    # hinge axis = body z
    I_zz_com = float(I_com_body[2, 2])
    I_hinge = I_zz_com + m * (com[0] ** 2 + com[1] ** 2)

    return {
        "mass": m,
        "com_body": com,
        "I_prin": I_prin,
        "I_zz_com": I_zz_com,
        "I_hinge": I_hinge,
        "frictionloss": float(model.dof_frictionloss[0]),
        "damping": float(model.dof_damping[0]),
    }


def simulate(tau_fn) -> dict:
    """Forward-simulate; log hinge torque, qpos/qvel/qacc from MuJoCo."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    assert abs(model.opt.timestep - DT) < 1e-12

    hinge_qpos = model.joint("hinge").qposadr[0]
    hinge_dof = model.joint("hinge").dofadr[0]
    handle_sid = model.site("handle").id
    door_bid = model.body("door").id
    gt = true_hinge_inertia(model)

    t = np.arange(N_STEPS) * DT
    theta = np.zeros(N_STEPS)
    theta_dot = np.zeros(N_STEPS)
    theta_ddot = np.zeros(N_STEPS)
    tau_applied = np.zeros(N_STEPS)
    ncon = np.zeros(N_STEPS, dtype=int)

    for i in range(N_STEPS):
        th = float(data.qpos[hinge_qpos])
        tau_des = float(tau_fn(t[i]))
        force = (tau_des / HANDLE_DIST) * tangential_direction(th)

        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data, force, np.zeros(3),
            data.site_xpos[handle_sid].copy(), door_bid, data.qfrc_applied,
        )
        # generalized force about the hinge DOF = applied hinge torque
        tau_h = float(data.qfrc_applied[hinge_dof])

        mujoco.mj_step(model, data)

        theta[i] = float(data.qpos[hinge_qpos])
        theta_dot[i] = float(data.qvel[hinge_dof])
        theta_ddot[i] = float(data.qacc[hinge_dof])
        tau_applied[i] = tau_h
        ncon[i] = int(data.ncon)

    return dict(
        t=t, theta=theta, theta_dot=theta_dot, theta_ddot=theta_ddot,
        tau_applied=tau_applied, ncon=ncon, gt=gt, model=model,
    )


def fit_inertial_params(theta_dot, theta_ddot, tau_applied, mask):
    """Same LS as Phase 0: tau = I * theta_ddot + mu * sign(theta_dot)."""
    sign_thd = np.sign(theta_dot)
    Phi = np.stack([theta_ddot[mask], sign_thd[mask]], axis=1)
    Y = tau_applied[mask]
    params, residuals, rank, sv = np.linalg.lstsq(Phi, Y, rcond=None)
    cond = float(np.linalg.cond(Phi))
    return params, cond, Phi, Y


def analyze(label: str, log: dict) -> dict:
    gt = log["gt"]
    I_true = gt["I_hinge"]
    mu_true = gt["frictionloss"]  # MuJoCo Coulomb magnitude (not viscous)

    # same moving mask spirit as Phase 0
    moving = np.abs(log["theta_dot"]) > 0.02
    moving[:20] = False
    n_moving = int(moving.sum())

    params, cond, Phi, Y = fit_inertial_params(
        log["theta_dot"], log["theta_ddot"], log["tau_applied"], moving,
    )
    I_hat, mu_hat = float(params[0]), float(params[1])
    rel_err = abs(I_hat - I_true) / I_true * 100.0

    # residual of the Phase-0 model on MuJoCo data
    Y_hat = Phi @ params
    rmse = float(np.sqrt(np.mean((Y - Y_hat) ** 2))) if len(Y) else np.nan

    # structural signal strength
    thdd_rms = float(np.sqrt(np.mean(log["theta_ddot"][moving] ** 2))) if n_moving else 0.0
    thdd_max = float(np.max(np.abs(log["theta_ddot"][moving]))) if n_moving else 0.0

    print(f"=== {label} ===")
    print(f"  Ground truth (from MuJoCo body):")
    print(f"    mass          = {gt['mass']:.4f} kg")
    print(f"    CoM (body)    = {gt['com_body']}")
    print(f"    I_zz @ CoM    = {gt['I_zz_com']:.4f} kg*m^2")
    print(f"    I_hinge       = {I_true:.4f} kg*m^2   "
          f"(= I_zz_com + m*(x^2+y^2))")
    print(f"    frictionloss  = {mu_true:.4f} N*m")
    print(f"    damping       = {gt['damping']:.4f} N*m*s/rad  "
          f"(NOT in Phase-0 regressor)")
    print(f"  Trajectory:")
    print(f"    theta end     = {np.degrees(log['theta'][-1]):.2f} deg")
    print(f"    moving samples= {n_moving}/{N_STEPS}")
    print(f"    max |ncon|    = {log['ncon'].max()}  "
          f"(should be 0 if door clears floor/wall)")
    print(f"    theta_ddot RMS (moving) = {thdd_rms:.4f} rad/s^2")
    print(f"    theta_ddot max (moving) = {thdd_max:.4f} rad/s^2")
    print(f"  Least-squares fit (Phase-0 regressor):")
    print(f"    I_hinge hat   = {I_hat:.4f} kg*m^2")
    print(f"    mu_k hat      = {mu_hat:.4f} N*m")
    print(f"    I_hinge rel. error = {rel_err:.1f}%")
    print(f"    cond(Phi)     = {cond:.1f}")
    print(f"    residual RMSE = {rmse:.4f} N*m")
    print()

    return dict(
        label=label, I_true=I_true, I_hat=I_hat, mu_true=mu_true, mu_hat=mu_hat,
        rel_err=rel_err, cond=cond, rmse=rmse, thdd_rms=thdd_rms,
        n_moving=n_moving, damping=gt["damping"],
    )


def main():
    print("MuJoCo door observability (Phase 0 effect in a physics engine)\n")
    print("Regressor:  tau = I_hinge * theta_ddot + mu_k * sign(theta_dot)")
    print("Signals:    MuJoCo qvel / qacc (no finite differencing)\n")

    results = {}
    for label, tau_fn in [("quasi-static", tau_quasistatic), ("excited", tau_excited)]:
        log = simulate(tau_fn)
        results[label] = analyze(label, log)

    qs, ex = results["quasi-static"], results["excited"]
    print("--- Summary (compare to Phase 0 NumPy demo) ---")
    print(f"  Quasi-static: I error {qs['rel_err']:.1f}%,  cond={qs['cond']:.1f},  "
          f"thdd_rms={qs['thdd_rms']:.4f}")
    print(f"  Excited:      I error {ex['rel_err']:.1f}%,  cond={ex['cond']:.1f},  "
          f"thdd_rms={ex['thdd_rms']:.4f}")

    # Honest diagnosis if the effect is weak / inverted / polluted
    print("\n--- Diagnosis ---")
    if qs["n_moving"] < 50:
        print("  WARN: quasi-static barely moves — mask may be too thin for a fair LS fit.")
    if qs["thdd_rms"] > 0.5 * ex["thdd_rms"] and ex["thdd_rms"] > 0:
        print("  WARN: quasi-static theta_ddot RMS is not much smaller than excited;")
        print("        structural unobservability may be weaker than in the toy ODE.")
    if qs["rel_err"] < ex["rel_err"]:
        print("  WARN: excited did NOT beat quasi-static on I_hinge error.")
        print("        Effect not reproduced under this MuJoCo model + regressor.")
    else:
        print("  Excited recovers I_hinge better than quasi-static "
              f"({ex['rel_err']:.1f}% vs {qs['rel_err']:.1f}%).")

    if qs["damping"] > 0:
        print(f"  NOTE: viscous damping={qs['damping']} is present in MuJoCo but absent")
        print("        from the Phase-0 2-parameter model. Residual RMSE and mu bias")
        print("        partly reflect that mismatch, not just observability.")


if __name__ == "__main__":
    main()
