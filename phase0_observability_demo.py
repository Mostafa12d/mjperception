"""
Phase 0: Observability demo for online inertial parameter estimation of a revolute door.

Ground-truth model (vertical-hinge door, so gravity contributes no torque about the hinge):

    I_hinge * theta_ddot(t) = tau_applied(t) - mu_k * sign(theta_dot(t))

This is linear-in-parameters: theta_ddot and sign(theta_dot) form a regressor,
[I_hinge, mu_k] are the unknowns. We generate two applied-torque profiles:

  A) "quasi-static": torque is raised just enough to overcome friction and held
     roughly constant -> theta_ddot stays near zero almost everywhere.
  B) "excited": torque oscillates -> theta_ddot swings meaningfully.

We then fit [I_hinge, mu_k] via least squares from noisy, differentiated
position measurements (mimicking a real vision-based angle signal) and compare
recovered parameters + regressor conditioning between the two conditions.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Ground truth -----
I_TRUE = 0.8        # kg*m^2, lumped hinge inertia (I_cm + m*d^2)
MU_K_TRUE = 0.35     # N*m, kinetic friction torque
STICTION = 0.5       # N*m, threshold to start motion
DT = 0.002
T_END = 6.0
N = int(T_END / DT)

RNG = np.random.default_rng(0)


def simulate(tau_fn, theta0=0.0, theta_dot0=0.0, vel_noise_std=0.01, acc_noise_std=0.05):
    """Integrate the door dynamics under a given applied-torque function tau_fn(t).

    We return the simulator's own theta_dot / theta_ddot (with modest additive
    sensor noise), rather than reconstructing them by double-differentiating a
    noisy angle signal. This isolates the STRUCTURAL observability argument
    (does theta_ddot carry information about I_hinge at all?) from the separate
    engineering problem of noise-robust differentiation, which would otherwise
    muddy the comparison between conditions.
    """
    theta = np.zeros(N)
    theta_dot = np.zeros(N)
    theta_ddot = np.zeros(N)
    tau_applied = np.zeros(N)
    t = np.arange(N) * DT

    th, thd = theta0, theta_dot0
    moving = False
    for i in range(N):
        tau = tau_fn(t[i])
        tau_applied[i] = tau

        # Coulomb friction with stiction threshold
        if not moving:
            if abs(tau) > STICTION:
                moving = True
            friction = np.clip(tau, -MU_K_TRUE, MU_K_TRUE)  # static equilibrium
            thdd = (tau - friction) / I_TRUE if moving else 0.0
        else:
            friction = MU_K_TRUE * np.sign(thd) if thd != 0 else 0.0
            thdd = (tau - friction) / I_TRUE
            if abs(thd) < 1e-6 and abs(tau) < MU_K_TRUE:
                moving = False
                thdd = 0.0

        theta[i] = th
        theta_dot[i] = thd
        theta_ddot[i] = thdd
        th += thd * DT
        thd += thdd * DT

    # modest sensor noise on the (in practice, filtered/estimated) velocity and
    # acceleration signals -- small relative to the true excited-case swings
    theta_dot_meas = theta_dot + RNG.normal(0, vel_noise_std, size=N)
    theta_ddot_meas = theta_ddot + RNG.normal(0, acc_noise_std, size=N)
    return t, theta, theta_dot_meas, theta_ddot_meas, tau_applied


def fit_inertial_params(theta_dot, theta_ddot, tau_applied, mask):
    """Linear least squares fit of [I_hinge, mu_k] from
    tau_applied = I_hinge * theta_ddot + mu_k * sign(theta_dot)."""
    sign_thd = np.sign(theta_dot)
    Phi = np.stack([theta_ddot[mask], sign_thd[mask]], axis=1)
    Y = tau_applied[mask]
    params, residuals, rank, sv = np.linalg.lstsq(Phi, Y, rcond=None)
    cond_number = np.linalg.cond(Phi)
    return params, cond_number


# ----- Condition A: quasi-static -----
def tau_quasistatic(t):
    # ramp up to just above stiction, then hold ~constant
    if t < 0.3:
        return (STICTION * 1.1) * (t / 0.3)
    return STICTION * 1.1


# ----- Condition B: excited -----
def tau_excited(t):
    # oscillating torque -> real acceleration swings, still safely bounded
    base = STICTION * 1.1
    return base + 0.6 * np.sin(2 * np.pi * 0.5 * t)


results = {}
for label, tau_fn in [("quasi-static", tau_quasistatic), ("excited", tau_excited)]:
    t, theta, theta_dot, theta_ddot, tau_applied = simulate(tau_fn)

    # only fit once the door is actually moving (skip startup transient)
    moving_mask = np.abs(theta_dot) > 0.02
    moving_mask[:20] = False

    params, cond_number = fit_inertial_params(theta_dot, theta_ddot, tau_applied, moving_mask)
    results[label] = dict(t=t, theta=theta, I_hat=params[0], mu_hat=params[1], cond=cond_number)

    print(f"--- {label} ---")
    print(f"  True:      I_hinge = {I_TRUE:.4f}, mu_k = {MU_K_TRUE:.4f}")
    print(f"  Estimated: I_hinge = {params[0]:.4f}, mu_k = {params[1]:.4f}")
    print(f"  I_hinge relative error: {abs(params[0]-I_TRUE)/I_TRUE*100:.1f}%")
    print(f"  Regressor condition number: {cond_number:.1f}")
    print()

# ----- Plot -----
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for label, style in [("quasi-static", "C0"), ("excited", "C1")]:
    r = results[label]
    axes[0].plot(r["t"], r["theta"], style, label=label)
axes[0].set_ylabel("theta (rad)")
axes[0].legend()
axes[0].set_title("Door angle trajectories under the two interaction strategies")

labels = ["quasi-static", "excited"]
I_hats = [results[l]["I_hat"] for l in labels]
axes[1].bar(labels, I_hats, color=["C0", "C1"])
axes[1].axhline(I_TRUE, color="k", linestyle="--", label="ground truth I_hinge")
axes[1].set_ylabel("Estimated I_hinge (kg*m^2)")
axes[1].legend()
axes[1].set_title("Recovered inertia: quasi-static vs. mildly excited interaction")

plt.tight_layout()
plt.savefig("/home/claude/phase0_observability_result.png", dpi=150)
print("Saved plot to phase0_observability_result.png")
