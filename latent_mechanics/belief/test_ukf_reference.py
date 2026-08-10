"""
STEP 3 CHECKPOINT: validate the UKF core against filterpy's reference.

Nothing here touches the learned predictor, the latent basis, or any project
data. The point is to establish that the filter is numerically correct as a
filter, on synthetic nonlinear systems, before it is trusted with anything real.

The comparison is against ``filterpy.kalman.UnscentedKalmanFilter`` with
``MerweScaledSigmaPoints`` (Labbe, filterpy 1.4.5). Agreement is asserted to
1e-10 on every intermediate quantity, not just the final estimate -- sigma
points, weights, prior mean/covariance, innovation, innovation covariance,
Kalman gain and posterior -- because a filter can produce a plausible-looking
trajectory while getting the gain subtly wrong.

Run:
    python3.10 -m latent_mechanics.belief.test_ukf_reference
"""

from __future__ import annotations

import numpy as np

from latent_mechanics.belief.ukf import (
    MerweSigmaPoints,
    UnscentedKalmanFilter,
    nearest_pd,
    unscented_transform,
)

_FAILURES: list[str] = []
TOL = 1e-10


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def close(a, b, tol=TOL) -> tuple[bool, str]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    err = float(np.max(np.abs(a - b)))
    return err <= tol, f"max abs diff {err:.3e}"


# ---------------------------------------------------------------------------
# Synthetic systems
# ---------------------------------------------------------------------------

def fx_nl(x, dt=1.0):
    """Mildly nonlinear 3-state process."""
    return np.array([
        x[0] + dt * x[1],
        0.9 * x[1] + 0.1 * np.sin(x[2]),
        x[2] + 0.05 * x[0] * x[1],
    ])


def hx_nl(x):
    """2-D nonlinear measurement."""
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2 + 1.0), np.arctan2(x[1], x[0] + 3.0)])


# ---------------------------------------------------------------------------

def test_sigma_points_and_weights() -> None:
    print("\nSigma points and weights vs filterpy")
    from filterpy.kalman import MerweScaledSigmaPoints

    rng = np.random.default_rng(0)
    for n, alpha, beta, kappa in [(3, 1e-3, 2.0, 0.0), (4, 0.5, 2.0, 1.0),
                                  (6, 0.1, 2.0, 3.0 - 6)]:
        A = rng.normal(size=(n, n))
        P = A @ A.T + n * np.eye(n)
        x = rng.normal(size=n)

        ref = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        mine = MerweSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)

        ok, d = close(mine.compute(x, P), ref.sigma_points(x, P))
        check(f"n={n} a={alpha}: sigma points match", ok, d)
        Wm, Wc = mine.weights()
        ok1, d1 = close(Wm, ref.Wm)
        ok2, d2 = close(Wc, ref.Wc)
        check(f"n={n} a={alpha}: Wm matches", ok1, d1)
        check(f"n={n} a={alpha}: Wc matches", ok2, d2)
        check(f"n={n} a={alpha}: weights sum to 1", abs(Wm.sum() - 1.0) < 1e-12,
              f"{Wm.sum():.12f}")


def test_unscented_transform() -> None:
    print("\nUnscented transform vs filterpy")
    from filterpy.kalman import unscented_transform as ref_ut

    rng = np.random.default_rng(1)
    n = 4
    pts = MerweSigmaPoints(n=n, alpha=0.3, beta=2.0, kappa=0.0)
    Wm, Wc = pts.weights()
    sig = rng.normal(size=(2 * n + 1, n))
    Rn = np.diag(rng.uniform(0.1, 1.0, n))

    x1, P1 = unscented_transform(sig, Wm, Wc, Rn)
    x2, P2 = ref_ut(sig, Wm, Wc, Rn)
    ok1, d1 = close(x1, x2); ok2, d2 = close(P1, P2)
    check("transformed mean matches", ok1, d1)
    check("transformed covariance matches", ok2, d2)


def test_full_filter_trajectory() -> None:
    print("\nFull predict/update cycle vs filterpy (20 steps, nonlinear)")
    from filterpy.kalman import MerweScaledSigmaPoints
    from filterpy.kalman import UnscentedKalmanFilter as RefUKF

    rng = np.random.default_rng(2)
    n, m = 3, 2
    Q = np.diag([0.01, 0.02, 0.005])
    R = np.diag([0.05, 0.02])
    x0 = np.array([0.5, -0.2, 0.1])
    P0 = np.diag([1.0, 0.5, 0.3])

    ref_pts = MerweScaledSigmaPoints(n=n, alpha=0.3, beta=2.0, kappa=0.0)
    ref = RefUKF(dim_x=n, dim_z=m, dt=1.0, fx=fx_nl, hx=hx_nl, points=ref_pts)
    ref.x, ref.P, ref.Q, ref.R = x0.copy(), P0.copy(), Q.copy(), R.copy()

    mine = UnscentedKalmanFilter(
        dim_x=n, dim_z=m, points=MerweSigmaPoints(n=n, alpha=0.3, beta=2.0, kappa=0.0),
        fx=lambda s: fx_nl(s, 1.0), hx=hx_nl, Q=Q, R=R, x0=x0, P0=P0)

    truth = x0.copy()
    worst = {"x": 0.0, "P": 0.0, "K": 0.0, "y": 0.0, "S": 0.0, "prior": 0.0}
    for k in range(20):
        truth = fx_nl(truth) + rng.normal(0, 0.05, n)
        z = hx_nl(truth) + rng.normal(0, 0.05, m)

        ref.predict(); mine.predict()
        worst["prior"] = max(worst["prior"], float(np.max(np.abs(ref.x - mine.x))))
        ref.update(z); mine.update(z)

        worst["x"] = max(worst["x"], float(np.max(np.abs(ref.x - mine.x))))
        worst["P"] = max(worst["P"], float(np.max(np.abs(ref.P - mine.P))))
        worst["K"] = max(worst["K"], float(np.max(np.abs(ref.K - mine.state.K))))
        worst["y"] = max(worst["y"], float(np.max(np.abs(ref.y - mine.state.y))))
        worst["S"] = max(worst["S"], float(np.max(np.abs(ref.S - mine.state.S))))

    for key, label in (("prior", "prior mean after predict"),
                       ("x", "posterior mean"), ("P", "posterior covariance"),
                       ("K", "Kalman gain"), ("y", "innovation"),
                       ("S", "innovation covariance")):
        check(f"{label} matches over 20 steps", worst[key] <= TOL,
              f"max abs diff {worst[key]:.3e}")


def test_batched_hx_matches_pointwise() -> None:
    """The batched measurement path must be numerically identical to the loop."""
    print("\nBatched hx == pointwise hx")
    rng = np.random.default_rng(3)
    n, m = 5, 2
    pts = MerweSigmaPoints(n=n, alpha=0.4, beta=2.0, kappa=0.0)
    A = rng.normal(size=(n, n)); P0 = A @ A.T + n * np.eye(n)
    x0 = rng.normal(size=n)
    Q = 0.01 * np.eye(n); R = 0.05 * np.eye(m)

    def hx(x):
        return np.array([np.tanh(x[0] + 0.3 * x[2]), np.sin(x[1]) + 0.1 * x[3] ** 2])

    def hx_batch(S):
        return np.stack([hx(s) for s in S])

    f1 = UnscentedKalmanFilter(n, m, pts, fx=None, hx=hx, Q=Q, R=R, x0=x0, P0=P0)
    f2 = UnscentedKalmanFilter(n, m, pts, fx=None, hx=hx, Q=Q, R=R, x0=x0, P0=P0)
    for _ in range(10):
        z = rng.normal(size=m)
        f1.predict(); f1.update(z)
        f2.predict(); f2.update(z, hx_batch=hx_batch)
    ok1, d1 = close(f1.x, f2.x); ok2, d2 = close(f1.P, f2.P)
    check("means agree", ok1, d1)
    check("covariances agree", ok2, d2)


def test_linear_case_equals_kalman() -> None:
    """On a linear system, which UKF variant reproduces the exact Kalman filter?

    The filterpy convention reuses the fx-propagated sigma points for the
    measurement update, so those points carry covariance F P F^T while the prior
    covariance is F P F^T + Q. The gain is therefore computed without the process
    noise and the filter is NOT exactly the Kalman filter when Q != 0 -- it is
    when Q = 0. Regenerating the sigma points from the prior restores exactness.
    Both behaviours are asserted, because the difference is a design choice this
    branch has to make deliberately rather than inherit.
    """
    print("\nLinear system: which variant equals the textbook Kalman filter?")
    rng = np.random.default_rng(4)
    n, m = 3, 2
    F = np.array([[1.0, 0.1, 0.0], [0.0, 0.95, 0.05], [0.0, 0.0, 0.9]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    Q = np.diag([0.01, 0.01, 0.02]); R = np.diag([0.1, 0.05])
    x = np.array([1.0, 0.5, -0.3]); P = np.diag([0.5, 0.5, 0.5])

    def run(Qm, regen):
        ukf = UnscentedKalmanFilter(n, m, MerweSigmaPoints(n, 0.5, 2.0, 0.0),
                                    fx=lambda s: F @ s, hx=lambda s: H @ s,
                                    Q=Qm, R=R, x0=x.copy(), P0=P.copy(),
                                    regenerate_sigma_points=regen)
        xk, Pk = x.copy(), P.copy()
        wx = wP = 0.0
        r = np.random.default_rng(4)
        for _ in range(15):
            z = r.normal(size=m)
            xk = F @ xk; Pk = F @ Pk @ F.T + Qm
            S = H @ Pk @ H.T + R
            K = Pk @ H.T @ np.linalg.inv(S)
            xk = xk + K @ (z - H @ xk); Pk = Pk - K @ S @ K.T
            ukf.predict(); ukf.update(z)
            wx = max(wx, float(np.max(np.abs(xk - ukf.x))))
            wP = max(wP, float(np.max(np.abs(Pk - ukf.P))))
        return wx, wP

    wx, wP = run(np.zeros((n, n)), False)
    check("Q=0, filterpy variant: matches exact KF", max(wx, wP) < 1e-9,
          f"{max(wx, wP):.3e}")
    wx, wP = run(Q, True)
    check("Q>0, regenerating variant: matches exact KF", max(wx, wP) < 1e-9,
          f"{max(wx, wP):.3e}")
    wx, wP = run(Q, False)
    check("Q>0, filterpy variant: deviates, as expected", max(wx, wP) > 1e-4,
          f"deviation only {max(wx, wP):.3e}")


def test_numerical_hygiene() -> None:
    print("\nNumerical hygiene")
    A = np.array([[1.0, 0.9], [0.9, 0.81]])         # rank deficient
    B = nearest_pd(A - 1e-9 * np.eye(2), floor=1e-10)
    check("nearest_pd yields a Cholesky-able matrix",
          np.all(np.linalg.eigvalsh(B) > 0))
    check("nearest_pd barely moves an already-PD matrix",
          float(np.max(np.abs(nearest_pd(np.eye(3)) - np.eye(3)))) < 1e-12)

    pts = MerweSigmaPoints(n=2, alpha=1e-3, beta=2.0, kappa=0.0)
    s = pts.compute(np.zeros(2), np.eye(2))
    Wm, Wc = pts.weights()
    mu, cov = unscented_transform(s, Wm, Wc, None)
    check("sigma set reproduces its own mean", float(np.max(np.abs(mu))) < 1e-12)
    check("sigma set reproduces its own covariance",
          float(np.max(np.abs(cov - np.eye(2)))) < 1e-9,
          f"{float(np.max(np.abs(cov - np.eye(2)))):.3e}")


def main() -> None:
    print("=" * 74)
    print("STEP 3 CHECKPOINT -- UKF core validated against filterpy 1.4.5")
    print("=" * 74)
    test_sigma_points_and_weights()
    test_unscented_transform()
    test_full_filter_trajectory()
    test_batched_hx_matches_pointwise()
    test_linear_case_equals_kalman()
    test_numerical_hygiene()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed -- UKF core matches the reference to 1e-10")


if __name__ == "__main__":
    main()
