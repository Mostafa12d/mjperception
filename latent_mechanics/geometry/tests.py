"""
Self-checks for the latent-geometry investigation.

The analysis is only as trustworthy as its statistics, so the checks here are
mostly about the measurement instruments rather than the pipeline: that the
matched null actually behaves like a null, that the multimodality machinery can
detect real modes when they exist and stays quiet when they do not, and that the
Jacobian code computes PER-SAMPLE derivatives (an earlier version summed over the
batch and inflated the linearisation error by ~200x).

Run:
    python3.10 -m latent_mechanics.geometry.tests
"""

from __future__ import annotations

import numpy as np
import torch

from latent_mechanics.geometry import analyses as an
from latent_mechanics.model import MechanicsDynamicsModel

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def test_null_is_a_null() -> None:
    print("\nMatched null preserves the moments it should")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(120, 16)) @ rng.normal(size=(16, 16))
    nulls = an.matched_null(z, 8)
    check("null has the same shape", all(n.shape == z.shape for n in nulls))
    m_err = np.mean([np.linalg.norm(n.mean(0) - z.mean(0)) for n in nulls])
    c_err = np.mean([np.linalg.norm(np.cov(n, rowvar=False) - np.cov(z, rowvar=False))
                     for n in nulls]) / np.linalg.norm(np.cov(z, rowvar=False))
    # A correct null draws from N(mu, cov), so its sample mean is off by exactly
    # the sampling error sqrt(tr(cov)/n) -- not by zero. Judge against that,
    # not against an arbitrary constant.
    expected = np.sqrt(np.trace(np.cov(z, rowvar=False)) / len(z))
    check("null matches the mean to within sampling error",
          m_err < 2.0 * expected, f"{m_err:.3f} vs expected {expected:.3f}")
    check("null matches the covariance", c_err < 0.35, f"relative {c_err:.3f}")


def test_detects_real_modes() -> None:
    print("\nMultimodality machinery detects modes when they exist")
    rng = np.random.default_rng(1)
    far = np.concatenate([rng.normal(0, 0.4, (60, 8)),
                          rng.normal(9, 0.4, (60, 8))])
    ev = an.multimodality_evidence(far, k_max=6, n_null=6, covariance_type="diag")
    check("well-separated modes beat the null on silhouette",
          ev["best_silhouette"] > ev["null_silhouette_max"],
          f"{ev['best_silhouette']:.3f} vs {ev['null_silhouette_max']:.3f}")
    check("and are called significant", ev["silhouette_p_value"] < 0.2,
          f"p={ev['silhouette_p_value']:.3f}")


def test_quiet_on_unimodal() -> None:
    print("\n...and stays quiet when they do not")
    rng = np.random.default_rng(2)
    z = rng.normal(size=(120, 16))
    ev = an.multimodality_evidence(z, k_max=6, n_null=6, covariance_type="diag")
    check("unimodal data does not beat its own null",
          ev["silhouette_p_value"] > 0.05, f"p={ev['silhouette_p_value']:.3f}")


def test_jacobian_is_per_sample() -> None:
    print("\nJacobians are per-sample, not batch-summed")
    torch.manual_seed(0)
    m = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32]).freeze()
    rng = np.random.default_rng(0)
    S = rng.normal(size=(64, 2)).astype(np.float32)
    A = rng.normal(size=(64, 1)).astype(np.float32)
    z0 = np.zeros(8)

    # The relative first-order error must not depend on how many samples are
    # scored; a batch-summed Jacobian makes it grow with n.
    e_small = an.linearization_error(m, z0, S[:8], A[:8], (0.05,), n_points=8, n_dirs=3)
    e_large = an.linearization_error(m, z0, S, A, (0.05,), n_points=64, n_dirs=3)
    check("linearisation error is independent of batch size",
          abs(e_small[0.05] - e_large[0.05]) < 0.05,
          f"{e_small[0.05]:.3f} vs {e_large[0.05]:.3f}")
    check("small steps linearise well", e_large[0.05] < 0.15, f"{e_large[0.05]:.3f}")

    growing = an.linearization_error(m, z0, S, A, (0.05, 0.5, 2.0), n_points=32, n_dirs=3)
    check("error grows with step size",
          growing[0.05] < growing[0.5] < growing[2.0], str(growing))

    js = an.jacobian_stats(m, rng.normal(size=(10, 8)), S, A, n_points=20)
    check("Jacobian stats are finite and positive",
          np.isfinite(js["norm_median"]) and js["norm_median"] > 0)


def test_interpolation_and_spectrum() -> None:
    print("\nInterpolation profile and spectrum")
    torch.manual_seed(0)
    m = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32]).freeze()
    n = 200
    s = torch.randn(n, 2) * 0.2
    a = torch.randn(n, 1)
    ns = s + torch.randn(n, 2) * 0.01
    p = an.interpolation_profile(m, np.zeros(8), np.ones(8), (s, a, ns), (s, a, ns),
                                 n_steps=9)
    check("profile has one point per alpha", len(p["err_a"]) == 9)
    check("barrier ratio is finite and positive",
          np.isfinite(p["barrier_ratio"]) and p["barrier_ratio"] > 0)

    rng = np.random.default_rng(0)
    one = np.zeros((100, 8)); one[:, 0] = rng.normal(size=100)
    check("spectrum finds effective dim ~1 for a 1-D cloud",
          abs(an.spectrum(one)["effective_dim"] - 1.0) < 0.2)
    iso = rng.normal(size=(600, 8))
    check("and ~8 for an isotropic one",
          an.spectrum(iso)["effective_dim"] > 7.0)


def test_read_only() -> None:
    print("\nAnalysis never modifies model weights")
    torch.manual_seed(0)
    m = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32]).freeze()
    before = [p.detach().clone() for p in m.parameters()]
    rng = np.random.default_rng(0)
    S = rng.normal(size=(32, 2)).astype(np.float32)
    A = rng.normal(size=(32, 1)).astype(np.float32)
    an.jacobian_stats(m, rng.normal(size=(5, 8)), S, A, n_points=10)
    an.linearization_error(m, np.zeros(8), S, A, (0.1,), n_points=8, n_dirs=2)
    s = torch.as_tensor(S); a = torch.as_tensor(A)
    an.fit_oracle_latent(m, s, a, s + 0.01, np.zeros(8), steps=20)
    check("all weights unchanged after every analysis",
          all(torch.equal(b, p) for b, p in zip(before, m.parameters())))
    check("no gradients left on weights", all(p.grad is None for p in m.parameters()))


def main() -> None:
    print("latent_mechanics.geometry self-checks")
    test_null_is_a_null()
    test_detects_real_modes()
    test_quiet_on_unimodal()
    test_jacobian_is_per_sample()
    test_interpolation_and_spectrum()
    test_read_only()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
