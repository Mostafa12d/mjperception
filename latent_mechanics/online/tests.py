"""Self-checks for Stage-2 online adaptation.

The load-bearing ones are the frozen-network checks, verified three ways
(``requires_grad``, gradient leakage, weight checksum), because a violation would
just make the experiments look better.

    python3.10 -m latent_mechanics.online.tests
"""

from __future__ import annotations

import numpy as np
import torch

from latent_mechanics.model import MechanicsDynamicsModel
from latent_mechanics.online.adaptor import (
    GradientLatentAdaptor,
    OnlineLatentAdaptor,
)
from latent_mechanics.online.loop import (
    AdaptationLog,
    init_strategies,
    run_online_adaptation,
)
from latent_mechanics.online.rls_adaptor import RLSAdaptor
from latent_mechanics.online.viz import LatentPCA

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def _toy_model(embed_dim: int = 8) -> MechanicsDynamicsModel:
    torch.manual_seed(0)
    m = MechanicsDynamicsModel(embed_dim=embed_dim, hidden_sizes=[32, 32])
    with torch.no_grad():
        m.set_norm_stats({
            "state_mean": torch.zeros(2), "state_std": torch.ones(2),
            "action_mean": torch.zeros(1), "action_std": torch.ones(1),
            "delta_mean": torch.zeros(2), "delta_std": torch.full((2,), 0.05),
        })
    return m


def _toy_stream(n: int = 400, seed: int = 0):
    """A deterministic linear plant, so a correct adaptor must improve on it."""
    rng = np.random.default_rng(seed)
    s = np.array([0.0, 0.0], dtype=np.float32)
    out = []
    for _ in range(n):
        a = np.array([rng.uniform(-3, 3)], dtype=np.float32)
        acc = (a[0] - 0.4 * s[1] - 1.2 * s[0]) / 3.0
        ns = np.array([s[0] + 0.02 * s[1], s[1] + 0.02 * acc], dtype=np.float32)
        out.append((s.copy(), a, ns.copy()))
        s = ns
    return out


def test_frozen_network() -> None:
    print("\nFrozen network (the Stage-2 constraint)")
    model = _toy_model()
    # Deliberately hand it a model whose weights still require grad.
    for p in model.parameters():
        p.requires_grad_(True)

    ad = GradientLatentAdaptor(model, lr=0.05, window=8)
    check("constructor freezes the network",
          all(not p.requires_grad for p in ad.model.parameters()))

    before = [p.detach().clone() for p in ad.model.parameters()]
    stream = _toy_stream(200)
    for s, a, ns in stream:
        ad.observe(s, a, ns)

    check("no weight changed after 200 updates",
          all(torch.equal(b, p) for b, p in zip(before, ad.model.parameters())))
    check("no gradient accumulated on any weight",
          all(p.grad is None for p in ad.model.parameters()))
    try:
        ad.assert_network_unchanged()
        ok = True
    except RuntimeError:
        ok = False
    check("assert_network_unchanged passes", ok)

    # And that the guard actually fires when a weight is tampered with.
    with torch.no_grad():
        next(iter(ad.model.parameters())).add_(1e-3)
    try:
        ad.assert_network_unchanged()
        fired = False
    except RuntimeError:
        fired = True
    check("assert_network_unchanged detects tampering", fired)

    # The optimiser must own exactly one tensor: the latent.
    owned = [p for g in ad._opt.param_groups for p in g["params"]]
    check("optimiser owns only the latent",
          len(owned) == 1 and owned[0] is ad._z, f"{len(owned)} tensors")


def test_adaptor_learns() -> None:
    print("\nGradient adaptor actually adapts")
    model = _toy_model()
    ad = GradientLatentAdaptor(model, lr=0.05, window=32)
    stream = _toy_stream(600)
    log = run_online_adaptation(ad, stream, door_id=0)

    check("log length matches the stream", len(log) == len(stream))
    check("latent moved away from its init",
          np.linalg.norm(log.latents[-1] - np.zeros(model.embed_dim)) > 1e-3)
    early, late = log.rmse(0, last=100), log.final_rmse(0, 0.25)
    check("prediction error decreased", late < early, f"{early:.3e} -> {late:.3e}")
    check("errors are one-step-ahead (prediction precedes update)",
          log.error.shape == (len(stream), 2))


def test_online_not_batch() -> None:
    """Belief must change on every step, not once at the end."""
    print("\nOnline, not whole-trajectory")
    ad = GradientLatentAdaptor(_toy_model(), lr=0.05, window=1)
    log = run_online_adaptation(ad, _toy_stream(120), door_id=0)
    steps = np.linalg.norm(np.diff(log.latents, axis=0), axis=1)
    check("latent changes at essentially every interaction",
          float((steps > 0).mean()) > 0.95, f"{100 * (steps > 0).mean():.0f}% of steps")
    check("update cost does not grow with stream length",
          log.update_seconds[-20:].mean() < 5 * log.update_seconds[:20].mean() + 1e-4)


def test_interface_is_algorithm_agnostic() -> None:
    """A non-gradient update rule must drop in without touching the driver."""
    print("\nInterface does not assume gradient descent")

    class RandomWalkAdaptor(OnlineLatentAdaptor):
        """Stand-in for a future Kalman/learned/Bayesian rule."""
        name = "random-walk"

        def _update(self, state, action, next_state):
            with torch.no_grad():
                self._z.add_(torch.randn_like(self._z) * 1e-3)
            return 0.0, {"rule": 1.0}

    ad = RandomWalkAdaptor(_toy_model())
    log = run_online_adaptation(ad, _toy_stream(50), door_id=0)
    check("a non-gradient adaptor runs through the same driver", len(log) == 50)
    check("it still cannot touch the network", True)
    ad.assert_network_unchanged()

    b = ad.belief()
    check("belief() exposes a mean and a covariance slot",
          "mean" in b and "cov" in b)


def test_rls_adaptor() -> None:
    print("\nRLS baseline behind the same interface")
    ad = RLSAdaptor(dt=0.02, n_params=5)
    stream = _toy_stream(600)
    log = run_online_adaptation(ad, stream, door_id=0)
    check("RLS runs through the same driver", len(log) == len(stream))
    early, late = log.rmse(0, last=100), log.final_rmse(0, 0.25)
    check("RLS prediction error decreased", late < early, f"{early:.3e} -> {late:.3e}")

    # The toy plant is exactly the 5-parameter model: I=3, b=0.4, k=1.2, mu=0.
    I_hat = ad.params[0]
    check("RLS recovers the toy plant inertia (true 3.0)",
          abs(I_hat - 3.0) < 0.3, f"I_hat={I_hat:.3f}")

    b = ad.belief()
    check("RLS exposes covariance through the same belief() API",
          b["cov"] is not None and b["cov"].shape == (5, 5))


def test_prediction_is_before_update() -> None:
    """The reported error must not benefit from the transition being scored."""
    print("\nPrequential protocol")
    ad = GradientLatentAdaptor(_toy_model(), lr=0.5, window=1)
    s, a, ns = _toy_stream(1)[0]
    expected = ad.predict(s, a)
    step = ad.observe(s, a, ns)
    check("reported prediction equals the pre-update prediction",
          np.allclose(step.prediction, expected, atol=1e-7))
    after = ad.predict(s, a)
    check("belief did change as a result of observing",
          not np.allclose(after, expected, atol=1e-9))


def test_init_strategies_and_pca() -> None:
    print("\nInit strategies and PCA frame")
    rng = np.random.default_rng(0)
    train = rng.normal(size=(20, 8)).astype(np.float32) * 2.0
    inits = init_strategies(train, seed=0)
    check("four strategies provided",
          set(inits) == {"zero", "random_trained", "mean", "medoid"})
    check("zero is zero", np.allclose(inits["zero"], 0))
    check("mean matches the table mean", np.allclose(inits["mean"], train.mean(0), atol=1e-5))
    check("medoid is an actual training row",
          any(np.allclose(inits["medoid"], r) for r in train))
    check("random_trained is an actual training row",
          any(np.allclose(inits["random_trained"], r) for r in train))

    pca = LatentPCA.fit(train)
    check("PCA frame is 2-D", pca.components.shape == (2, 8))
    check("training projection matches the stored one",
          np.allclose(pca.project(train), pca.train_xy, atol=1e-5))
    check("projection is a fixed affine map (frame does not drift)",
          np.allclose(pca.project(train[:3]), pca.train_xy[:3], atol=1e-5))


def test_log_metrics() -> None:
    print("\nAdaptationLog metrics")
    err = np.zeros((100, 2))
    err[:50, 0] = 1.0
    err[50:, 0] = 0.1
    log = AdaptationLog(name="t", door_id=0, error=err, loss=np.zeros(100),
                        latents=np.zeros((100, 4)), update_seconds=np.ones(100) * 1e-4)
    check("final_rmse uses the tail", abs(log.final_rmse(0, 0.25) - 0.1) < 1e-9)
    check("rmse over all steps sits between", 0.1 < log.rmse(0) < 1.0)
    r = log.rolling_rmse(0, 10)
    check("rolling curve has one point per step", len(r) == 100)
    check("rolling curve drops after the change", r[-1] < r[40])
    check("steps_to finds the crossing", log.steps_to(0.5, 0, 10, hold=10) is not None)
    check("steps_to returns None when never reached", log.steps_to(1e-6, 0, 10) is None)


def main() -> None:
    print("latent_mechanics.online self-checks")
    test_frozen_network()
    test_adaptor_learns()
    test_online_not_batch()
    test_interface_is_algorithm_agnostic()
    test_rls_adaptor()
    test_prediction_is_before_update()
    test_init_strategies_and_pca()
    test_log_metrics()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
