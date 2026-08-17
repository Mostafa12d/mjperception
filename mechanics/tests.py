"""Self-checks for the experimental core, exiting non-zero on failure.

    python3.10 -m mechanics.tests

Two kinds of check, and the second kind is the point of the whole exercise:

  1. UNIT -- the new types and components behave as documented.
  2. EQUIVALENCE -- for every migrated estimator, the new path and the OLD path
     produce the same numbers on the same stream, to 1e-12 or exactly. This is
     what makes the refactor a refactor. If these fail, a published result moved.

Equivalence is checked in-process, running both implementations side by side,
rather than against pinned files -- so it cannot pass by comparing two copies of
the same stale artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import MechanicsDynamicsModel
from latent_mechanics.online.adaptor import (
    GradientLatentAdaptor,
    StaticLatentAdaptor,
)
from latent_mechanics.online.loop import episode_stream, run_online_adaptation
from latent_mechanics.online.rls_adaptor import RLSAdaptor
from mechanics import (
    FullLatent,
    IdentityObservation,
    JointSensor,
    MethodConfig,
    Transition,
    Workspace,
    build_method,
    run,
    transitions_from_dataset,
)
from mechanics.metrics import motion_scale, nrmse, score
from mechanics.observation import apply_to_sequence
from mechanics.predictor import AnalyticalPredictor, MisspecifiedPredictor
from mechanics.representation import PhysicalParameters, ReducedLatent
from mechanics.types import Belief

DATA = "data/door_mechanics.npz"
CKPT = "runs/latent_mechanics/base/best.pt"
ALL_FAMILIES = "runs/latent_mechanics/geometry/runs/all_families/best.pt"
ALL_FAMILIES_DATA = "runs/latent_mechanics/geometry/data_all_families.npz"
BASIS = "runs/latent_mechanics/belief/latent_basis.npz"

_failures: list[str] = []
_checks = 0


def check(cond: bool, msg: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


def close(a, b, tol: float = 1e-12) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    return bool(np.max(np.abs(a - b)) <= tol) if a.size else True


def maxdiff(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _toy_model(seed: int = 0) -> MechanicsDynamicsModel:
    torch.manual_seed(seed)
    m = MechanicsDynamicsModel(embed_dim=4, hidden_sizes=[16, 16])
    m.set_norm_stats({
        "state_mean": torch.zeros(2), "state_std": torch.ones(2),
        "action_mean": torch.zeros(1), "action_std": torch.ones(1),
        "delta_mean": torch.zeros(2), "delta_std": torch.full((2,), 0.05),
    })
    return m.freeze()


def _toy_stream(n: int = 60, seed: int = 0) -> list[Transition]:
    rng = np.random.default_rng(seed)
    out = []
    s = np.array([0.1, 0.0], dtype=np.float32)
    for _ in range(n):
        a = np.array([rng.normal(0, 2)], dtype=np.float32)
        ns = (s + np.array([s[1] * 0.02, a[0] * 0.01], dtype=np.float32)).astype(np.float32)
        out.append(Transition(obs=s.copy(), action=a, next_obs=ns.copy()))
        s = ns
    return out


def _have(path: str) -> bool:
    return Path(path).exists()


# --------------------------------------------------------------------------
# 1. unit checks
# --------------------------------------------------------------------------

def test_types() -> None:
    section("Types")
    t = Transition(obs=np.zeros(2), action=np.zeros(1), next_obs=np.ones(2))
    check(close(t.target, np.ones(2)), "target falls back to next_obs when truth is absent")

    t2 = Transition(obs=np.zeros(2), action=np.zeros(1), next_obs=np.ones(2),
                    truth=np.full(2, 0.5))
    check(close(t2.target, np.full(2, 0.5)),
          "target is the clean truth when the stream carries it")

    b = Belief(mean=[1.0, 2.0, 3.0], space="test")
    check(b.dim == 3 and b.mean.dtype == np.float64, "Belief coerces mean to float64 1-D")


def test_representations() -> None:
    section("Mechanics representations")
    z = np.arange(16, dtype=np.float64)
    full = FullLatent(init=z)
    check(full.dim == 16 and close(full.to_predictor(full.initial()), z),
          "FullLatent is the identity chart")

    phys = PhysicalParameters(init=np.array([5.0, 3.0, 0.2, 0.0, 0.0]))
    check(phys.dim == 5 and phys.names == ("I", "mu", "b", "k", "c"),
          "PhysicalParameters names its five coordinates")

    if not _have(BASIS):
        print("  [skip] reduced-latent round trip (no basis artifact)")
        return
    rep = ReducedLatent.from_path(BASIS, dim=6, init=np.zeros(16))
    x = rep.initial()
    check(x.shape == (6,), "ReducedLatent encodes into a 6-D chart")
    # projection is idempotent: decode then re-encode must return the same point
    check(close(rep.from_predictor(rep.to_predictor(x)), x, 1e-10),
          "ReducedLatent encode/decode round-trips on the chart")


def test_observation_models() -> None:
    section("Observation models")
    rng = np.random.default_rng(0)
    states = np.cumsum(np.ones((20, 2)), axis=0).astype(np.float32)

    ident = IdentityObservation()
    check(close(ident.observe(states, 0.02, rng), states),
          "IdentityObservation is exactly the identity")

    obs, next_obs = apply_to_sequence(ident, states[:-1], states[1:], 0.02, rng)
    check(close(obs, states[:-1]) and close(next_obs, states[1:]),
          "apply_to_sequence re-splits a sequence without shifting it")

    # the invariant that matters: a shared state is observed ONCE
    noisy = JointSensor(theta_sigma=0.01)
    o, n = apply_to_sequence(noisy, states[:-1], states[1:], 0.02,
                             np.random.default_rng(3))
    check(close(o[1:], n[:-1]),
          "a state shared by two transitions gets ONE noise draw, not two")

    part = PartialObservationCheck()
    check(part, "PartialObservation drops channels rather than zeroing them")


def PartialObservationCheck() -> bool:
    from mechanics.observation import PartialObservation
    rng = np.random.default_rng(0)
    states = np.ones((5, 2), dtype=np.float32)
    out = PartialObservation(keep=(0,)).observe(states, 0.02, rng)
    return out.shape == (5, 1)


def test_predictors() -> None:
    section("Predictors")
    m = _toy_model()
    rep = FullLatent(init=np.zeros(4))
    from mechanics.predictor import LatentNetworkPredictor
    pred = LatentNetworkPredictor(model=m, representation=rep)

    b = Belief(mean=np.zeros(4))
    out = pred.predict(np.array([0.1, 0.2]), np.array([1.0]), b)
    check(out.shape == (2,), "LatentNetworkPredictor returns a next observation")

    # h(x) for a single point must agree with the one-point batch
    single = pred.predict_measurement(np.array([0.1, 0.2]), np.array([1.0]),
                                      np.zeros((1, 4)))
    check(single.shape == (1, 2), "predict_measurement is batched over sigma points")

    # forward() and raw_output() must be consistent: s + denorm(delta) == next
    y = pred.measurement(np.array([0.1, 0.2]), out)
    check(close(y, single[0], 1e-5),
          "measurement(obs, predict(obs)) equals the predicted measurement")

    pred.assert_unchanged()
    check(True, "frozen-network checksum survives prediction")

    mis = MisspecifiedPredictor(inner=pred, gain=1.0, bias=0.0)
    check(close(mis.predict(np.array([0.1, 0.2]), np.array([1.0]), b), out, 1e-6),
          "MisspecifiedPredictor with gain=1, bias=0 is the wrapped predictor")

    mis2 = MisspecifiedPredictor(inner=pred, gain=2.0)
    d1 = mis2.predict(np.array([0.1, 0.2]), np.array([1.0]), b) - np.array([0.1, 0.2])
    d0 = out - np.array([0.1, 0.2])
    check(close(d1, 2 * d0, 1e-5), "MisspecifiedPredictor scales the predicted delta")


def test_analytical_predictor_matches_rls() -> None:
    section("AnalyticalPredictor reproduces RLSAdaptor.predict")
    rep = PhysicalParameters(init=np.array([4.0, 2.5, 0.3, 1.2, 0.1]))
    pred = AnalyticalPredictor(dt=0.02, representation=rep, n_substeps=10)
    legacy = RLSAdaptor(dt=0.02, n_params=5, n_substeps=10)
    legacy._rls.theta[:] = rep.init

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        s = np.array([rng.uniform(-0.1, 2.0), rng.uniform(-2, 2)])
        a = np.array([rng.uniform(-10, 10)])
        worst = max(worst, maxdiff(pred.predict(s, a, Belief(mean=rep.init)),
                                   legacy.predict(s, a)))
    check(worst == 0.0,
          f"analytical ODE is bit-identical to the RLS integrator (max diff {worst:.1e})")


def test_metrics() -> None:
    section("Metrics")
    trs = _toy_stream(40)
    scale = motion_scale(trs)
    check(scale.shape == (2,) and np.all(scale > 0), "motion_scale is per-dimension and positive")

    m = _toy_model()
    ws_rep = FullLatent(init=np.zeros(4))
    from mechanics.predictor import LatentNetworkPredictor
    from mechanics.estimators import StaticEstimator
    pred = LatentNetworkPredictor(model=m, representation=ws_rep)
    tr = run(StaticEstimator(pred), pred, trs, object_id=0)
    s = score(tr, trs)
    check(s["n_steps"] == 40, "score reports the stream length")
    check(np.isfinite(s["angle_nrmse_final"]), "tail nRMSE is finite")
    check(s["innovation_space"] == "normalised_delta",
          "the trace declares which space its innovation lives in")


def test_loop_protocol() -> None:
    section("Loop protocol")
    m = _toy_model()
    from mechanics.estimators import GradientEstimator
    from mechanics.predictor import LatentNetworkPredictor

    rep = FullLatent(init=np.zeros(4))
    pred = LatentNetworkPredictor(model=m, representation=rep)
    est = GradientEstimator(pred, lr=0.05, window=8)
    trs = _toy_stream(50)
    tr = run(est, pred, trs, object_id=0)

    check(len(tr) == 50, "one record per transition")
    check(tr.beliefs.shape == (50, 4), "one belief per step")
    check(tr.innovation.shape == (50, 2), "one innovation per step")

    # prequential: the step-0 prediction must use the PRIOR, not the updated belief
    prior_pred = pred.predict(trs[0].obs, trs[0].action, Belief(mean=rep.initial()))
    check(close(tr.error[0] + trs[0].target, prior_pred, 1e-6),
          "the step-0 prediction uses the belief held BEFORE the update")

    check(not close(tr.beliefs[-1], rep.initial(), 1e-9), "the belief actually moved")
    pred.assert_unchanged()
    check(True, "the predictor network is unchanged after a full run")


# --------------------------------------------------------------------------
# 2. equivalence with the legacy implementations
# --------------------------------------------------------------------------

def test_stream_equivalence() -> None:
    section("EQUIVALENCE  stream construction")
    if not _have(DATA):
        print("  [skip] no dataset at " + DATA)
        return
    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=True)
    did = int(ds.door_ids[0])

    legacy = episode_stream(ds, did, exclude_near_limit=True)
    new, _ = transitions_from_dataset(ds, did, IdentityObservation(),
                                      exclude_near_limit=True)
    check(len(legacy) == len(new), f"same number of transitions ({len(legacy)})")
    ok = all(close(a, t.obs) and close(b, t.action) and close(c, t.next_obs)
             for (a, b, c), t in zip(legacy, new))
    check(ok, "every (obs, action, next_obs) triple is identical to episode_stream")
    check(all(close(t.truth, t.next_obs) for t in new),
          "under an identity observation model, truth == next_obs")


def _legacy_vs_new(name: str, legacy_adaptor, ws: Workspace, cfg: MethodConfig,
                   stream, transitions, dt: float) -> None:
    """Run both paths over the same stream and compare every recorded quantity.

    Everything is required to be BIT-IDENTICAL, with one documented exception:
    the legacy classes store the reported belief in a float32 torch tensor, while
    the new ``Trace`` keeps float64 throughout. Where that is the only difference,
    the comparison is made at float32 -- and it must then be exactly zero, which
    is a stronger statement than "close in float64".
    """
    old = run_online_adaptation(legacy_adaptor, stream, door_id=0, verify_frozen=False)
    method = build_method(name, ws, dt, cfg)
    new = run(method.estimator, method.predictor, transitions, object_id=0,
              verify_frozen=False)

    check(len(old) == len(new), f"{name}: same number of steps")

    d_err = maxdiff(old.error, new.error)
    check(d_err == 0.0, f"{name}: prequential error bit-identical (diff {d_err:.2e})")

    # The belief. For the UKF the legacy log stores the decoded 16-D latent while
    # the new trace stores the 6-D chart coordinate, so decode before comparing.
    new_b = new.beliefs
    if new_b.shape[1] != old.latents.shape[1]:
        new_b = np.stack([method.predictor.representation.to_predictor(x)
                          for x in new.beliefs])
    d_bel = maxdiff(old.latents, new_b)
    if d_bel == 0.0:
        check(True, f"{name}: belief trajectory bit-identical")
    else:
        d32 = maxdiff(old.latents, new_b.astype(np.float32))
        check(d32 == 0.0,
              f"{name}: belief identical at legacy's float32 storage precision "
              f"(float64 diff {d_bel:.2e}, float32 diff {d32:.2e})")

    old_l = np.nan_to_num(old.loss)
    new_l = np.nan_to_num(new.loss)
    d_loss = maxdiff(old_l, new_l)
    if d_loss == 0.0:
        check(True, f"{name}: per-step loss bit-identical")
    else:
        # legacy accumulates the loss in float32 (torch mse_loss); the new path
        # uses float64. Anything at float32 epsilon is that, not a behaviour change.
        rel = float(np.max(np.abs(old_l - new_l) / np.maximum(np.abs(old_l), 1e-30)))
        check(rel <= 4 * float(np.finfo(np.float32).eps),
              f"{name}: per-step loss agrees to float32 precision "
              f"(rel diff {rel:.2e}, float32 eps {np.finfo(np.float32).eps:.2e})")


def test_estimator_equivalence() -> None:
    section("EQUIVALENCE  estimators vs their legacy implementations")
    if not (_have(DATA) and _have(CKPT)):
        print(f"  [skip] need {DATA} and {CKPT}")
        return

    ws = Workspace.load(CKPT, stage="mechanics_tests")
    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=True)
    did = int(ds.door_ids[0])
    stream = episode_stream(ds, did, exclude_near_limit=True)[:400]
    transitions, _ = transitions_from_dataset(ds, did, IdentityObservation(),
                                              exclude_near_limit=True)
    transitions = transitions[:400]
    dt = ds.dt_model
    cfg = MethodConfig(init="medoid")
    z0 = ws.init_latent("medoid")

    _legacy_vs_new(
        "no-adaptation",
        StaticLatentAdaptor(ws.model, init=z0),
        ws, cfg, stream, transitions, dt)

    _legacy_vs_new(
        "gradient",
        GradientLatentAdaptor(
            ws.model, init=z0, lr=cfg.lr, optimizer=cfg.optimizer,
            n_inner_steps=cfg.n_inner_steps, window=cfg.window,
            prior_weight=cfg.prior_weight, loss_space=cfg.loss_space,
            max_grad_norm=cfg.max_grad_norm, lr_decay=cfg.lr_decay),
        ws, cfg, stream, transitions, dt)

    for n_params in (5, 3):
        _legacy_vs_new(
            f"rls-{n_params}p",
            RLSAdaptor(dt=dt, n_substeps=cfg.n_substeps, n_params=n_params,
                       lam=cfg.lam, delta=cfg.delta, vel_thresh=cfg.vel_thresh),
            ws, cfg, stream, transitions, dt)


def test_ukf_equivalence() -> None:
    section("EQUIVALENCE  UKF vs UKFLatentAdaptor")
    if not (_have(ALL_FAMILIES) and _have(ALL_FAMILIES_DATA) and _have(BASIS)):
        print("  [skip] need the all-families checkpoint, its data and the basis")
        return
    from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
    from latent_mechanics.belief.basis import LatentBasis

    ws = Workspace.load(ALL_FAMILIES, stage="mechanics_tests:ukf")
    ds = DoorTransitionDataset(ALL_FAMILIES_DATA, "heldout_door",
                               exclude_near_limit=False)
    did = int(ds.door_ids[0])
    stream = episode_stream(ds, did, exclude_near_limit=False)[:250]
    transitions, _ = transitions_from_dataset(ds, did, IdentityObservation(),
                                              exclude_near_limit=False)
    transitions = transitions[:250]

    z0 = ws.init_latent("medoid")
    ukf_cfg = UKFConfig()
    basis = LatentBasis.load(BASIS)
    if basis.dim > ukf_cfg.dim:
        basis = basis.truncate(ukf_cfg.dim)

    legacy = UKFLatentAdaptor(ws.model, basis, ukf_cfg, init=z0,
                              prior_latents=ws.train_latents)
    cfg = MethodConfig(init="medoid", basis_path=BASIS, ukf=ukf_cfg)
    _legacy_vs_new("ukf", legacy, ws, cfg, stream, transitions, ds.dt_model)


def test_sensor_equivalence() -> None:
    section("EQUIVALENCE  JointSensor vs SensorPipeline")
    from latent_mechanics.mismatch.sensors import SensorPipeline

    states = np.cumsum(np.ones((50, 2)), axis=0).astype(np.float32) * 0.03
    for kwargs in ({"theta_sigma": 0.01},
                   {"quantize_bits": 12},
                   {"dropout_prob": 0.2},
                   {"latency_steps": 3},
                   {"theta_sigma": 0.005, "quantize_bits": 14, "dropout_prob": 0.1}):
        a = SensorPipeline(**kwargs).apply(states, 0.02, np.random.default_rng(7))
        b = JointSensor(**kwargs).observe(states, 0.02, np.random.default_rng(7))
        check(close(a, b), f"JointSensor({kwargs}) matches SensorPipeline exactly")

    # the fix: an explicit span changes the quantiser, and only the quantiser
    door = JointSensor(quantize_bits=10).observe(states, 0.02, np.random.default_rng(1))
    drawer = JointSensor(quantize_bits=10, joint_span=0.5).observe(
        states, 0.02, np.random.default_rng(1))
    check(not close(door, drawer),
          "an explicit joint_span quantises against the mechanism's own travel")


def test_clean_scoring_identity() -> None:
    section("EQUIVALENCE  clean-truth scoring")
    if not (_have(DATA) and _have(CKPT)):
        print(f"  [skip] need {DATA} and {CKPT}")
        return
    ws = Workspace.load(CKPT, stage="mechanics_tests:clean")
    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=True)
    did = int(ds.door_ids[0])

    sensor = JointSensor(theta_sigma=0.004)
    trs, _ = transitions_from_dataset(ds, did, sensor, exclude_near_limit=True)
    trs = trs[:200]
    check(any(not close(t.obs, t.truth) for t in trs[:-1]),
          "the sensor really did corrupt the stream")
    check(all(close(t.target, t.truth) for t in trs),
          "scoring targets the clean state, not the corrupted reading")

    m = build_method("no-adaptation", ws, ds.dt_model, MethodConfig(init="medoid"))
    tr = run(m.estimator, m.predictor, trs, object_id=did)
    # legacy identity: clean_error = raw_error + (observed_next - clean_next)
    manual = np.stack([
        m.predictor.predict(t.obs, t.action, Belief(mean=m.predictor.representation.initial()))
        - t.truth for t in trs])
    check(close(tr.error, manual, 1e-6),
          "trace error equals prediction - clean truth (the mismatch clean_errors identity)")


# --------------------------------------------------------------------------

def main() -> int:
    print("=" * 74)
    print("mechanics/ core self-check")
    print("=" * 74)

    test_types()
    test_representations()
    test_observation_models()
    test_predictors()
    test_analytical_predictor_matches_rls()
    test_metrics()
    test_loop_protocol()
    test_stream_equivalence()
    test_estimator_equivalence()
    test_ukf_equivalence()
    test_sensor_equivalence()
    test_clean_scoring_identity()

    print("\n" + "=" * 74)
    if _failures:
        print(f"{len(_failures)} of {_checks} checks FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"all {_checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
