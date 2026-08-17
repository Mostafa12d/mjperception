"""Self-checks for the Stage-3 robustness study.

The load-bearing one is ``test_simulator_equivalence``: drift between Stage 3's
own loop and ``dyn.simulate`` would be indistinguishable from a real mismatch
effect, so equality is asserted exactly. The rest guard the measurement protocol.

    python3.10 -m latent_mechanics.mismatch.tests
"""

from __future__ import annotations

import numpy as np
import torch

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics import door_sampler
from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.data_gen import episode_length
from latent_mechanics.excitation import sample_profile
from latent_mechanics.mismatch.config import default_sweeps
from latent_mechanics.mismatch.perturbations import (
    NonlinearCompliance,
    ParameterDrift,
    PositionDependentFriction,
    StribeckFriction,
    build_perturbation,
)
from latent_mechanics.mismatch.sensors import SensorPipeline
from latent_mechanics.mismatch.simulate import simulate_perturbed, verify_matches_baseline
from latent_mechanics.mismatch.streams import (
    build_door_stream,
    clean_errors,
    frozen_predict_errors,
)

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def _door(idx: int = 0):
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    train, held = door_sampler.sample_door_population(cfg.doors, cfg.sim.seed)
    return cfg, held[idx]


def test_simulator_equivalence() -> None:
    print("\nPerturbed simulator == dyn.simulate when nothing is perturbed")
    cfg, params = _door()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(3):
        model = door_sampler.build_model(params)
        prof = sample_profile(cfg.excitation, rng, 3000, 10, params.frictionloss)
        with episode_length(6.0):
            d = verify_matches_baseline(prof.as_fn(), model, 3000, tol=0.0)
        worst = max(worst, max(d.values()))
    check("exact agreement on every logged signal", worst == 0.0, f"max dev {worst:.3e}")


def test_perturbations_off_by_default() -> None:
    print("\nPerturbations are genuinely inert at level zero")
    for kind, zero in [("stribeck", {"excess": 0.0}),
                       ("position_friction", {"amplitude": 0.0}),
                       ("compliance", {"k_cubic": 0.0})]:
        p = build_perturbation(kind, **zero)
        vals = [p.extra_torque(t, th, td)
                for t in (0.0, 1.0) for th in (-0.5, 0.0, 1.5) for td in (-1.0, 0.0, 2.0)]
        check(f"{kind} contributes no torque when its parameter is 0",
              all(v == 0.0 for v in vals))

    cfg, params = _door()
    model = door_sampler.build_model(params)
    drift = ParameterDrift(friction_rate=0.0, damping_rate=0.0, stiffness_rate=0.0)
    drift.reset(model)
    before = (float(model.dof_frictionloss[0]), float(model.dof_damping[0]))
    drift.update_model(5.0, model)
    after = (float(model.dof_frictionloss[0]), float(model.dof_damping[0]))
    check("drift with zero rates leaves the model untouched", before == after)


def test_perturbations_have_effect() -> None:
    print("\nEach perturbation changes the plant, and stays stable")
    cfg, params = _door()
    rng = np.random.default_rng(5)
    prof = sample_profile(cfg.excitation, rng, 3000, 10, params.frictionloss)
    base = simulate_perturbed(prof.as_fn(), door_sampler.build_model(params), 3000)

    for name, pert in [
        ("stribeck", StribeckFriction(excess=4.0)),
        ("position_friction", PositionDependentFriction(amplitude=3.0)),
        ("compliance", NonlinearCompliance(k_cubic=2.0)),
        ("drift", ParameterDrift(friction_rate=0.4)),
    ]:
        lg = simulate_perturbed(prof.as_fn(), door_sampler.build_model(params), 3000, [pert])
        moved = float(np.abs(lg.theta - base.theta).max())
        check(f"{name} perturbs the trajectory", moved > 1e-4, f"max dtheta {moved:.2e}")
        check(f"{name} stays finite and bounded",
              bool(np.isfinite(lg.theta).all()) and np.abs(lg.theta).max() < 10.0)

    # The recorded action must never contain the unmodelled torque.
    lg = simulate_perturbed(prof.as_fn(), door_sampler.build_model(params), 3000,
                            [StribeckFriction(excess=4.0)])
    check("action excludes the unmodelled torque",
          np.allclose(lg.tau_ft, base.tau_ft, atol=1e-9),
          "tau_ft changed when a perturbation was added")


def test_drift_tracks_time() -> None:
    print("\nParameter drift is time-varying and clamped")
    cfg, params = _door()
    model = door_sampler.build_model(params)
    d = ParameterDrift(friction_rate=0.2, mode="linear")
    d.reset(model)
    f0 = d.current_params(0.0)["frictionloss"]
    f6 = d.current_params(6.0)["frictionloss"]
    check("friction grows over the episode", f6 > f0 * 2.0, f"{f0:.3f} -> {f6:.3f}")

    neg = ParameterDrift(friction_rate=-10.0, mode="linear")
    neg.reset(model)
    neg.update_model(6.0, model)
    check("negative drift is clamped at zero, never energy-injecting",
          float(model.dof_frictionloss[0]) >= 0.0)


def test_sensor_pipeline() -> None:
    print("\nSensor pipeline")
    rng = np.random.default_rng(0)
    T, dt = 4000, 0.02
    states = np.stack([np.linspace(0, 1.5, T), np.full(T, 0.3)], axis=1).astype(np.float32)

    check("identity when everything is off",
          np.array_equal(SensorPipeline().apply(states, dt, rng), states))

    sp = SensorPipeline(theta_sigma=1e-3, theta_dot_sigma=0.0)
    out = sp.apply(states, dt, np.random.default_rng(1))
    sigma = float(np.std(out[:, 0] - states[:, 0]))
    check("Gaussian noise has the requested sigma", abs(sigma - 1e-3) < 1.5e-4, f"{sigma:.2e}")
    check("velocity untouched when its sigma is 0",
          np.allclose(out[:, 1], states[:, 1], atol=1e-6))

    derived = SensorPipeline(theta_sigma=1e-3).velocity_sigma(dt)
    check("derived velocity noise amplifies by sqrt(2)/dt",
          abs(derived - 1e-3 * np.sqrt(2) / dt) < 1e-9, f"{derived:.4f}")

    sp = SensorPipeline(quantize_bits=10)
    out = sp.apply(states, dt, np.random.default_rng(2))
    step = (2.09 - (-0.17)) / 2**10
    resid = np.abs(out[:, 0] / step - np.round(out[:, 0] / step))
    check("quantised angles land on the encoder grid", float(resid.max()) < 1e-4)

    sp = SensorPipeline(dropout_prob=0.3)
    out = sp.apply(states, dt, np.random.default_rng(3))
    repeats = np.mean(np.all(np.diff(out, axis=0) == 0, axis=1))
    check("dropout holds the previous reading", 0.15 < repeats < 0.45, f"{repeats:.2f}")

    sp = SensorPipeline(latency_steps=3)
    out = sp.apply(states, dt, rng)
    check("latency shifts the stream by exactly k samples",
          np.allclose(out[3:], states[:-3], atol=1e-6))
    check("latency holds the first sample rather than inventing data",
          np.allclose(out[:3], states[0], atol=1e-6))


def test_sensor_applied_to_sequence() -> None:
    """The shared state between consecutive transitions must be ONE reading."""
    print("\nSensor noise is applied to the state sequence, not per transition")
    cfg, params = _door()
    stream = build_door_stream(
        params, cfg, n_episodes=1, episode_seconds=2.0, frame_skip=10,
        sensors=SensorPipeline(theta_sigma=1e-3, theta_dot_sigma=1e-3), seed=0,
    )
    tr = stream.transitions
    # next_state of transition i must be bit-identical to state of transition i+1.
    mismatches = sum(
        0 if np.array_equal(tr[i][2], tr[i + 1][0]) else 1 for i in range(len(tr) - 1)
    )
    check("a shared state is measured exactly once", mismatches == 0,
          f"{mismatches} of {len(tr) - 1} boundaries disagree")


def test_scoring_recovers_truth() -> None:
    print("\nScoring")
    cfg, params = _door()
    clean = build_door_stream(params, cfg, 1, 2.0, 10, seed=0)
    check("with an identity sensor, observed target == clean target",
          np.allclose(clean.observed_next, clean.clean_next, atol=0))

    noisy = build_door_stream(
        params, cfg, 1, 2.0, 10, sensors=SensorPipeline(theta_sigma=2e-3), seed=0
    )
    check("with noise, observed and clean targets differ",
          not np.allclose(noisy.observed_next, noisy.clean_next, atol=1e-9))

    # clean_errors must reconstruct the error against ground truth exactly.
    class FakeLog:
        pass

    n = len(noisy)
    fake = FakeLog()
    pred = np.random.default_rng(0).normal(size=(n, 2))
    fake.error = pred - noisy.observed_next[:n]
    recovered = clean_errors(fake, noisy)
    check("clean_errors reconstructs prediction - truth exactly",
          np.allclose(recovered, pred - noisy.clean_next[:n], atol=1e-12))

    scale = clean.motion_scale()
    check("motion scale is positive and finite",
          bool(np.all(np.isfinite(scale)) and np.all(scale > 0)), str(scale))


def test_frozen_evaluation_does_not_learn() -> None:
    print("\nHold-out evaluation uses a frozen belief")
    from latent_mechanics.model import load_checkpoint
    from latent_mechanics.online.adaptor import GradientLatentAdaptor

    try:
        model, table, _, _ = load_checkpoint("runs/latent_mechanics/base/best.pt")
    except FileNotFoundError:
        print("  [skip] no checkpoint on disk")
        return
    model.freeze()
    cfg, params = _door()
    stream = build_door_stream(params, cfg, 1, 2.0, 10, seed=0)

    ad = GradientLatentAdaptor(model, init=table.weight.detach().numpy()[0], lr=0.03)
    before = ad.latent.copy()
    err = frozen_predict_errors(ad, stream)
    check("belief is unchanged by evaluation", np.array_equal(before, ad.latent))
    check("one error per transition", err.shape == (len(stream), 2))
    check("no updates were counted", ad.n_updates == 0)


def test_sweeps_well_formed() -> None:
    print("\nSweep definitions")
    sweeps = default_sweeps()
    check("all four experiments are covered",
          {s.experiment for s in sweeps} == {1, 2, 3, 4})
    check("sweep names are unique", len({s.name for s in sweeps}) == len(sweeps))
    for s in sweeps:
        check(f"{s.name}: first level is the unperturbed control",
              s.levels[0] in (0, 0.0, None))
        check(f"{s.name}: has an axis label", bool(s.axis_label()))
        if s.kind == "plant":
            check(f"{s.name}: names a real perturbation type",
                  build_perturbation(s.target, **{s.param: s.levels[-1]}, **s.fixed) is not None)


def test_stages_1_and_2_untouched() -> None:
    print("\nStages 1 and 2 still work")
    state = dyn.rls_init(2, lam=0.99)
    state = dyn.rls_step(state, np.array([1.0, 0.5]), 2.0)
    check("RLS baseline runs", bool(np.all(np.isfinite(state.theta))))
    check("dyn globals intact", (dyn.T_END, dyn.N_STEPS, dyn.DT) == (6.0, 3000, 0.002))

    from latent_mechanics.model import MechanicsDynamicsModel
    from latent_mechanics.online.adaptor import GradientLatentAdaptor

    m = MechanicsDynamicsModel(embed_dim=4, hidden_sizes=[16])
    ad = GradientLatentAdaptor(m, lr=0.01)
    ad.observe(np.zeros(2, np.float32), np.zeros(1, np.float32), np.zeros(2, np.float32))
    ad.assert_network_unchanged()
    check("Stage-2 adaptor still refuses to touch network weights", True)


def main() -> None:
    print("latent_mechanics.mismatch self-checks")
    test_simulator_equivalence()
    test_perturbations_off_by_default()
    test_perturbations_have_effect()
    test_drift_tracks_time()
    test_sensor_pipeline()
    test_sensor_applied_to_sequence()
    test_scoring_recovers_truth()
    test_frozen_evaluation_does_not_learn()
    test_sweeps_well_formed()
    test_stages_1_and_2_untouched()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
