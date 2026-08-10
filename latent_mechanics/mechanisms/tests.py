"""
Self-checks for the Stage-4 mechanism suite.

The claim this stage rests on is "every mechanism exposes the same interaction
interface". That is only meaningful if it is checkable, so the tests assert it
directly: one state/action shape for all six families, one code path through the
simulator with no per-family branching, and no mechanism label or physical
parameter reachable from anything the model consumes.

Run:
    python3.10 -m latent_mechanics.mechanisms.tests
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np

import run_door_dynamics_validation as dyn
from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms import rollout as rl
from latent_mechanics.mechanisms.data_gen import build_dataset_npz
from latent_mechanics.mechanisms.rollout import rollout_mechanism, simulate_mechanism

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def _one(fam: str, idx: int = 0):
    rng = np.random.default_rng(idx)
    return lib.sample_params(fam, rng, idx)


# ---------------------------------------------------------------------------

def test_all_families_build() -> None:
    print("\nEvery family builds and exposes one joint")
    for fam in lib.FAMILY_ORDER:
        p = _one(fam)
        m = lib.build_model(p)
        qadr, dof, jid = lib.joint_info(m)
        gt = lib.ground_truth(m, p)
        check(f"{fam}: model builds with a named 'hinge' joint", jid >= 0)
        check(f"{fam}: inertia/mass positive and finite",
              np.isfinite(gt["inertia"]) and gt["inertia"] > 0, str(gt["inertia"]))
        expect_prismatic = fam == "drawer"
        check(f"{fam}: joint type recorded correctly",
              bool(gt["is_prismatic"]) == expect_prismatic)


def test_uniform_interface() -> None:
    print("\nThe interaction interface is identical across families")
    shapes, dtypes = set(), set()
    for fam in lib.FAMILY_ORDER:
        p = _one(fam)
        cfg = load_stage1_config("configs/latent_mechanics.yaml")
        ep = rollout_mechanism(p, cfg, 1, 2.0, 10, seed=0)
        if len(ep) == 0:
            check(f"{fam}: produced transitions", False, "empty rollout")
            continue
        shapes.add((ep.state.shape[1], ep.action.shape[1], ep.next_state.shape[1]))
        dtypes.add((ep.state.dtype, ep.action.dtype))
    check("all families share one (state, action, next_state) shape",
          shapes == {(2, 1, 2)}, str(shapes))
    check("all families share one dtype", len(dtypes) == 1, str(dtypes))

    # No per-family branching inside the simulator loop.
    src = inspect.getsource(simulate_mechanism)
    leaked = [f for f in lib.FAMILY_ORDER if f'"{f}"' in src or f"'{f}'" in src]
    check("simulator contains no family-specific branch", not leaked, str(leaked))


def test_bifold_is_underactuated() -> None:
    print("\nThe bi-fold cabinet is genuinely partially observed")
    p = _one("bifold")
    m = lib.build_model(p)
    check("bifold has two degrees of freedom", m.nv == 2, f"nv={m.nv}")
    log = simulate_mechanism(lambda t: 8.0, m, 1500)
    check("only one joint is logged", log.q.ndim == 1 and log.qdot.ndim == 1)

    import mujoco
    m2 = lib.build_model(p)
    d = mujoco.MjData(m2)
    leaf_dof = m2.jnt_dofadr[m2.joint("leaf_hinge").id]
    obs_dof = m2.jnt_dofadr[m2.joint("hinge").id]
    leaf = []
    for _ in range(1500):
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[obs_dof] = 8.0
        mujoco.mj_step(m2, d)
        leaf.append(float(d.qpos[m2.jnt_qposadr[m2.joint("leaf_hinge").id]]))
    # The leaf must swing appreciably: if it stayed put, the "two-link" family
    # would just be a heavier door and the partial-observability claim would be
    # false. Measured travel on this instance is ~2 rad.
    check("the unobserved leaf has its own dynamics",
          float(np.ptp(leaf)) > 0.1, f"leaf travel {np.ptp(leaf):.2e}")

    for fam in lib.FAMILY_ORDER:
        mm = lib.build_model(_one(fam))
        expect = 2 if fam == "bifold" else 1
        check(f"{fam}: nv == {expect}", mm.nv == expect, f"nv={mm.nv}")


def test_excitation_scaling() -> None:
    print("\nExcitation is scaled to each family's force unit")
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    for fam in lib.FAMILY_ORDER:
        p = _one(fam)
        rng = np.random.default_rng(0)
        prof = lib.scaled_profile(cfg.excitation, rng, 3000, 10, p)
        u = lib.FAMILIES[fam].force_unit
        peak = float(np.abs(prof.values).max())
        check(f"{fam}: torques respect the scaled clip",
              peak <= cfg.excitation.tau_clip * u + 1e-6,
              f"peak {peak:.2f} > clip {cfg.excitation.tau_clip * u:.2f}")
        check(f"{fam}: excitation is non-trivial", peak > 1e-6)


def test_labels_never_reach_the_model() -> None:
    print("\nMechanism labels are analysis-only")
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    pops = [rollout_mechanism(_one(f, i), cfg, 2, 2.0, 10, seed=0)
            for i, f in enumerate(lib.FAMILY_ORDER)]
    pops = [p for p in pops if len(p) > 20]
    with tempfile.TemporaryDirectory() as d:
        npz = build_dataset_npz(pops, ["door"], Path(d) / "t.npz", cfg, 10,
                                val_episodes=1)
        ds = DoorTransitionDataset(npz, "train", exclude_near_limit=False)
        item = ds[0]
        check("a sample contains only door_id/state/action/next_state",
              set(item) == {"door_id", "state", "action", "next_state"}, str(set(item)))
        check("state is 2-D", tuple(item["state"].shape) == (2,))
        check("action is 1-D", tuple(item["action"].shape) == (1,))
        check("door_id is an opaque integer index",
              item["door_id"].dtype.is_floating_point is False)

        held = DoorTransitionDataset(npz, "heldout_door", exclude_near_limit=False)
        check("held-out ids sit above the embedding table",
              int(held.door_ids.min()) >= ds.num_embedding_rows,
              f"{held.door_ids.min()} vs {ds.num_embedding_rows}")
        with np.load(npz, allow_pickle=False) as z:
            fams = np.array([str(x) for x in z["mechanism_family"]])
        check("family labels are stored separately for analysis",
              len(fams) == len(pops))


def test_dataset_roundtrip() -> None:
    print("\nDataset packing is consistent")
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    pops = [rollout_mechanism(_one(f, i), cfg, 2, 2.0, 10, seed=0)
            for i, f in enumerate(["door", "drawer", "laptop"])]
    with tempfile.TemporaryDirectory() as d:
        npz = build_dataset_npz(pops, ["door"], Path(d) / "t.npz", cfg, 10)
        ds = DoorTransitionDataset(npz, "all", exclude_near_limit=False)
        total = sum(len(p) for p in pops)
        check("no transitions lost in packing", len(ds) == total,
              f"{len(ds)} vs {total}")
        ep = ds.episode(0)
        chained = np.allclose(ep.next_state[:-1], ep.state[1:], atol=1e-6)
        check("transitions chain within an episode", chained)
        tr = DoorTransitionDataset(npz, "train", exclude_near_limit=False)
        va = DoorTransitionDataset(npz, "val", exclude_near_limit=False)
        check("train and val episodes are disjoint",
              not (set(tr.episode_ids()) & set(va.episode_ids())))


def test_soft_close_behaviour() -> None:
    print("\nSoft-close damper engages only near closed")
    from latent_mechanics.mechanisms.library import SoftCloseDamper
    sc = SoftCloseDamper(gain=10.0, width=0.2)
    near = abs(sc.extra_torque(0.0, 0.0, 1.0))
    far = abs(sc.extra_torque(0.0, 1.5, 1.0))
    check("damps strongly at the closed position", near > 5.0, f"{near:.2f}")
    check("negligible far from closed", far < 1e-6, f"{far:.2e}")
    check("no torque when stationary", sc.extra_torque(0.0, 0.0, 0.0) == 0.0)


def test_earlier_stages_intact() -> None:
    print("\nEarlier stages still work")
    s = dyn.rls_init(2, lam=0.99)
    s = dyn.rls_step(s, np.array([1.0, 0.5]), 2.0)
    check("RLS baseline runs", bool(np.all(np.isfinite(s.theta))))
    check("dyn globals intact", (dyn.T_END, dyn.N_STEPS, dyn.DT) == (6.0, 3000, 0.002))

    from latent_mechanics.model import MechanicsDynamicsModel
    from latent_mechanics.online.adaptor import GradientLatentAdaptor
    m = MechanicsDynamicsModel(embed_dim=4, hidden_sizes=[16])
    ad = GradientLatentAdaptor(m, lr=0.01)
    ad.observe(np.zeros(2, np.float32), np.zeros(1, np.float32), np.zeros(2, np.float32))
    ad.assert_network_unchanged()
    check("Stage-2 adaptor still refuses to touch network weights", True)


def test_analysis_metrics() -> None:
    print("\nAnalysis metrics behave on synthetic input")
    from latent_mechanics.mechanisms.analysis import (
        family_separability, mechanics_readout,
    )
    rng = np.random.default_rng(0)
    # Two well-separated blobs -> separability near 1.
    z = np.concatenate([rng.normal(0, 0.2, (20, 6)),
                        rng.normal(8, 0.2, (20, 6))])
    fam = np.array(["a"] * 20 + ["b"] * 20)
    sep = family_separability(z, fam)
    check("separable families give high 1-NN accuracy", sep["accuracy"] > 0.95,
          f"{sep['accuracy']:.2f}")

    # One blob, random labels -> separability near chance.
    z2 = rng.normal(0, 1.0, (40, 6))
    sep2 = family_separability(z2, fam)
    check("unstructured latents give near-chance accuracy", sep2["accuracy"] < 0.8,
          f"{sep2['accuracy']:.2f}")

    y = z2[:, 0] * 3.0 + 1.0
    check("readout recovers a linear function of the latent",
          mechanics_readout(z2, y) > 0.9, f"{mechanics_readout(z2, y):.2f}")
    check("readout rejects noise",
          mechanics_readout(z2, rng.normal(size=40)) < 0.5)


def test_near_limit_uses_each_familys_own_range() -> None:
    """A2: the near-limit flag must reflect the mechanism's real joint range.

    ``data_gen.JOINT_RANGE`` is the door's ``[-0.17, 2.09]``. A drawer travels
    ``[0, 0.5] m`` and a laptop hinge ``[0, 2.2] rad``, so scoring either against
    the door range flags nothing however hard the mechanism is jammed against its
    stop. This asserts the flag now matches the family's own range, and that the
    door default is unchanged.
    """
    import mujoco

    from latent_mechanics.config import ExperimentConfig
    from latent_mechanics.data_gen import JOINT_RANGE, transitions_from_log
    from latent_mechanics.mechanisms.rollout import (
        limit_margin_for,
        near_limit_mask,
        simulate_mechanism,
    )

    print("\nNear-limit flag follows each family's own joint range")
    cfg = ExperimentConfig()
    n_steps = 3000
    frame_skip = cfg.sim.frame_skip

    for fam, expect_range in (("drawer", (0.0, 0.5)),
                              ("laptop", (0.0, 2.2)),
                              ("door", JOINT_RANGE)):
        rng = np.random.default_rng(1000)
        params = lib.sample_params(fam, rng, 0)
        model = lib.build_model(params)
        _, _, jid = lib.joint_info(model)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        check(f"{fam}: joint range is {expect_range}",
              abs(lo - expect_range[0]) < 1e-9 and abs(hi - expect_range[1]) < 1e-9,
              f"got ({lo}, {hi})")

        profile = lib.scaled_profile(cfg.excitation, rng, n_steps, frame_skip, params)
        log = simulate_mechanism(profile.as_fn(), model, n_steps,
                                 lib.perturbations_for(params))
        margin = limit_margin_for(lo, hi)
        tr = transitions_from_log(log.as_stage1_dict(), frame_skip,
                                  joint_range=(lo, hi), limit_margin=margin)
        want = near_limit_mask(tr["state"], tr["next_state"], lo, hi)
        check(f"{fam}: flag matches the family's own range exactly",
              bool(np.array_equal(tr["near_limit"], want)),
              f"{int((tr['near_limit'] != want).sum())} disagreeing transitions")

        # And show the stale door range would have been wrong where it matters.
        stale = transitions_from_log(log.as_stage1_dict(), frame_skip)["near_limit"]
        if fam == "door":
            check("door: door-range default still reproduces the old behaviour",
                  bool(stale.any()) or not want.any(),
                  f"stale={100*stale.mean():.1f}% want={100*want.mean():.1f}%")
        else:
            check(f"{fam}: stale door range really does disagree",
                  not np.array_equal(stale, want),
                  f"stale={100*stale.mean():.1f}% vs true={100*want.mean():.1f}%")
            print(f"        {fam}: door-range flag {100*stale.mean():5.1f}% of "
                  f"transitions vs {100*want.mean():5.1f}% with the true range")


def main() -> None:
    print("latent_mechanics.mechanisms self-checks")
    test_all_families_build()
    test_uniform_interface()
    test_bifold_is_underactuated()
    test_excitation_scaling()
    test_labels_never_reach_the_model()
    test_dataset_roundtrip()
    test_soft_close_behaviour()
    test_earlier_stages_intact()
    test_analysis_metrics()
    test_near_limit_uses_each_familys_own_range()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
