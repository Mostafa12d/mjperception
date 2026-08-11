"""
Self-checks for the Stage-5 curriculum study.

Two properties carry the whole stage, and both are asserted here rather than
assumed:

  *the training budget is fixed* -- every level trains on the same number of
   instances, so movement along the diversity axis cannot be confused with
   movement along a data-quantity axis. This is the difference between a
   diversity result and a scaling-law truism.

  *the evaluation suite never changes* -- the same unseen instances, in the same
   order, are the held-out split of every level's dataset. If this drifted, the
   scaling curves would be comparing models on different tests.

Run:
    python3.10 -m latent_mechanics.curriculum.tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.curriculum.figures import (
    effective_dimension,
    geometry_correlation,
)
from latent_mechanics.curriculum.levels import (
    CURRICULUM,
    EVAL_FAMILIES,
    CurriculumConfig,
    split_budget,
)
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.data_gen import build_dataset_npz
from latent_mechanics.mechanisms.rollout import rollout_mechanism

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_budget_is_fixed() -> None:
    print("\nTraining budget is identical at every level")
    cc = CurriculumConfig()
    totals = set()
    for lv in CURRICULUM:
        comp = split_budget(lv, cc.train_instances)
        totals.add(sum(comp.values()))
        check(f"{lv.label()}: every family gets at least one instance",
              min(comp.values()) >= 1, str(comp))
        check(f"{lv.label()}: composition covers exactly its families",
              set(comp) == set(lv.families))
    check("all levels use the same instance budget", totals == {cc.train_instances},
          str(totals))


def test_curriculum_is_monotone() -> None:
    print("\nDiversity increases monotonically")
    counts = [lv.n_families for lv in CURRICULUM]
    check("family count is non-decreasing",
          all(b >= a for a, b in zip(counts, counts[1:])), str(counts))
    for a, b in zip(CURRICULUM, CURRICULUM[1:]):
        if a.n_families < b.n_families:
            check(f"{b.label()} keeps everything {a.label()} had",
                  set(a.families) <= set(b.families),
                  f"{set(a.families) - set(b.families)} dropped")
    check("levels 1 and 2 differ in parameter spread, not mechanism",
          CURRICULUM[0].families == ("door_narrow",)
          and CURRICULUM[1].families == ("door",))
    narrow = lib.FAMILIES["door_narrow"]
    wide = lib.FAMILIES["door"]
    span = lambda r: r[1] - r[0]
    check("narrow doors really do have a narrower friction range",
          span(narrow.friction_range) < span(wide.friction_range),
          f"{narrow.friction_range} vs {wide.friction_range}")
    check("narrow doors have no spring variation",
          span(narrow.stiffness_range) == 0 < span(wide.stiffness_range))


def test_eval_suite_is_fixed_and_unseen() -> None:
    print("\nEvaluation suite is fixed, complete and disjoint from training")
    cc = CurriculumConfig()
    check("suite covers every real mechanism family",
          set(EVAL_FAMILIES) == set(lib.FAMILY_ORDER) - {"door_narrow"},
          str(set(lib.FAMILY_ORDER) - {"door_narrow"} - set(EVAL_FAMILIES)))
    check("suite excludes the training-only narrow-door family",
          "door_narrow" not in EVAL_FAMILIES)
    check("eval seed is far from the training seed",
          abs(cc.eval_seed - cc.train_seed) > 100)

    # Same seed -> byte-identical suite, which is what "never changes" means.
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    a = lib.sample_params("door", np.random.default_rng(cc.eval_seed), 10_000)
    b = lib.sample_params("door", np.random.default_rng(cc.eval_seed), 10_000)
    check("suite sampling is deterministic", a == b)
    c = lib.sample_params("door", np.random.default_rng(cc.train_seed), 0)
    check("training draw differs from the eval draw",
          abs(a.frictionloss - c.frictionloss) > 1e-9)


def test_eval_instances_are_heldout_everywhere() -> None:
    print("\nEval instances are held out in every level's dataset")
    cfg = load_stage1_config("configs/latent_mechanics.yaml")
    train = [rollout_mechanism(lib.sample_params(f, np.random.default_rng(i), i),
                               cfg, 2, 2.0, 10, seed=1)
             for i, f in enumerate(["door", "drawer"])]
    suite = [rollout_mechanism(lib.sample_params(f, np.random.default_rng(500 + i),
                                                 10_000 + i),
                               cfg, 2, 2.0, 10, seed=999)
             for i, f in enumerate(["door", "laptop"])]
    train = [p for p in train if len(p) > 20]
    suite = [p for p in suite if len(p) > 20]

    with tempfile.TemporaryDirectory() as d:
        for label, fams in (("L2", ["door"]), ("L4", ["door", "drawer"])):
            npz = build_dataset_npz(train, fams, Path(d) / f"{label}.npz",
                                    cfg, 10, heldout_pops=suite)
            tr = DoorTransitionDataset(npz, "train", exclude_near_limit=False)
            he = DoorTransitionDataset(npz, "heldout_door", exclude_near_limit=False)
            n_suite_transitions = sum(len(p) for p in suite)
            check(f"{label}: the suite is entirely in the held-out split",
                  len(he) >= n_suite_transitions,
                  f"{len(he)} < {n_suite_transitions}")
            check(f"{label}: held-out ids sit above the embedding table",
                  int(he.door_ids.min()) >= tr.num_embedding_rows)
            check(f"{label}: an unseen DOOR stays held out even though doors "
                  f"are trained on",
                  int(he.door_ids.max()) - int(he.door_ids.min()) + 1 == len(suite),
                  "a suite instance leaked into training")

    # A single-episode instance must still contribute training data.
    one = [rollout_mechanism(lib.sample_params("door", np.random.default_rng(7), 7),
                             cfg, 1, 2.0, 10, seed=1)]
    with tempfile.TemporaryDirectory() as d:
        npz = build_dataset_npz(one, ["door"], Path(d) / "one.npz", cfg, 10)
        tr = DoorTransitionDataset(npz, "train", exclude_near_limit=False)
        check("a one-episode instance still yields training transitions", len(tr) > 0)


def test_geometry_measures() -> None:
    print("\nLatent geometry measures behave")
    rng = np.random.default_rng(0)
    mech = rng.normal(size=(40, 3))
    check("a latent that IS the mechanics scores near 1",
          geometry_correlation(np.hstack([mech, mech]), mech) > 0.9)
    check("a latent unrelated to mechanics scores near 0",
          abs(geometry_correlation(rng.normal(size=(40, 6)), mech)) < 0.4)

    one_axis = np.zeros((40, 6)); one_axis[:, 0] = rng.normal(size=40)
    check("collapsed latent has effective dimension ~1",
          abs(effective_dimension(one_axis) - 1.0) < 0.2,
          f"{effective_dimension(one_axis):.2f}")
    iso = rng.normal(size=(400, 6))
    check("isotropic latent has effective dimension ~6",
          effective_dimension(iso) > 5.0, f"{effective_dimension(iso):.2f}")


def test_frozen_components() -> None:
    print("\nFrozen components untouched")
    s = dyn.rls_init(2, lam=0.99)
    s = dyn.rls_step(s, np.array([1.0, 0.5]), 2.0)
    check("RLS baseline runs", bool(np.all(np.isfinite(s.theta))))
    check("dyn globals intact", (dyn.T_END, dyn.N_STEPS, dyn.DT) == (6.0, 3000, 0.002))

    from latent_mechanics.model import MechanicsDynamicsModel
    from latent_mechanics.online.adaptor import GradientLatentAdaptor
    m = MechanicsDynamicsModel(embed_dim=16, hidden_sizes=[256, 256])
    check("architecture defaults unchanged (embed 16, 2x256)",
          m.embed_dim == 16 and sum(p.numel() for p in m.parameters()) == 71426,
          str(sum(p.numel() for p in m.parameters())))
    ad = GradientLatentAdaptor(m, lr=0.01)
    ad.observe(np.zeros(2, np.float32), np.zeros(1, np.float32), np.zeros(2, np.float32))
    ad.assert_network_unchanged()
    check("Stage-2 adaptor still refuses to touch network weights", True)

    # Adding door_narrow must not have perturbed any pre-existing family.
    d = lib.FAMILIES["door"]
    check("the 'door' family spec is unchanged",
          d.friction_range == (0.5, 6.0) and d.damping_range == (0.02, 1.5)
          and d.stiffness_range == (0.0, 8.0), str(d))


# ---------------------------------------------------------------------------

# Drawn in a fresh interpreter so the parent's hash salt cannot leak in. Prints
# the per-family seeds and the first sampled parameter vector per family, which
# together pin down the whole population draw without simulating anything.
_DRAW_SNIPPET = """
import json, sys
sys.path.insert(0, %(root)r)
import numpy as np
from latent_mechanics.curriculum.levels import CurriculumConfig, EVAL_FAMILIES
from latent_mechanics.curriculum.study import family_seed
from latent_mechanics.mechanisms import library as lib

cc = CurriculumConfig()
out = {}
for fam in EVAL_FAMILIES:
    s = family_seed(cc.eval_seed, fam)
    rng = np.random.default_rng(s)
    p = lib.sample_params(fam, rng, mechanism_id=0)
    out[fam] = [s, p.density_scale, p.frictionloss, p.damping, p.stiffness,
                p.springref, sorted(p.extra.items())]
print(json.dumps(out))
"""


def _draw_in_subprocess(hashseed: str) -> dict:
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    r = subprocess.run([sys.executable, "-c", _DRAW_SNIPPET % {"root": root}],
                       capture_output=True, text=True, env=env, cwd=root)
    if r.returncode != 0:
        raise RuntimeError(f"subprocess failed (PYTHONHASHSEED={hashseed}):\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_population_draw_is_reproducible_across_processes() -> None:
    """A1: the population must follow from the seed, not from the hash salt.

    Two fresh interpreters with deliberately different ``PYTHONHASHSEED`` values
    must draw the identical population. Under the old
    ``base_seed + abs(hash(fam)) % 10_000`` this fails, because ``hash`` on a
    ``str`` is salted per process.
    """
    print("\nPopulation draw is reproducible from the seed alone (fresh processes)")
    try:
        a = _draw_in_subprocess("1")
        b = _draw_in_subprocess("12345")
    except RuntimeError as e:
        check("two fresh interpreters draw the same population", False, str(e))
        return

    check("both subprocesses drew every eval family",
          set(a) == set(b) == set(EVAL_FAMILIES), f"{sorted(a)} vs {sorted(b)}")
    check("per-family seeds agree across processes",
          all(a[f][0] == b[f][0] for f in a),
          str({f: (a[f][0], b[f][0]) for f in a if a[f][0] != b[f][0]}))
    check("first sampled instance agrees across processes", a == b,
          str({f: (a[f], b[f]) for f in a if a[f] != b[f]}))
    # And the seeds must actually differ between families, or the fix would be
    # trivially satisfied by a constant.
    seeds = [a[f][0] for f in a]
    check("per-family seeds are distinct", len(set(seeds)) == len(seeds), str(seeds))


def main() -> None:
    print("latent_mechanics.curriculum self-checks")
    test_budget_is_fixed()
    test_curriculum_is_monotone()
    test_eval_suite_is_fixed_and_unseen()
    test_eval_instances_are_heldout_everywhere()
    test_geometry_measures()
    test_frozen_components()
    test_population_draw_is_reproducible_across_processes()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
