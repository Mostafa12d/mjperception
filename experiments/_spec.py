"""One experiment = one ``ExperimentSpec`` + one ``run.py``.

The rule this enforces: **you should be able to read a spec and know exactly what
hypothesis is being tested**, without opening another file. Everything the brief
asks an experiment to declare is a field here -- plant, observation model,
predictor, mechanics representation, estimator, initialisation, disturbances,
metrics -- and nothing is inherited from an ambient config.

Three constraints are checked rather than trusted, each because the audit found a
way the old code lost them:

  * ``question`` must be a real sentence. If you cannot state the hypothesis, the
    experiment is not ready to run.
  * ``no-adaptation`` is included by default and warns loudly if removed. It is
    the control that produced Stage 5's negative result.
  * the predictor checkpoint is recorded, and ``expected_sha256`` can pin it.
    Open item B1 -- nine different predictors across the stages -- became invisible
    precisely because no experiment recorded which one it used.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from latent_mechanics.dataset import DoorTransitionDataset
from mechanics import (
    IdentityObservation,
    MethodConfig,
    ObservationModel,
    Workspace,
    build_method,
    run,
    transitions_from_dataset,
)
from mechanics.metrics import score, write_csv
from mechanics.types import Trace

DEFAULT_METHODS = ("no-adaptation", "gradient", "ukf", "rls-5p")


@dataclass
class ExperimentSpec:
    """A complete, self-contained description of one experiment."""

    # -- what is being asked -------------------------------------------------
    question: str

    # -- plant / data --------------------------------------------------------
    data: str                                   # transitions .npz
    split: str = "heldout_door"                 # unseen objects
    object_ids: tuple[int, ...] = ()            # empty -> every object in the split
    max_objects: int | None = None
    max_episodes: int | None = None
    exclude_near_limit: bool = True

    # -- observation model ---------------------------------------------------
    observation: ObservationModel = field(default_factory=IdentityObservation)

    # -- predictor -----------------------------------------------------------
    checkpoint: str = "runs/latent_mechanics/base/best.pt"
    expected_sha256: str | None = None
    device: str = "cpu"

    # -- mechanics representation + estimators -------------------------------
    # (the representation is implied by the method: gradient -> full latent,
    #  ukf -> reduced chart, rls -> physical parameters. build_method pairs them.)
    methods: tuple[str, ...] = DEFAULT_METHODS
    method_config: MethodConfig = field(default_factory=MethodConfig)

    # -- protocol ------------------------------------------------------------
    seeds: tuple[int, ...] = (0,)
    min_steps: int = 100                        # objects with fewer are skipped

    # -- output --------------------------------------------------------------
    out_dir: str = "runs/experiments/unnamed"
    metrics: tuple[str, ...] = (
        "angle_nrmse_final", "angle_rmse_final", "us_per_update", "belief_travel")

    def validate(self) -> None:
        if len(self.question.split()) < 5 or "?" not in self.question:
            raise ValueError(
                "ExperimentSpec.question must be a real question in a sentence. "
                f"Got: {self.question!r}")
        if not Path(self.data).exists():
            raise FileNotFoundError(f"data not found: {self.data}")
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        if "no-adaptation" not in self.methods:
            print("  WARNING: this spec has no 'no-adaptation' control. Without it, "
                  "'the error went down' may only mean the stream got easier.")

    def describe(self) -> dict:
        d = asdict(self)
        d["observation"] = self.observation.describe()
        d["method_config"] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in asdict(self.method_config).items()}
        return d


def run_experiment(spec: ExperimentSpec, verbose: bool = True
                   ) -> tuple[list[dict], dict[tuple[int, str], Trace]]:
    """Run every method on every object. Returns per-(object, method) rows."""
    spec.validate()
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    ds = DoorTransitionDataset(spec.data, spec.split,
                               exclude_near_limit=spec.exclude_near_limit)
    ids = list(spec.object_ids) or [int(d) for d in ds.door_ids]
    if spec.max_objects is not None:
        ids = ids[: spec.max_objects]

    if verbose:
        print("=" * 74)
        print(spec.question)
        print("=" * 74)
        print(f"  predictor   : {spec.checkpoint}")
        print(f"  data        : {spec.data} [{spec.split}]  dt={ds.dt_model:.3f}s")
        print(f"  observation : {spec.observation.describe()}")
        print(f"  methods     : {', '.join(spec.methods)}")
        print(f"  objects     : {len(ids)}   seeds: {list(spec.seeds)}\n")

    rows: list[dict] = []
    traces: dict[tuple[int, str], Trace] = {}

    for seed in spec.seeds:
        ws = Workspace.load(spec.checkpoint, device=spec.device,
                            stage=f"experiment:{Path(spec.out_dir).name}",
                            expected_sha256=spec.expected_sha256, seed=seed)
        for oid in ids:
            transitions, bounds = transitions_from_dataset(
                ds, oid, spec.observation,
                max_episodes=spec.max_episodes,
                exclude_near_limit=spec.exclude_near_limit, seed=seed)
            if len(transitions) < spec.min_steps:
                continue

            for name in spec.methods:
                method = build_method(name, ws, ds.dt_model, spec.method_config)
                trace = run(method.estimator, method.predictor, transitions,
                            object_id=oid, boundaries=bounds,
                            init_name=spec.method_config.init,
                            verify_frozen=False)
                traces[(oid, method.name)] = trace
                rows.append({
                    "seed": seed, "object_id": oid, "method": method.name,
                    "predictor": method.predictor.name,
                    "representation": method.predictor.representation.name,
                    **score(trace, transitions),
                })
            if verbose:
                best = min((r for r in rows if r["object_id"] == oid),
                           key=lambda r: r["angle_nrmse_final"])
                print(f"  object {oid:4d}  {len(transitions):5d} steps  "
                      f"best: {best['method']:14s} "
                      f"nRMSE {best['angle_nrmse_final']:.3f}")

    if verbose:
        _summary(rows, spec)
    write_csv(out / "results.csv", rows)
    (out / "spec.json").write_text(json.dumps(spec.describe(), indent=2, default=str))
    if verbose:
        print(f"\n  results -> {out / 'results.csv'}")
        print(f"  spec    -> {out / 'spec.json'}")
        print(f"  done in {time.perf_counter() - t0:.1f}s")
    return rows, traces


def _summary(rows: list[dict], spec: ExperimentSpec) -> None:
    """Median over objects, per method. Median because the distribution across
    objects is heavy-tailed and a mean is dominated by the worst instance."""
    if not rows:
        print("\n  no objects produced enough transitions")
        return
    names = list(dict.fromkeys(r["method"] for r in rows))
    cols = [m for m in spec.metrics if m in rows[0]]
    print(f"\n  {'method':16s}" + "".join(f"{c:>22}" for c in cols))
    for n in names:
        sub = [r for r in rows if r["method"] == n]
        cells = "".join(f"{np.median([r[c] for r in sub]):>22.4g}" for c in cols)
        print(f"  {n:16s}{cells}")

    ctrl = [r["angle_nrmse_final"] for r in rows if r["method"] == "no-adaptation"]
    if ctrl:
        base = float(np.median(ctrl))
        print(f"\n  gain over the no-adaptation control (>1 = adaptation helped):")
        for n in names:
            if n == "no-adaptation":
                continue
            v = float(np.median([r["angle_nrmse_final"] for r in rows if r["method"] == n]))
            print(f"    {n:16s} {base / v if v > 0 else float('nan'):.2f}x")


__all__ = ["ExperimentSpec", "run_experiment", "DEFAULT_METHODS"]
