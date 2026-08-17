# REFACTOR_PROPOSAL.md

*Companion to [CURRENT_SYSTEM.md](CURRENT_SYSTEM.md). Read that first — every
claim here rests on a finding there.*

The goal is **not** maximum modularity. It is this one sentence from the brief:

> I should be able to replace the predictor, observation model, mechanics
> representation, or online estimator without rewriting the rest of the experiment.

Everything below is justified by asking whether it makes a stated research
question cheaper to test. Where the answer was no, the abstraction is not here.

---

## 1. Is the proposed loop correct for this project?

The brief proposes:

```
Environment → Observation → Dynamics Predictor → Predicted Observation
Predicted Observation − Actual Observation → Innovation → Estimator → Belief
```

**Mostly yes. Two corrections, both discovered in the audit.**

### Correction 1: observation ≠ predictor input, and pretending otherwise will bite

Today `state == observation == [θ, θ̇]`, so the distinction is invisible. But the
project's own next step is vision-based estimation, and the mechanism suite
already contains **bifold**, where the observed joint is a strict subset of the
state. The moment observations become partial or noisy, the predictor must
consume a *state estimate*, not the raw observation.

So the loop needs an explicit `ObservationModel` sitting between plant and
observation — a component that is the identity today and therefore costs nothing,
but gives vision and partial observability somewhere to go that is not a rewrite.

```
Plant ──state──► ObservationModel ──observation──► ...
```

### Correction 2: the innovation is not always a difference of observations

The UKF deliberately works in **normalised-delta space**, because that is the
only space where one $R$ is meaningful across families
([belief/adaptor.py:3-6](latent_mechanics/belief/adaptor.py#L3)). RLS works in
**torque space**. Both choices are correct for their estimator. Forcing every
estimator to innovate in raw observation units would be an *algorithmic* change
smuggled in under a refactor — exactly what the brief forbids.

The fix is not to unify the space, but to **make the space explicit**. The
innovation becomes a first-class recorded object that carries a label saying what
space it lives in. That turns today's silent incomparability (§B.6 of the audit)
into a declared property.

### The corrected loop

```
        Plant  (MuJoCo, one mechanism instance)
          │  state trajectory  [θ, θ̇]        ← ground truth, never seen by a model
          ▼
   ObservationModel   (identity | SensorPipeline | vision | F/T)
          │  observation
          ▼
   ┌──────────────────────────── Transition (obs, action, next_obs) ───────┐
   │                                                                        │
   │   Predictor.predict(obs, action, belief) ──► predicted next observation│
   │            │                                          │                │
   │            │  (reported, prequential, raw SI)          │                │
   │            ▼                                          ▼                │
   │        ───────────── reported error ─────────────────                  │
   │                                                                        │
   │   Estimator.update(belief, transition)                                 │
   │        └─ forms its own innovation, in its own declared space          │
   │           (the Predictor is the measurement function for latent methods)│
   │                       │                                                │
   │                       ▼                                                │
   │              Belief  (MechanicsRepresentation-typed)                   │
   └────────────────────────────────────────────────────────────────────────┘
```

Note that `Predictor` appears twice on purpose: once as the thing that produces
the *reported* prediction, and once *inside* the estimator as the measurement
function $h$. That is not redundancy — it is the honest description of what the
code already does, and it is why the two residuals differ.

---

## 2. Proposed architecture

A new package `mechanics/` holding the five interfaces and one driver. Roughly
**700 lines total**, no framework, no registry magic, no plugin system.

```
mechanics/
    types.py          Transition, Belief, StepRecord, Trace          (~120 lines)
    plant.py          Plant protocol + MujocoPlant                    (~120)
    observation.py    ObservationModel protocol + Identity, Sensor    (~90)
    predictor.py      Predictor protocol + LatentNetwork, Analytical  (~140)
    representation.py MechanicsRepresentation + Full/Reduced/Physical (~110)
    estimator.py      Estimator protocol                              (~60)
    loop.py           run() — the driver                              (~90)
    metrics.py        one scoring implementation                      (~80)

    estimators/       thin adapters over the EXISTING algorithms
        static.py  gradient.py  ukf.py  rls.py

experiments/
    _spec.py                 ExperimentSpec — one dataclass, one runner
    sanity_one_door/         the canonical end-to-end experiment (§6)
    predictor_accuracy/
    estimator_convergence/
    ...
```

### 2.1 `types.py` — the vocabulary

```python
@dataclass(frozen=True)
class Transition:
    obs:      np.ndarray   # (n_obs,)  what was observed at t
    action:   np.ndarray   # (n_act,)  ZOH command over [t, t+dt)
    next_obs: np.ndarray   # (n_obs,)  what was observed at t+dt
    truth:    np.ndarray | None = None   # clean next state, for scoring only

@dataclass(frozen=True)
class Belief:
    mean: np.ndarray                  # in representation coordinates
    cov:  np.ndarray | None = None
    space: str = ""                   # which MechanicsRepresentation
    extras: dict = field(default_factory=dict)

@dataclass(frozen=True)
class StepRecord:
    prediction:       np.ndarray      # predicted next_obs, raw units, PRE-update
    innovation:       np.ndarray      # the residual that DROVE the update
    innovation_space: str             # "observation" | "normalised_delta" | "torque"
    loss:             float
    seconds:          float
    extras:           dict
```

`Transition.truth` is how the clean-scoring identity from
[streams.py:138](latent_mechanics/mismatch/streams.py#L138) becomes a property of
the data rather than a bolt-on in one package.

`innovation_space` is the whole of correction 2, in one string field.

### 2.2 The four swappable interfaces

Plain `typing.Protocol` — structural, no inheritance required, so existing objects
can satisfy them without edits.

```python
class Plant(Protocol):
    dt: float
    def rollout(self, excitation, n_steps) -> StateTrajectory: ...

class ObservationModel(Protocol):
    def observe(self, states, dt, rng) -> np.ndarray: ...

class Predictor(Protocol):
    def predict(self, obs, action, belief) -> np.ndarray: ...

class MechanicsRepresentation(Protocol):
    dim: int
    name: str
    def initial(self) -> np.ndarray: ...
    def prior_covariance(self) -> np.ndarray | None: ...
    def to_predictor(self, x: np.ndarray) -> np.ndarray: ...

class Estimator(Protocol):
    name: str
    def initialize(self) -> Belief: ...
    def update(self, belief, transition) -> tuple[Belief, StepRecord]: ...
```

### 2.3 The driver

```python
def run(estimator, predictor, transitions, *, verify_frozen=True) -> Trace:
    belief = estimator.initialize()
    records = []
    for tr in transitions:
        prediction = predictor.predict(tr.obs, tr.action, belief)   # BEFORE update
        belief, rec = estimator.update(belief, tr)
        records.append(replace(rec, prediction=prediction))
    ...
```

This is `run_online_adaptation` with the prediction lifted out of `observe()`.
The prequential guarantee, the frozen check and `AdaptationLog`'s metrics all
carry over unchanged.

### 2.4 Why `MechanicsRepresentation` is worth its 110 lines

It is the axis the brief explicitly asks for, and **the code is already 90% there
and just misnamed**. `LatentBasis` ([belief/basis.py](latent_mechanics/belief/basis.py))
is precisely a mechanics representation — a frozen affine chart with `encode` /
`decode` / `decode_covariance`. Promoting it and adding two siblings costs almost
nothing:

| implementation | dim | source |
|---|---|---|
| `FullLatent` | 16 | identity map, wraps the embedding table |
| `ReducedLatent` | 6 | **existing `LatentBasis`, renamed** |
| `PhysicalParameters` | 5 | $[I, \mu, b, k, c]$, extracted from `RLSAdaptor` |
| `Hybrid` | d₁+d₂ | concatenation — trivial once the above exist |

This also fixes the `RLSAdaptor.latent` name-lie (audit C7) and makes
`belief_travel` a well-defined quantity again.

---

## 3. Old → new module mapping

### Preserve unchanged (behaviour must be bit-identical)

| module | why |
|---|---|
| `belief/ukf.py` | reference-validated to 1e-10. Do not touch. |
| `belief/noise.py` | adaptive R is where the UKF's win actually comes from |
| `belief/basis.py` | **rename/re-export** as `ReducedLatent`; math unchanged |
| `model.py` | the predictor stays; gains an interface *above* it |
| `excitation.py` | ZOH profiles, correct as-is |
| `dataset.py`, `rollout.py` | data access and rollout metrics |
| `provenance.py` | checkpoint hashing |
| `train.py`, `evaluate.py` | Stage-1 training, off the online path |
| `geometry/*` | analysis-only, reads artifacts |
| `dyn.rls_init` / `rls_step` | the RLS kernel — **must not move** |

### Wrap (new interface, existing algorithm called unmodified)

| existing | new adapter | guarantee |
|---|---|---|
| `StaticLatentAdaptor` | `estimators/static.py` | identical predictions |
| `GradientLatentAdaptor` | `estimators/gradient.py` | identical latent trajectory |
| `UKFLatentAdaptor` | `estimators/ukf.py` | identical belief trajectory |
| `RLSAdaptor` | `estimators/rls.py` | **identical to float64** |
| `SensorPipeline` | `observation.py:SensorObservation` | identical corrupted stream |
| `MechanicsDynamicsModel` | `predictor.py:LatentNetworkPredictor` | identical outputs |

The adapters **call the existing classes**. No algorithm is reimplemented. This is
what makes Phase-3 equivalence checkable rather than hopeful.

### Simplify

| item | change |
|---|---|
| 8 adaptor factories (audit C3) | one `build_estimator(spec)` |
| 4 scoring implementations | one `metrics.py` |
| 4 `write_csv` copies | one |
| 3 config systems + `UKFConfig` | one `ExperimentSpec` per experiment |
| 3 simulators (audit C4) | one `MujocoPlant` with a perturbation hook |
| `dyn.DT` reached into 41× | `plant.dt`, a property of the plant |
| `data_gen.episode_length` global mutation | `n_steps` as an argument |

### Remove

| item | justification |
|---|---|
| `NormStats` class | `validate()` never called; keep `KEYS` as a tuple |
| `OnlineAdaptor.reset(*args, **kwargs)` | never called by the driver |
| `episode_boundaries` threading | keep the function; stop passing it through 8 modules |
| `InnovationAdaptiveNoise` | superseded by the residual form, by its own docstring |
| `UKFConfig.floor` | legacy-only field |
| `AdaptationLog.extras` silent drop | record everything or fail loudly |

### Explicitly **not** doing

- No dependency-injection container, no plugin registry, no abstract factory.
- No renaming of things that are already clear.
- **No algorithmic changes.** Not fixing B1/B2/B3. Not retuning anything. Not
  replacing the UKF. Those are research decisions and the brief says they wait
  until the simplified system is understood.
- Not touching `perception/`, `iiwa/` or `archive/` — outside the loop.

---

## 4. Migration plan

Additive and reversible. **The old packages keep working at every step**, so no
result is ever unreproducible.

### Stage 0 — pin the baseline *(do first)*
Record current outputs of everything cheap enough to re-run, into
`validation/baseline/`. Without this, Phase 3 has nothing to compare against.

### Stage 1 — build `mechanics/` alongside the old code
Nothing imports it yet. Old code untouched. Add unit tests for the new types.

### Stage 2 — adapters + equivalence tests
Each adapter gets a test asserting **bit-identical or 1e-12** agreement with the
class it wraps, on a fixed stream. This is the load-bearing step: it is what makes
the refactor a refactor.

### Stage 3 — the sanity experiment (§6 of the brief)
First real consumer of the new core. Small, visual, end-to-end.

### Stage 4 — migrate experiments one at a time
Per experiment: run old, run new, compare within tolerance, commit. Order chosen
by ratio of value to risk:

1. `estimator_convergence` (Stage-2 exp 1/3) — the most-used path
2. `observation_noise` (Stage-3 sensor sweeps) — exercises `ObservationModel`
3. `model_mismatch` (Stage-3 plant sweeps)
4. `cross_mechanism_generalization` (Stage 4)
5. `predictor_accuracy` — new, but nearly free once the interface exists

Stage 5 (curriculum) migrates last: it retrains models, so it is expensive and
least urgent.

### Stage 5 — delete the old drivers, only after their replacements match
`online/experiments.py`, `mismatch/study.py`, `mechanisms/study.py`,
`curriculum/study.py`, `belief/sweep.py` become thin shims, then go.

### Rollback
Every stage is a separate commit; stages 1–3 add files only.

---

## 5. The experiment system

One dataclass, declared per experiment, that answers every question the brief
lists in a single readable block:

```python
SPEC = ExperimentSpec(
    question = "Does a UKF over a 6-D latent recover an unseen door's mechanics "
               "faster than gradient descent, under a clean observation model?",

    plant        = MujocoPlant(family="door", instance=52),
    observation  = IdentityObservation(),
    predictor    = LatentNetworkPredictor(checkpoint=".../all_families/best.pt",
                                          expected_sha256="a3f1..."),
    representation = ReducedLatent(basis=".../latent_basis.npz", dim=6),

    estimators   = ["no-adaptation", "gradient", "ukf", "rls-5p"],
    initialization = "medoid",
    disturbances = [],
    metrics      = ["nrmse_final", "steps_to_converge", "us_per_update"],
    seeds        = [0],
)
```

Rules kept deliberately strict, because they are what make the directory useful:

- **`question` is mandatory and is a sentence.** If you cannot write it, the
  experiment is not ready.
- **`"no-adaptation"` is included by default** and warns if removed. It is what
  produced Stage 5's negative result; making it optional would be a regression.
- Every spec names its predictor checkpoint **with a sha256 pin** — this closes
  audit item B1 structurally: two experiments using different predictors becomes
  visible by diffing two spec files.
- One `run.py` per directory, ~20 lines, calling the shared runner.

---

## 6. The canonical sanity experiment

`experiments/sanity_one_door/` — deliberately the smallest thing that exercises
every component:

1. One door, one held-out instance, ~600 transitions.
2. Start from a **deliberately wrong** belief (the medoid of a different family).
3. Predict the next observation each step.
4. Compare with the actual observation.
5. Update the belief.
6. Produce three panels:
   - true θ(t) vs predicted θ(t)
   - prediction error over time, adapted vs no-adaptation control
   - belief over time, projected into the frozen PCA chart, with the
     nearest-training-object's true parameters annotated

Success is **not** accuracy. Success is that a reader can point at each arrow in
the diagram and name the variable, the file and the line.

---

## 7. Risks

| risk | mitigation |
|---|---|
| Refactor silently changes a result | Stage-2 equivalence tests at 1e-12; Stage 0 baseline pinning |
| RLS baseline drifts | adapter calls `dyn.rls_init`/`rls_step` unmodified; float64 equality test |
| Two parallel systems during migration | strictly time-boxed; old drivers deleted in Stage 5 |
| New abstraction turns out unhelpful | it is 700 lines and additive — deletable |
| Losing the reasoning in the uncommitted doc-compression diff | decide on that diff **before** starting (audit §E) |

---

## 8. What this does not fix

Stated plainly so it is not mistaken for progress:

- **B1** — the stages still use nine different predictors. The refactor makes the
  discrepancy *visible* (sha256 in every spec) but does not resolve it. Resolving
  it means re-running Stage 3, which is an algorithmic/experimental decision.
- **B2, B3** — untouched by design.
- **Single-seed** — the spec has a `seeds` list; nothing is replicated yet.
- **No perception** — `ObservationModel` gives vision a place to plug in. It does
  not implement it.
- **Partial observability** — a static belief still cannot represent a hidden
  dynamic state. That needs an architecture change to the *predictor*, which the
  new interface makes possible but does not perform.

---

## 9. Recommendation

Proceed with Stages 0–3 (baseline, core, adapters, sanity experiment) and migrate
`estimator_convergence` as the first real experiment. Stop there and review before
migrating the rest — that is the point at which the simplified system is
understandable, and per the brief, the next algorithmic decision should be made
from there rather than before.
