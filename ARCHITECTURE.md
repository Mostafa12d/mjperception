# ARCHITECTURE.md — the refactored system

*Final deliverable. Read [CURRENT_SYSTEM.md](CURRENT_SYSTEM.md) for what this
replaced and [REFACTOR_PROPOSAL.md](REFACTOR_PROPOSAL.md) for why.*

---

## A. What are we trying to prove, and what happens at every timestep?

### The claim

A robot interacting with an articulated object can estimate its hidden mechanics —
inertia, Coulomb friction, viscous damping, closer-spring stiffness — **online,
from the interaction it is already performing**, by watching where a learned
dynamics model's predictions go wrong.

Formally: the plant is $s_{t+1} = F(s_t, a_t; \phi_i)$ with hidden parameters
$\phi_i$. We have a **frozen** network $f_\psi$ and a per-object **mechanics
belief** $x_i$ such that $s_{t+1} \approx f_\psi(s_t, a_t, z_i)$, where
$z_i = g(x_i)$ maps the belief into whatever the predictor consumes. Only $x_i$ is
estimated online; $\psi$ never moves, and this is enforced by checksum rather than
assumed.

The specific claim under test — and the project's strongest result — is that the
crossover against classical system identification is **sensing, not physics**. RLS
must form $\ddot\theta = \Delta\dot\theta/\Delta t$, so encoder noise enters its
regressor and biases the estimate. The learned predictor never differentiates.
Reproduced on the new core in `experiments/observation_noise`: RLS is 12× more
accurate than the UKF on a clean encoder, and already behind it at
$\sigma = 10^{-4}\,\mathrm{rad}$ — about 14-bit resolution over the door's travel.

### What happens at one timestep

```
        ┌─────────────────────────────────────────────────────────────────┐
        │  Plant  (MuJoCo, one mechanism instance, hidden φᵢ)             │
        └───────────────────────────┬─────────────────────────────────────┘
                                    │  sₜ = [θ, θ̇]   ground truth, never seen
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  ObservationModel        mechanics/observation.py               │
        │  identity | noisy encoder | quantised | delayed | partial       │
        └───────────────────────────┬─────────────────────────────────────┘
                                    │  oₜ, oₜ₊₁ (+ truth kept aside for scoring)
                                    ▼
                            Transition(o, a, o′, truth)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  ┌───────────────────────────┐              ┌──────────────────────────────┐
  │ Predictor.predict         │              │ Estimator.update             │
  │   (oₜ, aₜ, belief) → ô    │              │   (belief, transition)       │
  │ mechanics/predictor.py    │              │ mechanics/estimators/        │
  │ learned | analytical |    │              │                              │
  │ misspecified              │              │  forms its OWN innovation ν  │
  └───────────┬───────────────┘              │  in its OWN declared space   │
              │                              │  (the Predictor is h(·) for  │
              │ ô − o′  =  reported error    │   the latent estimators)     │
              │ (prequential, raw SI)        └──────────────┬───────────────┘
              │                                             │
              ▼                                             ▼
        StepRecord.error                            StepRecord.innovation
                                                    + innovation_space
                                                             │
                                                             ▼
                                             Belief(mean, cov, space)
                                             in MechanicsRepresentation
                                             coordinates
```

**The one thing to understand about this diagram**: `error` and `innovation` are
*different quantities in different spaces*, and that is real, not an artifact.

| estimator | innovation | space |
|---|---|---|
| gradient | per-sample residual (but the update is a gradient step on the last 32) | normalised delta |
| UKF | SLR-linearised $\nu = y - (Ax_{\text{prior}} + b)$ | normalised delta |
| RLS | $\tau - \varphi^\top\theta$ | **torque** |

The UKF works in normalised-delta space deliberately: it is the only space in
which one measurement-noise matrix $R$ is meaningful across mechanism families.
The old code had this same behaviour but never surfaced it, so the three
estimators' `loss` columns were silently incommensurable. Now every trace carries
`innovation_space` and says which.

### The loop, in code

The entirety of `mechanics/loop.py`:

```python
belief = estimator.initialize()

for transition in transitions:
    prediction     = predictor.predict(transition.obs, transition.action, belief)
    belief, record = estimator.update(belief, transition)
```

Two protocol guarantees, both inherited because they are what make the numbers
mean anything:

- **Prequential.** The step-$t$ prediction uses the belief held *before* step $t$.
  Never shuffled, never batched, never revisited. Tested.
- **Frozen predictor.** Verified by float64 parameter checksum plus a
  gradient-leak check after every run, not assumed.

And one that is now structural rather than conventional:

- **Scored against clean truth, fed corrupted observations.** `Transition.truth`
  carries the clean next state; `target` returns it when present. So a noisy
  observation model measures *the estimator*, not the sensor.

---

## B. Code map

### `mechanics/` — the experimental core (~1,050 lines)

| module | what it does | receives | outputs | research component |
|---|---|---|---|---|
| `types.py` | the vocabulary | — | `Transition`, `Belief`, `StepRecord`, `Trace` | the arrows in the diagram |
| `loop.py` | `run()` — the driver | estimator, predictor, transitions | `Trace` | **the research loop** |
| `observation.py` | what can be measured | clean state sequence | observation sequence | **observation model** |
| `predictor.py` | one-step prediction | `(obs, action, belief)` | predicted next obs | **dynamics predictor** |
| `representation.py` | belief coordinates | belief vector | predictor input | **mechanics representation** |
| `estimator.py` | the estimator protocol | — | — | **online estimator** |
| `estimators/static.py` | the no-adaptation control | transition | unchanged belief | control |
| `estimators/gradient.py` | Adam on the latent | transition | belief + innovation | Stage-2 update rule |
| `estimators/ukf.py` | UKF in a reduced chart | transition | Gaussian belief + $\nu$ | current best method |
| `estimators/rls.py` | RLS on physical params | transition | Gaussian belief + $\nu$ | **the baseline** |
| `data.py` | dataset → transitions | `.npz`, observation model | `list[Transition]` | plant + observation |
| `build.py` | `Workspace`, `build_method` | checkpoint, method name | `Method` | wiring (was 8 copies) |
| `metrics.py` | one scoring implementation | trace, transitions | metric dict | scoring (was 4 copies) |
| `tests.py` | unit + **equivalence** checks | — | exit code | the safety net |

### Component implementations available today

```
Predictor            LatentNetworkPredictor   frozen Stage-1 MLP
                     AnalyticalPredictor      the identified hinge ODE
                     MisspecifiedPredictor    deliberately wrong, for controls

ObservationModel     IdentityObservation      perfect proprioception
                     JointSensor              noise / quantisation / dropout / latency
                     PartialObservation       drop channels (e.g. no velocity)

Representation       FullLatent               16-D embedding
                     ReducedLatent            frozen PCA chart (6-D)
                     PhysicalParameters       [I, μ, b, k, c]
                     Hybrid                   concatenation of two

Estimator            StaticEstimator          the control
                     GradientEstimator        Adam + Robbins–Monro decay
                     UKFEstimator             sigma points + adaptive R
                     RLSEstimator             recursive least squares
```

### `experiments/` — one directory per question

| experiment | question | status |
|---|---|---|
| `sanity_one_door/` | can a deliberately wrong belief be corrected? | **new**, canonical |
| `estimator_convergence/` | does adaptation beat the control; does RLS still win? | migrated from Stage 2 |
| `observation_noise/` | where is the sensing crossover? | migrated from Stage 3 |

### What was left untouched

`latent_mechanics/` is **fully intact and all seven of its test suites still
pass**. Nothing was deleted. The new core wraps the old algorithms; it does not
replace them. `belief/ukf.py`, `belief/noise.py`, `dyn.rls_init`/`rls_step`,
`model.py`, `train.py`, `geometry/`, `perception/`, `iiwa/` are unmodified.

---

## C. How to modify this project

### "I want to test a new observation — say, vision-based joint estimates."

**One file.** Add a class to `mechanics/observation.py`:

```python
@dataclass
class VisionObservation:
    name: str = "vision"

    def observe(self, states, dt, rng):
        # render → estimate θ → return (T, n_obs)
        ...

    def describe(self) -> dict:
        return {"name": self.name, ...}
```

Then in any spec: `observation=VisionObservation()`. Nothing else changes —
scoring already targets clean truth, so the metric measures your estimator, not
your camera.

*If the observation has a different dimension* (e.g. angle only), you also need a
predictor that consumes it. That is the honest boundary: a genuinely different
observation space is a modelling change, not a wiring change.

### "I want to replace the UKF."

**Implement two methods** (`mechanics/estimator.py`):

```python
class MyEstimator:
    name = "my-estimator"

    def initialize(self) -> Belief: ...

    def update(self, belief, transition) -> tuple[Belief, StepRecord]:
        y     = self.predictor.measurement(transition.obs, transition.next_obs)
        y_hat = self.predictor.predict_measurement(
            transition.obs, transition.action, belief.mean[None, :])[0]
        nu = y - y_hat                      # your innovation
        new_mean = ...                      # your update rule
        return Belief(mean=new_mean, space=self.representation.name), \
               StepRecord(..., innovation=nu,
                          innovation_space=self.predictor.measurement_space, ...)
```

Add one branch to `build_method` in `mechanics/build.py`. Done — every experiment
picks it up by name. A particle filter, an EKF, or a learned GRU update all fit
this shape; none of them need to know how the belief is parameterised.

### "I want to test a different predictor."

Implement `predict(obs, action, belief)` in `mechanics/predictor.py`. If a filter
must push sigma points through it, also implement `measurement` and
`predict_measurement` (the `MeasurementPredictor` protocol).

Two already exist as worked examples: `AnalyticalPredictor` (physics, no network)
and `MisspecifiedPredictor` (wraps another predictor and distorts it, so
`gain=1, bias=0` is provably the identity and any deviation is attributable).

To swap architectures only, retrain via `latent_mechanics/train.py` and point the
spec's `checkpoint` at the result.

### "I want to add a transient-disturbance experiment."

```
experiments/transient_disturbance/
    __init__.py
    run.py          ← an ExperimentSpec and ~20 lines
```

The disturbance itself is a plant perturbation
(`latent_mechanics/mismatch/perturbations.py`, which already has Stribeck,
position-dependent friction, drift and compliance) or an observation model, and it
goes in the spec. `experiments/observation_noise/run.py` is the template for a
sweep over levels.

### "I want a different mechanics representation."

Implement the five-method protocol in `mechanics/representation.py`. `Hybrid`
already shows how to combine two. Note that `GradientEstimator` requires
`FullLatent` and `UKFEstimator` requires `ReducedLatent` — both raise a clear
`TypeError` explaining why, rather than silently projecting.

### Two traps the code now catches for you

1. **A latent chart from the wrong checkpoint.** A PCA basis fitted on one
   embedding table spans a different subspace than another; projecting through it
   destroys the belief silently. `Workspace.basis()` computes the chart for *its
   own* checkpoint, and an explicitly-passed basis is rejected if its
   `source_table` does not match. This was found by the sanity experiment during
   this refactor — the shipped default basis is fitted on the 120-object
   all-families table, and using it with the 48-door checkpoint cost the UKF a
   factor of two.

2. **Dropping the control.** `ExperimentSpec.validate()` warns if
   `no-adaptation` is absent from `methods`.

---

## D. Verification

Everything below was run, not assumed.

| check | result |
|---|---|
| 7 pre-existing test suites | **all pass**, unmodified |
| `mechanics/tests.py` | **60/60 pass** |
| `no-adaptation` vs legacy | error, belief, loss **bit-identical** |
| `gradient` vs legacy | error, belief, loss **bit-identical** |
| `rls-5p`, `rls-3p` vs legacy | error, belief, loss **bit-identical** |
| `ukf` vs legacy | error and loss bit-identical; belief identical at legacy's float32 storage precision (float32 diff **0.0**) |
| `AnalyticalPredictor` vs `RLSAdaptor.predict` | bit-identical over 200 random states |
| `JointSensor` vs `SensorPipeline` | identical on all five configurations |
| stream construction vs `episode_stream` | identical over 2,392 transitions |
| clean-scoring identity | matches `mismatch.clean_errors` |

Two equivalence deltas were investigated rather than tolerated. Both are float32
storage precision in the *legacy* code (relative diff 1.9e-07 against a float32
epsilon of 1.19e-07); casting the new float64 result to float32 reproduces the
legacy value exactly. The new path is the more precise of the two.

### Results reproduced on the new core

`estimator_convergence` (4 unseen doors, clean observations) — RLS wins decisively
on a clean, linear-in-parameters plant, as Stage 2 reported:

| method | tail nRMSE | µs/update | gain vs control |
|---|---|---|---|
| no-adaptation | 0.0383 | 83 | — |
| gradient | 0.0143 | 319 | 2.7× |
| ukf | 0.0094 | 537 | 4.1× |
| **rls-5p** | **0.00075** | **6.8** | **50.9×** |

`observation_noise` — and the crossover, as Stage 3 reported:

| encoder σ [rad] | no-adaptation | gradient | ukf | rls-5p |
|---|---|---|---|---|
| 0 | 0.0383 | 0.0143 | **0.0094** | **0.0008** |
| 1e-4 | 0.1277 | 0.1758 | **0.1261** | 0.3283 |
| 3e-4 | 0.3512 | 0.4685 | **0.3364** | 0.9927 |
| 1e-3 | 0.7964 | 0.7899 | **0.7093** | 1.2900 |

RLS leads by 12× at σ=0 and is behind the UKF by σ=1e-4 — about 14-bit resolution
over the door's 2.26 rad travel, matching the documented crossover point. It
reaches nRMSE ≈ 1.0 (no better than predicting "nothing changes") by σ=3e-4.

This is a *reproduction on a subset*, not a re-run of the published study. It used
4 objects and 3 episodes on the doors-only predictor.

---

## E. What this deliberately did not do

Per the brief: **stop after the refactor and validation; the next algorithmic
decision comes after the simplified system is understood.**

- **No algorithmic changes.** No retuning, no new update rules, no "improvements".
- **B1 unresolved.** The stages still use nine different predictors. The refactor
  makes it *visible* — every `spec.json` records its checkpoint and can pin a
  sha256 — but re-running Stage 3 on the all-families predictor is a research
  decision, not a refactor.
- **B2, B3 untouched.**
- **Still single-seed.** `ExperimentSpec.seeds` is a list; nothing has been
  replicated.
- **Stages 3–5 not yet migrated.** `mismatch/study.py`, `mechanisms/study.py`,
  `curriculum/study.py` and `belief/sweep.py` still run their own drivers and
  still work. The proposal's migration order is in
  [REFACTOR_PROPOSAL.md §4](REFACTOR_PROPOSAL.md).
- **No vision.** `ObservationModel` gives it somewhere to plug in. That is all.

### Commit hygiene — do not land this as one commit

The working tree contains **three unrelated changesets**, only one of which is
mine:

1. **Algorithm work (pre-existing, uncommitted).** The residual-adaptive-R and
   iterated-update (IPLF) feature set — `ResidualAdaptiveNoise`, `psd_floor`,
   `IRREDUCIBLE_R`, `UnscentedKalmanFilter.iterated_update` — plus changed
   `UKFConfig` defaults (`noise_kind: adaptive → residual`, `n_iterations: 3`) and
   its tests. **None of this is in `HEAD`.** The documented UKF results describe
   this code, so right now they are unreproducible from any commit.
2. **A docstring-tightening pass (pre-existing),** interleaved into the same files.
3. **This refactor (additive):** `mechanics/`, `experiments/`, the three markdown
   deliverables, and a `README.md` edit.

Landing them together makes `git revert` useless — you could not back out the
refactor without also backing out the UKF work — and it directly contradicts the
brief's own instruction to separate refactoring changes from algorithmic ones.

It also weakens the central claim of §D. The equivalence tests prove the new core
matches the legacy code **as it stands in the working tree**. They say nothing
about `HEAD`. If the refactor and the UKF feature work land in one commit, no
future reader can tell which of the two moved the numbers.

Suggested split:

```
1. belief: residual-form adaptive R and iterated (IPLF) UKF update
     latent_mechanics/belief/{noise,ukf,adaptor,sweep,test_ukf_reference}.py
     latent_mechanics/belief/README.md, latent_mechanics/geometry/analyses.py
     + the 4 untracked modules (calibrate_noise, ablation, figures, online_latents)

2. docs: tighten module docstrings           (the remaining ~40 files)

3. mechanics: minimal experimental core + equivalence tests
     mechanics/, experiments/, ARCHITECTURE.md, CURRENT_SYSTEM.md,
     REFACTOR_PROPOSAL.md, README.md
```

Commits 1 and 2 are interleaved inside several files, so splitting them cleanly
needs `git add -p`. If that is not worth the effort, fold them together — the
boundary that matters is **(algorithm + docs) vs (refactor)**.

Whichever way it lands, `docs/PROJECT_STATUS.md` needs its "Working tree clean"
and "@ `956ddd9`" lines corrected.

---

## F. Quick start

```bash
# the smallest thing that exercises every component, with a narrated timestep
python3.10 -m experiments.sanity_one_door.run

# does adaptation beat the control? does RLS still win?
python3.10 -m experiments.estimator_convergence.run

# where is the sensing crossover?
python3.10 -m experiments.observation_noise.run

# the safety net: unit checks + bit-equivalence against the legacy implementations
python3.10 -m mechanics.tests
```
