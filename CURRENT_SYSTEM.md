# CURRENT_SYSTEM.md — audit of the codebase as it stands

*Written against the working tree at `main` @ `5973806`, all seven test suites
passing (verified by running them). No code was modified to produce this
document.*

This is a **critical** audit. Where the implementation embodies a choice that is
defensible but not obviously correct, it says so. Where the existing
documentation overstates what was shown, it says that too.

---

## A. The research goal

### A.1 The problem

A robot interacting with an articulated object — a door, a drawer, a laptop lid —
needs to know its **mechanics**: how much it weighs, how much its hinge sticks,
whether a spring pulls it closed. Those quantities are not visible; they only
reveal themselves through motion under load. The project asks whether a robot can
estimate them **online, from the interaction it is already performing**, rather
than from a dedicated calibration routine.

Formally, each object instance $i$ has hidden mechanics. The plant evolves as

$$ s_{t+1} = F(s_t, a_t; \phi_i), \qquad s_t = [\theta_t, \dot\theta_t] \in \mathbb{R}^2,\quad a_t = \tau_t \in \mathbb{R} $$

where $\phi_i$ are true physical parameters (inertia, Coulomb friction, viscous
damping, spring stiffness, spring reference). $\phi_i$ is **never** an input to
any model — it is analysis metadata only ([`door_sampler.ground_truth`](latent_mechanics/door_sampler.py), [`library.ground_truth`](latent_mechanics/mechanisms/library.py#L278)).

### A.2 The central hypothesis

There is a learned **mechanics code** $z \in \mathbb{R}^{16}$ and one shared
dynamics network $f_\psi$ such that

$$ s_{t+1} \approx f_\psi(s_t, a_t, z_i) $$

for every object $i$, and $z_i$ can be recovered **online** from the prediction
residual, faster and more robustly than classical system identification recovers
$\phi_i$. The claimed structural advantage is **degraded sensing**: RLS must form
$\ddot\theta = \Delta\dot\theta/\Delta t$ and so puts sensor noise directly into
its regressor; the learned model never differentiates.

### A.3 The learned dynamics predictor

[`MechanicsDynamicsModel`](latent_mechanics/model.py#L60) — an MLP
`[256, 256]`, SiLU, mapping $(s, a, z) \in \mathbb{R}^{2+1+16} \to \mathbb{R}^2$.

Two details matter more than the architecture:

- **It predicts the delta, not the next state** (`predict_delta=True`). Over one
  20 ms step the state barely changes, so predicting $s_{t+1}$ directly makes the
  identity map near-optimal and the latent contributes nothing measurable.
- **Normalisation lives in the module as buffers**, so a checkpoint consumes and
  returns raw SI units. `raw_output()` returns the *normalised delta* (what
  training regresses); `forward()` returns the *raw next state*. **These two
  spaces are the single biggest source of confusion in the codebase** — see §B.5.

### A.4 The mechanics belief

Currently a **point or Gaussian estimate over $z$**, and what exactly it is
depends on which estimator you picked:

| estimator | belief | dimension | space |
|---|---|---|---|
| `StaticLatentAdaptor` | fixed $z$ | 16 | latent |
| `GradientLatentAdaptor` | point $z$ | 16 | latent |
| `UKFLatentAdaptor` | Gaussian $(x_r, P_r)$ in a frozen PCA chart, decoded to 16-D | 6 (rank-6 in 16-D) | latent |
| `RLSAdaptor` | Gaussian $(\theta, P)$ | 5 | **physical parameters** $[I, \mu, b, k, c]$ |

RLS is the odd one out and the code knows it: `RLSAdaptor.latent` returns physical
parameters with a comment admitting these are "not comparable to `z` and never
projected into the embedding PCA" ([rls_adaptor.py:78-81](latent_mechanics/online/rls_adaptor.py#L78)).
The `AdaptationLog.latents` array is therefore **not a homogeneous quantity across
methods**, though downstream code treats it as one (e.g. `belief_travel` is
computed as `‖latents[-1] − latents[0]‖` for every method in
[mismatch/study.py:168](latent_mechanics/mismatch/study.py#L168) and
[mechanisms/study.py:105](latent_mechanics/mechanisms/study.py#L105) — a 16-D
latent norm and a 5-D physical-parameter norm reported in the same column).

### A.5 What is estimated online

Only $z$ (or $\theta_{RLS}$). **The dynamics network is frozen and provably so** —
this is enforced hard, and it is one of the genuinely good parts of the design:
`_assert_frozen()` at construction, plus `assert_network_unchanged()` comparing a
float64 parameter checksum and checking for leaked `.grad`
([adaptor.py:106-130](latent_mechanics/online/adaptor.py#L106)).

### A.6 The UKF's role

It is the **belief-update rule**, replacing gradient descent on $z$. Structure:

- **State**: $x_r \in \mathbb{R}^6$, the coordinates of $z$ in a frozen PCA basis
  of the training embedding table ([`LatentBasis`](latent_mechanics/belief/basis.py)).
- **Process model**: $f_x = \mathrm{identity}$. The mechanics are constant; $Q$
  alone decides how much a settled belief may still move.
- **Measurement model**: $h(x_r) = f_\psi^{\text{raw}}(s_t, a_t, V^\top x_r + \bar z)$ —
  i.e. **the predictor itself is the observation function**, evaluated in
  normalised-delta space.
- **Measurement**: $y_t = \mathrm{normalise}(s_{t+1} - s_t)$.
- **Innovation**: $\nu_t = y_t - \hat y_t$ in normalised-delta space.
- Adaptive $R$ via the residual form (Mohamed & Schwarz 1999), with a matrix
  floor `IRREDUCIBLE_R`.

Critically: **`PROJECT_STATUS.md` itself reports that the credit belongs to
adaptive $R$, not to the UKF machinery** — fixed-$R$ UKF performs at the
gradient-descent level (1.92× vs 1.72× ceiling ratio). So the UKF is *not*
established as the right estimator; an adaptive-noise-weighted least squares might
do as well. This is a strong argument for the refactor the user is asking for.

### A.7 Observations

**Currently: simulator ground truth.** `data.qpos[hinge]` and `data.qvel[hinge]`,
read directly out of MuJoCo ([run_door_dynamics_validation.py:176-178](baseline/run_door_dynamics_validation.py#L176)),
sliced to the model rate. There is no observation model in the main path.

The one exception is [`mismatch/sensors.SensorPipeline`](latent_mechanics/mismatch/sensors.py),
which adds noise → quantisation → dropout → latency. It exists **only inside the
`mismatch/` package**, is applied at stream-build time
([streams.py:114](latent_mechanics/mismatch/streams.py#L114)), and is wired only
into the door path. The mechanism suite (`mechanisms/rollout.py`) has **no sensor
hook at all**.

### A.8 Actions

$a_t = \tau_t$, a scalar hinge torque, **zero-order held on the model timestep
grid** — this is load-bearing, and `transitions_from_log` raises if a profile
violates it ([data_gen.py:96-102](latent_mechanics/data_gen.py#L96)).

There is a subtlety that is easy to miss: for the door path, the action is
`tau_ft`, the hinge torque **reconstructed from the commanded handle force**
($\tau = (r \times F)\cdot\hat a$), not MuJoCo's own `qfrc_applied` projection
(`tau_oracle`). For the mechanism suite the action is applied **directly to
`qfrc_applied[dof]`**. These are different actuation semantics, which is why
Stage-4 data is explicitly not mixable with Stage-1–3 data
([library.py:6-8](latent_mechanics/mechanisms/library.py#L6)).

### A.9 What is predicted, and what generates the innovation

**These are not the same quantity, and that is the core architectural finding of
this audit.** See §B.5 for the full trace. In summary, per estimator:

| estimator | reported `prediction` | residual that actually drives the update | space |
|---|---|---|---|
| gradient | $f_\psi(s_t,a_t,z)$, raw SI next state | MSE over a **32-step sliding window**, normalised delta | different |
| UKF | $f_\psi(s_t,a_t,z)$, raw SI next state | SLR-linearised innovation $\nu$, normalised delta | different |
| RLS | ODE integrated forward one step, raw SI | $\tau - \varphi^\top\theta$, **torque** | very different |

The `AdaptorStep.loss` field consequently holds three incommensurable quantities
depending on which object produced it.

### A.10 Experiments implemented

| stage | driver | question |
|---|---|---|
| 1 | `train.py`, `evaluate.py` | can one net + per-object $z$ fit a door population? |
| 2 | `online/experiments.py` | does online adaptation of $z$ beat no adaptation, and RLS? |
| 3 | `mismatch/study.py` | how do both degrade under plant and sensor mismatch? |
| 4 | `mechanisms/study.py` | does $z$ transfer across six mechanism families? |
| 5 | `curriculum/study.py` | does training diversity help at fixed budget? |
| — | `geometry/report.py` | what structure does the latent space have? |
| — | `belief/sweep.py`, `ablation.py`, `drift_check.py` | UKF settings and ablations |
| — | `online/hparam_sweep.py` | gradient-adaptor hyperparameters (built, **not applied**) |
| — | `mechanisms/time_resolution.py` | per-family sampling rate (built, **inert by default**) |

### A.11 Demonstrated vs hypothesised

**Demonstrated (with the caveats below):**
- One network + per-object latent fits a door population; the latent is
  load-bearing (wrong $z$ costs 3.8×, zero $z$ 8.7×).
- A linear probe recovers friction/inertia/stiffness from $z$ (R² 0.80–0.90) —
  parameters never given as input. Damping is *not* encoded (R²=0.02).
- Online adaptation beats a no-adaptation control **when the control exists**;
  the control is what revealed that constant-step-size adaptation is a net loss.
- The frozen-network guarantee is real and mechanically enforced.
- RLS beats the latent method decisively on clean, well-modelled plants (24×).
- RLS collapses under encoder noise (1873×) and quantisation (464×) where the
  latent degrades far less. This is the project's strongest claim.
- The UKF/adaptive-R beats no-adaptation on all six families.
- The UKF core is validated to 1e-10 against `filterpy`.

**Still hypothesis, or actively undermined:**
- **That the latent is a "mechanics code" in any categorical sense.** The
  geometry investigation is honest and negative: effective dimensionality 2.36 of
  16; cross-validated likelihood cannot distinguish the latents from a single
  Gaussian; the significant silhouette is **scale, not category** — residualising
  against *observed* scale drives excess silhouette to +0.016 at p=0.15. The
  latent is mostly a one-dimensional "how big and heavy is this thing" axis.
- **That the UKF is the right estimator.** Fixed-$R$ UKF ≈ gradient descent. The
  win is the adaptive noise model, which is not UKF-specific.
- **That the sensing-crossover result holds for the system as it now stands.**
  Open item B1: Stage 3 ran on the doors-only predictor; everything downstream
  uses the all-families predictor. The headline claim has never been reproduced
  on the model the rest of the project uses.
- **Anything about generalisation at scale.** Stage 5 is a negative result at
  fixed budget; break-even extrapolates to ~9–10 families and was not reached.
- **Anything statistical.** Single seed throughout. Stage 5's gain trend is
  p=0.076, i.e. not significant.
- **Anything about perception or hardware.** Simulation only; sensor degradation
  is injected synthetically into otherwise-perfect states.
- **Partial observability.** Unsolved by construction — a static latent cannot
  represent a hidden dynamic state (the bifold leaf).

---

## B. The actual data flow, one complete timestep

Traced through the **UKF path** (`belief/sweep.py`), which is the current best
method. Every arrow names the variable, where it is produced, where it is
consumed, and its epistemic status.

### B.1 Offline: plant → transition file

```
door_sampler.sample_door_population(cfg.doors, seed)         [data_gen.py:148]
  → DoorParams(density, frictionloss, damping, stiffness, springref)
      status: GROUND TRUTH, hidden from every model

door_sampler.build_model(params) → mujoco.MjModel            [data_gen.py:170]

excitation.sample_profile(cfg.excitation, rng, ...)          [data_gen.py:175]
  → TorqueProfile.values (n_ctrl,), ZOH on the frame_skip grid
      status: DESIGNED INPUT

dyn.simulate(profile.as_fn(), model)                         [data_gen.py:126]
  → log{theta, theta_dot, theta_ddot, tau_oracle, tau_ft, ncon}  at 500 Hz
      theta      = data.qpos[hinge]   SIMULATOR GROUND TRUTH
      theta_dot  = data.qvel[hinge]   SIMULATOR GROUND TRUTH
      tau_ft     = (r × F)·â          RECONSTRUCTED from commanded force

data_gen.transitions_from_log(log, frame_skip=10)            [data_gen.py:61]
  j = arange(K-1, n-K, K)                       # one ZOH block per transition
  state      = [theta[j],   theta_dot[j]]          (T,2)  GROUND TRUTH
  next_state = [theta[j+K], theta_dot[j+K]]        (T,2)  GROUND TRUTH
  action     = mean(tau_ft[j+1 : j+1+K])           (T,1)  + ZOH-constancy check
  near_limit = |theta| within margin of the joint stop
      → saved to data/*.npz
```

**No observation model is applied anywhere in this path.** `state` and
`next_state` are simulator truth.

### B.2 Stream construction

```
DoorTransitionDataset(npz, split="heldout_door", exclude_near_limit)  [dataset.py:50]
online.loop.episode_stream(ds, door_id)                              [loop.py:23]
  → list[(s, a, ns)]   plain numpy, episodes concatenated in recording order
      status: GROUND TRUTH throughout
```

Only in `mismatch/` does an observation model appear
([streams.py:110-120](latent_mechanics/mismatch/streams.py#L110)): the *state
sequence* is corrupted once (correctly — corrupting `state` and `next_state`
independently would measure the same instant twice and halve the effective
noise), then transitions are rebuilt from it, and `clean_next` is retained
separately for scoring.

### B.3 The driver

```
online.loop.run_online_adaptation(adaptor, transitions)      [loop.py:146]
  for (s, a, ns) in transitions:
      step = adaptor.observe(s, a, ns)                       # ← everything happens here
  → AdaptationLog(error (T,2), loss (T,), latents (T,d), update_seconds (T,))
```

The driver is a thin, honest prequential loop. **It is the one piece of the
current architecture that survives the refactor unchanged in spirit.**

### B.4 Inside `observe()` — where the abstraction breaks down

```
OnlineLatentAdaptor.observe(s, a, ns)                        [adaptor.py:173]
  ├─ prediction = self.predict(s, a)                         [adaptor.py:160]
  │     = model(s, a, z).numpy()          RAW SI NEXT STATE, belief BEFORE update
  │     → stored in AdaptorStep.prediction  →  used ONLY for scoring/plotting
  │
  └─ loss, extras = self._update(s_t, a_t, ns_t)             [adaptor.py:184]
        → the estimator's own, entirely separate residual computation
```

### B.5 Inside `UKFLatentAdaptor._update` — the real estimation path

```
y = model.target(state, next_state)                          [belief/adaptor.py:134]
  = normalize_delta(ns − s) = (ns − s − delta_mean)/delta_std
      status: DERIVED OBSERVATION, normalised-delta space
      ← this, NOT `prediction`, is the measurement the filter consumes

ukf.predict()                                                [belief/adaptor.py:137]
  fx = identity  ⇒  x unchanged, P ← P + Q

hx_batch(sigmas_r):                                          [belief/adaptor.py:119]
  z_full = basis.decode(sigmas_r)          (2d+1, 16)
  ŷ      = model.raw_output(s, a, z_full)  (2d+1, 2)   NORMALISED DELTA
      ← the predictor is the measurement function

ukf.iterated_update(y, R, hx_batch)                          [ukf.py:186]
  ν = y − (A x_prior + b)                  SLR-linearised innovation
      ← THE RESIDUAL THAT DRIVES THE BELIEF
  x ← x_prior + K ν ;  P ← P_prior − K S Kᵀ

noise.observe(ν, Pzz, K, residual=y_post)                    [belief/noise.py:198]
  R̂ = mean(εεᵀ) + Pzz_post,  floored in Loewner order at IRREDUCIBLE_R

_sync_latent(): z ← basis.decode(ukf.x)                      [belief/adaptor.py:110]
      → the 16-D belief, written back for the base class

return loss = mean(ν²),  extras{innovation_norm, gain_norm, P_trace, ...}
```

Then back in `observe()`:

```
AdaptorStep(prediction   = raw SI next state (pre-update belief),
            target       = ns, raw SI,
            error        = prediction − target,       RAW SI
            loss         = mean(ν²),                  NORMALISED DELTA
            latent       = z after the update)        16-D
```

### B.6 The finding

**There are two different residuals per timestep and they live in different
spaces.** `AdaptorStep.error` (raw SI, one-step-ahead, pre-update) is what every
metric and figure reports. The innovation $\nu$ (normalised delta, and with
`n_iterations=3` an SLR-linearised quantity, not even the sigma-mean residual) is
what actually moves the belief. They are related by an affine map only in the
non-iterated case.

For the gradient adaptor the mismatch is larger still: the update minimises a
loss over the **last 32 transitions**, so the quantity driving the belief at time
$t$ is not a function of the reported step-$t$ error at all. For RLS it is larger
again: the update residual is in **torque** units.

Nothing is *wrong* here — each estimator is internally sound. But the diagram the
user drew,

```
Predicted Observation − Actual Observation → Innovation → Estimator
```

**is not what the code does.** The code does
`observe(s, a, ns) → (opaque internal residual) → belief`, and separately computes
a prediction for reporting. The innovation is not a first-class object, is not
comparable across estimators, and cannot be inspected, logged, or swapped.

---

## C. Architectural problems

Ordered by how much each one obstructs the research.

### C1. `observe()` fuses prediction, innovation and update — **critical**

One abstract method, `_update(state, action, next_state)`
([adaptor.py:166](latent_mechanics/online/adaptor.py#L166)), receives the raw
transition and does everything: form the measurement, form the residual, weight
it, apply it. Consequences:

- The innovation is invisible and incomparable across estimators (§B.6).
- You cannot log or plot "the residual" in a method-agnostic way.
- You cannot reuse one estimator's residual definition with another's update rule.
- `AdaptorStep.loss` is three different quantities.
- Implementing a new estimator means re-deriving the measurement space from
  scratch — as `RLSAdaptor` did, landing in torque space.

### C2. There is no observation model — **critical for the stated next step**

The project's own recommended next step is *close the vision loop*. Right now
that is a rewrite, not a swap:

- Observations are produced by `transitions_from_log` slicing MuJoCo truth
  ([data_gen.py:86-87](latent_mechanics/data_gen.py#L86)). There is no seam.
- `SensorPipeline` is the only observation model and it lives in `mismatch/`,
  applies to a `(T,2)` state array, and is only wired into the door stream
  builder. The mechanism suite has no sensor hook.
- It hardcodes the **door's** joint span for quantisation
  (`JOINT_RANGE = (-0.17, 2.09)`, [sensors.py:21](latent_mechanics/mismatch/sensors.py#L21)),
  so applying it to a drawer measured in metres silently quantises to the wrong
  grid.
- The observation dimension is baked in as `STATE_DIM = 2`
  ([model.py:21](latent_mechanics/model.py#L21)) and reappears as
  `dim_z=self.model.state_dim` in the filter
  ([belief/adaptor.py:101](latent_mechanics/belief/adaptor.py#L101)). Moving to
  force/torque or partial observation touches the model, the filter, the noise
  floor and every metric.
- `IRREDUCIBLE_R` ([noise.py:22](latent_mechanics/belief/noise.py#L22)) is a
  hardcoded 2×2 matrix calibrated for *this* observation space on *one*
  checkpoint. Change the observation and it is silently wrong rather than absent.

### C3. Experiment logic is duplicated five to eight times — **high**

The "build an adaptor by name" factory is reimplemented, with drift, in:

| location | methods built |
|---|---|
| [online/experiments.py:79-95](latent_mechanics/online/experiments.py#L79) | static, gradient, rls |
| [mismatch/study.py:92-108](latent_mechanics/mismatch/study.py#L92) | static, gradient, rls |
| [mechanisms/study.py:84-95](latent_mechanics/mechanisms/study.py#L84) | static, gradient, rls |
| [curriculum/study.py:155-183](latent_mechanics/curriculum/study.py#L155) | static, gradient, rls |
| [belief/sweep.py:205-235](latent_mechanics/belief/sweep.py#L205) | static, gradient, ukf |
| [belief/drift_check.py:125-133](latent_mechanics/belief/drift_check.py#L125) | static, gradient, ukf |
| [belief/ablation.py:107-114](latent_mechanics/belief/ablation.py#L107) | ukf, gradient |
| [geometry/report.py:443](latent_mechanics/geometry/report.py#L443) | gradient |

Note the split: **no single site builds both RLS and the UKF.** The RLS baseline
and the current best learned method have never been run side by side by the same
code path.

Also duplicated: tail-quarter normalised RMSE scoring (four independent
implementations — `mechanisms/study.py:51`, `mismatch/study.py:149-161`,
`curriculum/study.py`, `belief/sweep.py`), and `write_csv` (four copies).

### C4. The environment is not an object; three simulators exist — **high**

| simulator | actuation | used by |
|---|---|---|
| [`dyn.simulate`](baseline/run_door_dynamics_validation.py#L132) | handle force → `mj_applyFT`, action = reconstructed hinge torque | Stage 1, 2 |
| [`simulate_perturbed`](latent_mechanics/mismatch/simulate.py#L48) | same, plus perturbation hooks; docstring says it "mirrors `dyn.simulate` line for line" | Stage 3 |
| [`simulate_mechanism`](latent_mechanics/mechanisms/rollout.py#L44) | direct `qfrc_applied[dof]` | Stage 4, 5, belief |

Two of these are near-copies of each other kept in sync by an assertion
(`verify_matches_baseline`). That assertion is good engineering *around* a
structural problem: the perturbation hook should have been a parameter of one
simulator.

### C5. Everything imports the baseline door *script* as a library — **high**

15 modules do `from baseline import run_door_dynamics_validation as dyn`. Usage
breakdown: `dyn.DT` 41×, `dyn.simulate` 13×, `dyn.T_END`/`dyn.N_STEPS` 20×,
`dyn.rls_init`/`rls_step` 12×, plus geometry helpers.

Worse, episode length is set by **mutating module globals**:

```python
@contextlib.contextmanager
def episode_length(seconds):        # data_gen.py:48
    old_t_end, old_n_steps = dyn.T_END, dyn.N_STEPS
    dyn.T_END = float(seconds); dyn.N_STEPS = int(round(seconds / dyn.DT))
    ...
```

because `simulate()` reads `N_STEPS` from its module namespace rather than taking
it as an argument. This is global mutable state in the data-generation path, and
it makes `simulate` unsafe to call concurrently.

`dyn.DT = 0.002` is a physics constant that 41 call sites reach into a *baseline
experiment script* to obtain.

### C6. The predictor is a concrete class, not an interface — **high**

`MechanicsDynamicsModel` is the type annotation throughout. `OnlineLatentAdaptor`
depends on `model.raw_output`, `model.target`, `model.freeze`, `model.embed_dim`,
`model.state_dim`, `model.action_dim`, and implicitly on the normalisation
buffers. To test **analytical dynamics** or an **intentionally misspecified
predictor** — both named as goals in the brief — you must subclass `nn.Module`
and fake the normalisation buffers so `raw_output`/`target` round-trip. That is a
real obstacle to the first experiment the user says they want.

### C7. The belief representation is hardwired to a 16-D dense vector — **medium**

`OnlineLatentAdaptor._z` is `nn.Parameter(embed_dim)`
([adaptor.py:140](latent_mechanics/online/adaptor.py#L140)). Explicit physical
parameters only work by escaping the hierarchy entirely (`RLSAdaptor` subclasses
`OnlineAdaptor`, not `OnlineLatentAdaptor`) and then lying in the `.latent`
property. A hybrid representation has no home at all.

### C8. Config is fragmented across three systems plus undeclared dataclasses — **medium**

`latent_mechanics/config.py` (`ExperimentConfig`), `online/config.py`
(`OnlineConfig`), `mismatch/config.py` (`StudyConfig`) — each with its own
near-identical `_build` / `config_from_dict` / `load_config` triple. `UKFConfig`
lives in [belief/adaptor.py:28](latent_mechanics/belief/adaptor.py#L28) and is in
no YAML schema at all; UKF settings are passed as Python kwargs from sweep
scripts.

`mismatch/study.py` loads **all three** config systems
([lines 87-88](latent_mechanics/mismatch/study.py#L87)) plus a checkpoint that
carries a fourth serialised copy of `ExperimentConfig`. There is no single artifact
that says what an experiment did.

### C9. Hidden assumptions in unrelated modules — **medium**

- `data_gen.JOINT_RANGE` (the door's) is the **default argument** of
  `transitions_from_log`. Audit item A2 fixed the mechanism path to pass its own
  range, but [`mismatch/streams._near_limit`](latent_mechanics/mismatch/streams.py#L63)
  still uses the door constant unconditionally.
- `sensors.JOINT_SPAN` — door-specific quantiser (C2).
- `noise.IRREDUCIBLE_R` — checkpoint- and observation-space-specific constant (C2).
- `basis.DEFAULT_TABLE = "runs/latent_mechanics/geometry/runs/all_families/best.pt"` —
  a default that points into a *results* directory, three stages deep.
- `model.STATE_DIM = 2`, `ACTION_DIM = 1` as module constants.
- `rls_adaptor.DEFAULT_I/MU/B` — door priors, used for every mechanism family.

### C10. Dead or low-value abstractions

Verified by grep, not assumed:

| item | finding |
|---|---|
| `NormStats` ([model.py:33](latent_mechanics/model.py#L33)) | a `dict` subclass whose `validate()` is **never called** anywhere; only `NormStats.KEYS` is used. The class earns nothing. |
| `OnlineAdaptor.reset(*args, **kwargs)` | abstract with an untyped signature that differs by subclass; **never called by the driver**, only from `__init__`. |
| `episode_boundaries` | computed and threaded through 8 modules; consumed at exactly one place, [viz.py:76-77](latent_mechanics/online/viz.py#L76), to draw vertical lines. |
| `AdaptationLog.extras` | silently **drops** any non-scalar extra ([loop.py:168-173](latent_mechanics/online/loop.py#L168)) — a filter that hides data rather than failing. |
| `InnovationAdaptiveNoise` | documented in its own docstring as legacy and inferior ("can go indefinite, 30-41% of steps"); still selectable. |
| `UKFConfig.floor` | "legacy innovation model only". |
| `UKFConfig.smoothing`, `warmup` | "unswept" starting points presented as configuration. |
| `mechanisms/time_resolution.py` | `FRAME_SKIP_OVERRIDES` empty by default — importing it changes nothing (open item B3). |
| `RLSAdaptor.latent` | a name that means something different from every other use of the word. |
| `_match_batch` ([model.py:186](latent_mechanics/model.py#L186)) | silently broadcasts; a genuinely mismatched batch passes quietly. |

### C11. What is genuinely good and must be preserved

An audit that only lists problems is not useful. These are load-bearing and
correct:

- **The frozen-network enforcement** (checksum + grad-leak check). Rare, and it
  caught real bugs.
- **The prequential protocol** — predict before update, never shuffle, never
  revisit. This is the only protocol under which the estimators compare fairly.
- **The no-adaptation control.** It is what revealed Stage 5's negative result.
  Any refactor that makes it optional is a regression.
- **Scoring against clean truth while feeding corrupted observations**
  ([streams.py:138-146](latent_mechanics/mismatch/streams.py#L138)) — measures the
  estimator, not the sensor. The `clean_error = error + (observed − clean)`
  identity is exact and elegant.
- **Corrupting the state *sequence*, not transitions independently.**
- **Normalising error by true motion scale.** Raw RMSE is meaningless across
  families and would let a perturbation that slows the door look like an
  improvement.
- **ZOH verification** in `transitions_from_log` — a real invariant, checked.
- **Provenance hashing** (`provenance.py`) — every checkpoint load records stage +
  sha256, pinnable.
- **`verify_matches_baseline`** — bit-exact agreement between the two door
  simulators.
- **The UKF reference test** — 1e-10 against `filterpy` on every intermediate.
- **`MerweSigmaPoints` / `psd_floor` / `nearest_pd`** — careful numerics with the
  reasoning recorded.

---

## D. Module responsibilities as they stand

| module | responsibility | verdict |
|---|---|---|
| `baseline/run_door_dynamics_validation.py` | door sim + RLS + Stage-A experiments, all in one 609-line script | **split**: sim, RLS and experiments are three things |
| `latent_mechanics/config.py` | Stage-1 config | keep, extend |
| `door_sampler.py` | sample + build door MuJoCo models | keep, fold into plant layer |
| `excitation.py` | ZOH torque profiles | keep as-is |
| `data_gen.py` | population rollout → npz; `transitions_from_log` | **split**: slicing is a core primitive, the CLI is not |
| `dataset.py` | npz → transitions/episodes | keep |
| `model.py` | predictor + embedding table + checkpoint IO | **keep, add an interface above it** |
| `train.py`, `evaluate.py` | Stage-1 training and eval | keep, out of the online path |
| `rollout.py` | multi-step rollout metrics | keep |
| `online/loop.py` | prequential driver + `AdaptationLog` | **keep — this is the core** |
| `online/adaptor.py` | estimator interface + gradient rule | **split**: interface, prediction and update rule are conflated |
| `online/rls_adaptor.py` | RLS baseline | **preserve behaviour exactly** |
| `online/experiments.py` | Stage-2 driver | becomes an experiment spec |
| `belief/ukf.py` | standalone UKF | keep — clean and tested |
| `belief/adaptor.py` | UKF ↔ latent wiring | rewrite against new interfaces |
| `belief/basis.py` | frozen PCA chart | keep — this is a *mechanics representation*, promote it |
| `belief/noise.py` | adaptive R/Q | keep; drop the legacy innovation form |
| `mismatch/sensors.py` | **the only observation model** | **promote out of `mismatch/`** |
| `mismatch/perturbations.py` | plant perturbations | promote to the plant layer |
| `mismatch/simulate.py` | door sim + hooks | merge with `dyn.simulate` |
| `mismatch/streams.py` | stream builder + clean scoring | **promote — the scoring identity is core** |
| `mechanisms/library.py` | six families | keep — this is the plant catalogue |
| `mechanisms/rollout.py` | generic simulator | **promote to the one simulator** |
| `*/study.py`, `belief/sweep.py` | five experiment drivers | **collapse into experiment specs** |
| `geometry/*` | latent-structure analysis | keep, analysis-only |
| `provenance.py` | checkpoint hashing | keep |
| `tools/live_viewer.py` | live MuJoCo viewer | keep; retarget at the new interfaces |

---

## E. Known limitations, restated honestly

Carried from `docs/PROJECT_STATUS.md` and confirmed against the code:

1. **Single seed throughout.** Nothing has been replicated.
2. **B1 is unresolved and it matters.** The sensing-crossover claim (Stage 3) was
   measured on the doors-only predictor; geometry and UKF results use the
   all-families predictor. Nine distinct frozen predictors exist across the stages.
3. **B2**: the gradient-descent baseline the UKF is compared against runs
   hyperparameters tuned in Stage 2 on a *different* predictor.
4. **Simulation only.** No perception, no hardware.
5. **`nonlinear_hinge` is poorly predicted by everything** (~4.8× ceiling) — a
   predictor limit, not a filter limit.
6. **Partial observability unsolved** (bifold); needs an architecture change.
7. **Scale and category are confounded** in the mechanism suite.
8. **Latent ratios below 1.0 are a scoring artefact**, not superhuman performance.

### One thing this audit adds to that list

**The documented UKF results were produced by code that exists in no commit.**

The working tree carries 55 modified tracked files and 4 untracked ones. From the
diffstat this looks like a docstring-tightening pass, and a docstring-tightening
pass *is* mixed into it — but interleaved in the same files is the entire
**residual-adaptive-R and iterated-update (IPLF) feature set**, none of which is
in `HEAD` (`5973806`):

| symbol | HEAD | working tree |
|---|---|---|
| `ResidualAdaptiveNoise` | absent | present |
| `psd_floor` (Loewner-order matrix floor) | absent | present |
| `IRREDUCIBLE_R` | absent | present |
| `UnscentedKalmanFilter.iterated_update` | absent | present |
| `UKFConfig.noise_kind` default | `"adaptive"` | `"residual"` |
| `UKFConfig.n_iterations` | field absent | `3` |

Also uncommitted: the tests for it (`test_ukf_reference.py` gains 7 `psd_floor`
references, 4 `iterated_update`, 2 `ResidualAdaptiveNoise`), and four untracked
support modules — `belief/calibrate_noise.py`, `belief/ablation.py`,
`belief/figures.py`, `geometry/online_latents.py`. `calibrate_noise.py` is
referenced four times by the tracked `belief/README.md` and by a comment in
`noise.py` that tells you to run it to regenerate `IRREDUCIBLE_R`.

Consequences:

- `docs/PROJECT_STATUS.md` says "Working tree clean" and "Last verified against
  `main` @ `956ddd9`". Both are stale; `HEAD` is `5973806` with 55 dirty files.
- The headline UKF numbers (0.87× ceiling, "the credit belongs to adaptive R")
  describe `noise_kind="residual"` behaviour that `HEAD` cannot produce.
- **If this working tree were lost, those results would be unreproducible.**

This audit, and the refactor built on it, target the **working tree**, since that
is the live code. Committing it is urgent; committing it as one lump is not
advisable — see [ARCHITECTURE.md §E](ARCHITECTURE.md).

---

## F. Summary judgement

The research is in better shape than the architecture. The experimental protocol
is careful — prequential scoring, real controls, clean-truth scoring under
corrupted observations, enforced frozen networks, bit-exact cross-checks between
simulators, a reference-validated filter. Several results are genuinely
established and several negative results are reported honestly rather than buried.

The architecture, though, has grown by **cloning the previous stage**. Each new
question produced a new package containing its own adaptor factory, its own
scoring function, its own CSV writer, and its own config system. The estimator
interface conflates prediction with update, so the innovation — the central object
in the research question — does not exist as a thing in the code. The observation
model exists in exactly one package and is hardcoded to the door. The predictor is
a concrete class.

Consequently the four experiment classes the brief names — swap the predictor,
swap the mechanics representation, swap the estimator, swap the observation — are
each currently a code-writing task rather than a configuration task. That is the
gap the refactor should close, and it can be closed without touching the
algorithms.
