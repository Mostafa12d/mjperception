# Stage 3 — Robustness to model mismatch

**Research question.** Under what conditions does latent online adaptation become
preferable to explicit parameter estimation?

Stage 2 established that on an ideal plant RLS wins decisively, because the
simulated door is *exactly* linear in its parameters and RLS's model is
therefore literally correct. This stage introduces progressively more realistic
violations of that assumption — one mechanism at a time — and asks where the
ranking changes.

Nothing in Stages 1 or 2 is modified. The learned dynamics network is **not
retrained** on the perturbed plants and RLS **keeps its regressor**. Both methods
hold their original assumptions while the world stops obeying them, which is the
only way the comparison isolates robustness rather than re-fitting.

```bash
python3.10 -m latent_mechanics.mismatch.tests    # 46 self-checks
python3.10 -m latent_mechanics.mismatch.study    # ~6 min, 8 sweeps
python3.10 -m latent_mechanics.mismatch.study --only stribeck,drift
```

---

## Answer

**Sensing quality flips the ranking; physics mismatch does not.**

![overview](../../runs/latent_mechanics/base/mismatch/overview.png)

Across every *physical* violation tested — Stribeck friction, position-dependent
friction, nonlinear compliance, drifting parameters — RLS degrades faster than
the latent (5–23× versus 2–4×) but never loses, because it starts 16× ahead and
has margin to spare. Across every *sensing* violation except latency, RLS
collapses and the latent overtakes it.

| Experiment | mismatch | latent degrades | RLS degrades | latent wins from |
|---|---|---|---|---|
| 1 | encoder noise | 41× | **1873×** | σ = 1e-4 rad |
| 1 | quantisation | 5.4× | **464×** | 14 bits |
| 1 | dropped samples | **1.2×** | 59× | p = 0.15 |
| 1 | sensor latency | 1.8× | 4.7× | never |
| 2 | Stribeck friction | 4.0× | 13× | never |
| 2 | position-dependent friction | 2.0× | 23× | never (within 1.4×) |
| 3 | nonlinear compliance | 2.1× | 5.4× | never |
| 4 | parameter drift | 3.2× | 11× | never |

**The mechanism is specific and predictable.** RLS's regressor needs
acceleration, which is not measurable, so it must form `θ̈ = Δθ̇/dt`. When
velocity itself comes from differencing a noisy encoder, position noise is
amplified by `2/dt² = 5000×` before it reaches the regressor. Worse, that noise
lands on the *regressor* rather than the target, which is the errors-in-variables
case: least squares is then not merely noisy but **biased**, attenuating `Î`
toward zero systematically. The learned model consumes `(θ, θ̇)` directly and
never differentiates anything, so it inherits no amplification.

That also explains the one exception. Latency delays the state but leaves it
*self-consistent* — the pair `(s_{t-k}, s_{t+1-k})` is a real transition, only the
action is misaligned — so no differentiation blow-up occurs and RLS keeps its
lead.

Dropped samples are the cleanest win: the latent is essentially immune (1.2×)
while RLS degrades 59×. A held reading produces a "nothing moved" transition,
which for the network is merely an unusual input diluted across its 32-sample
window, but for RLS is a `θ̈ = 0` row paired with a large torque — a direct,
unbounded corruption of the parameter estimate.

**Practical reading.** A 14-bit encoder is a *good* encoder (1.4e-4 rad over this
joint's travel), and that is already past the crossover. So the regime where
latent adaptation is preferable is not exotic: it is ordinary hardware. The
regime where explicit estimation wins is one with clean, high-rate, directly
measured state — which is what a simulator provides and a robot usually does not.

---

## The measurement protocol, and a trap in it

Two metrics are recorded, and the difference between them matters more than
anything else in this stage.

`angle_nrmse_final` — prequential error on the corrupted stream. What the robot
experiences moment to moment.

`holdout_nrmse` — the belief the estimator *finished with*, frozen and evaluated
on clean held-out episodes from the same plant. Whether the corrupted stream
poisoned what was learned. **This is the primary metric.**

The first metric alone is confounded, and badly. A stale or dropped reading makes
the instantaneous prediction wrong no matter how good the model is: you are asked
to predict forward from a state that is already out of date. That costs every
method the *same* bookkeeping offset — measured at ≈ k× the per-step motion for
latency and ≈ √p for dropout. The first run of this study duly reported latency
errors of 0.999 / 1.994 / 3.978 for k = 1 / 2 / 4, identical across all three
methods to within 0.2%, and a "crossover" that was really two failed methods
tying. Scoring the frozen belief on clean data removes the offset and asks the
question we actually care about.

A useful sanity check falls out of this: under sensor corruption the
no-adaptation control is *exactly flat* (1.0× degradation at every level), as it
must be, since a belief that never updates cannot be corrupted by bad data.

Two further protocol decisions:

**Errors are normalised by the true motion.** Absolute RMSE is confounded because
a perturbation that slows the door (more friction, say) shrinks every error and
masquerades as an improvement — an early run reported `no-adaptation` "improving"
0.7× under Stribeck friction for exactly this reason. Normalised error is the
fraction of actual motion left unexplained; 1.0 means no better than predicting
that nothing changes.

**Excitation is seeded per door and episode, never per severity level**, so the
identical torque profile is replayed at every level and a change in error cannot
come from a different trajectory.

---

## Design

### Everything is modular and config-driven

| file | role |
|---|---|
| `perturbations.py` | plant physics: `StribeckFriction`, `PositionDependentFriction`, `NonlinearCompliance`, `ParameterDrift` |
| `sensors.py` | `SensorPipeline`: noise → quantisation → dropout → latency |
| `simulate.py` | the perturbed integrator, plus `verify_matches_baseline` |
| `streams.py` | per-door interaction streams, hold-out streams, scoring helpers |
| `config.py` | `Sweep` definitions — one mechanism, one parameter, four severities |
| `study.py` | the driver |
| `figures.py` | figures and the LaTeX table |
| `tests.py` | 46 self-checks |

Adding a mismatch mechanism means writing a `PlantPerturbation` subclass and one
`Sweep` entry. The simulator asks whatever perturbations it was handed for their
contribution each step and hard-codes none of them; an empty list reproduces the
ideal plant exactly.

### The simulator duplication, and why it is safe

Stage 1 rolls out episodes with `dyn.simulate`, whose torque callback receives
only the time. Stage-3 physics is state-dependent, so it cannot go through that
callback and needs its own loop — the one genuine duplication in this stage.

Because a silent divergence there would be indistinguishable from a real effect,
`verify_matches_baseline` asserts that with no perturbations the Stage-3 loop
reproduces `dyn.simulate` **exactly**: max deviation `0.0` on θ, θ̇, θ̈, τ_oracle
and τ_ft, not merely within tolerance. It runs in the test suite and is the first
thing to check if a result looks surprising.

### Two invariants worth stating

**The recorded action never contains the perturbation.** The robot commands a
hinge torque and records that; the unmodelled physics is added to the joint
afterwards. A test asserts `tau_ft` is bit-identical with and without a
perturbation active.

**Sensor corruption is applied to the state sequence, not to transitions.**
Consecutive transitions share a state — the `next_state` of one is the `state` of
the next — so perturbing the two fields independently would give the robot two
different readings of the same instant and halve the effective noise through
averaging. A test asserts every shared state is measured exactly once.

### Severity calibration

Plant levels were chosen by measuring the RMS unmodelled torque each setting
produces against the 5.96 N·m RMS commanded torque of the held-out population,
giving roughly 0%, 3%, 10% and 30% unmodelled torque so the plant sweeps are
comparable to one another.

---

## Belief analysis

*Does it still converge?* Yes, under every plant perturbation and under mild
sensing corruption. Panel (b) of each `sweep_*.png` shows convergence curves at
the worst severity; `belief_*.png` shows the trajectory in the fixed Stage-1 PCA
frame, so it is directly comparable to the Stage-1 latent-space figure and the
Stage-2 trajectory.

*Does it oscillate, and how sensitive is it to noise?* Quantified by
`belief_drift_tail` — mean per-step belief motion over the final quarter,
normalised by belief magnitude, and defined identically for both methods so they
can be compared. Under the worst encoder noise the RLS parameter vector jitters
at 1.5e-2 while the latent sits at 1.3e-3, an **11× more stable belief**. That is
the same errors-in-variables story seen from the parameter side: RLS's estimate
is not just wrong under noise, it is restless.

The latent's own drift is largest not under noise but under position-dependent
friction (8.0e-3), which makes sense — that perturbation is genuinely
state-dependent, so the locally-best latent really does change as the door
sweeps, and the belief chases it.

---

## Honest limitations

- **The crossovers come from RLS falling, not the latent improving.** On a clean
  plant the latent sits at 1.5e-2 normalised error against RLS's 9.3e-4. The
  learned model's floor is its own approximation error, unchanged since Stage 1.
- **No retraining.** Whether a model trained *on* Stribeck friction could capture
  it is a different and also interesting question; this stage deliberately does
  not ask it.
- Torque sensing is left ideal, so actuation noise is an untested axis.
- Perturbations are evaluated individually by design. Combining them is the
  obvious next step and may not be additive.
- Unseen doors remain interpolation: same geometry, parameters inside the
  training ranges.
- Joint-limit transitions are still excluded, matching Stage 1.
