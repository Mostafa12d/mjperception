# Stage 2 — Online latent adaptation

A robot meets a door it has never seen and improves its mechanics belief while
interacting with it. The Stage-1 dynamics network is **completely frozen**; only
the latent changes, one transition at a time.

```
z_0 --observe--> z_1 --observe--> z_2 --> ... --> z_T
      ^                                             |
      |          frozen dynamics network            |
      +----- predict, compare, revise belief -------+
```

Stage 1 is imported, never modified.

```bash
python3.10 -m latent_mechanics.online.tests                                       # 37 self-checks
python3.10 -m latent_mechanics.online.experiments --config configs/online_adaptation.yaml
```

Full suite takes ~66 s on CPU and writes to `runs/latent_mechanics/base/online/`.

---

## Headline results

Eight unseen doors (never trained on, no embedding row), ~2 300 interactions each
at 50 Hz. Everything is measured **prequentially**: each prediction is made with
the belief held *before* that transition was seen, so no estimator is ever scored
on data it has already fitted.

**Experiment 1 — adaptation works.** Median **5.9× lower** angle prediction error
than the no-adaptation control, ranging 3.1× to 13.0× across doors. Two doors end
up *better* than the Stage-1 reference for a door the network was actually
trained on (2.4e-05 and 2.2e-05 rad vs 3.98e-05).

**Experiment 2 — initialisation.** All four strategies converge to roughly the
same place; they differ in how fast they get there.

| init | start RMSE | final RMSE | steps to converge |
|---|---|---|---|
| zero | 1.29e-04 | 6.01e-05 | 267 |
| random trained | 8.06e-05 | 5.89e-05 | 260 |
| mean | 8.54e-05 | 5.93e-05 | 206 |
| **medoid** | **3.84e-05** | **5.59e-05** | **0** |

The medoid — the training door closest to the centre of the latent cloud — starts
already converged. Since the latents lie on a shell whose centre contains no
door, "average" in the geometric sense is not a plausible door, but the medoid is
a real one. Use it.

**Experiment 3 — versus RLS. The baseline wins, decisively, and it matters.**

| method | angle RMSE | velocity RMSE | steps to own conv. | µs/update |
|---|---|---|---|---|
| no adaptation | 3.18e-04 | 5.85e-03 | — | 54 |
| latent-gd | 6.01e-05 | 9.75e-04 | 215 | 318 |
| **rls-5p** (spring-aware) | **2.52e-06** | **1.67e-04** | 365 | **8.2** |
| rls-3p (baseline regressor) | 2.48e-05 | 2.23e-03 | 130 | 7.8 |

RLS with a correctly-specified regressor is **24× more accurate** on angle, 5.8×
on velocity, and **39× cheaper per update**.

This is not a bug and not an unfair setup — it is the correct answer for this
plant. The simulated door is *exactly* linear in its parameters,
`tau = I*thdd + mu*sign(thd) + b*thd + k*th + c`, so RLS identifies the true
system and then integrates it with the same substeps MuJoCo uses. Verified
directly: it recovers inertia to four significant figures and stiffness exactly.

| door | Î | I true | k̂ | k true |
|---|---|---|---|---|
| 48 | 18.43 | 18.42 | 6.37 | 6.37 |
| 52 | 23.03 | 23.03 | 6.91 | 6.91 |

The learned model cannot compete on that terrain, and the reason is structural:
its floor is its **own approximation error**, about 4e-05 rad even on doors it
was trained on. No latent, however well optimised, gets below the network's own
accuracy. Online latent adaptation recovers essentially all of the headroom that
*is* available to it (3.18e-04 → 6.01e-05, within ~1.5× of the trained-door
reference) — the remaining gap belongs to the network, not the adaptor.

**Where the learned model does win.** Against `rls-3p`, the baseline's own
regressor, which has no spring term and so is misspecified for the 70% of doors
that have a door-closer: velocity RMSE 2.23e-03 for RLS versus 9.75e-04 for the
latent — **2.3× better**. That is the honest scope of the claim. A latent
mechanics model earns its keep when you do not know the parametric form; when you
do know it, write it down and use RLS.

The obvious next test is therefore a regime where no tidy regressor exists —
joint-limit contact, velocity-dependent or non-Coulomb friction, a door whose
handle geometry is unknown, or real hardware.

---

## Design

### The interface, in three layers

Gradient descent is the first update rule, not an assumption baked into the
design. `adaptor.py` separates the contract from the algorithm:

**`OnlineAdaptor`** — the generic streaming-estimator contract: `predict`,
`observe`, `reset`, `belief`. It mentions neither latents nor gradients nor
neural networks. Both the latent adaptor and RLS implement it, which is exactly
what makes Experiment 3 fair: one driver, one protocol, identical data.

**`OnlineLatentAdaptor`** — adds "the belief is a latent fed to a frozen
network". Implements prediction, latent bookkeeping, reset and the frozen-network
guarantees **once**, and leaves a single abstract method, `_update`.

**`GradientLatentAdaptor`** — the first `_update`: backprop the prediction loss
into `z` alone.

**`StaticLatentAdaptor`** — never updates. The control condition, and the
smallest possible example of the `_update` contract.

### Where future update rules plug in

Subclass `OnlineLatentAdaptor`, implement `_update`, change nothing else:

- **Kalman / EKF** — treat `z` as Gaussian. Linearise the frozen network around
  the current `z` (`torch.autograd.functional.jacobian` of the prediction w.r.t.
  `z` — cheap, only `2 × embed_dim`), apply the standard measurement update, and
  return the covariance from `belief()`, which already has the slot for it.
- **Learned updater** — `_update` calls a trained network
  `(z, state, action, error) -> dz`.
- **Bayesian / particle filter** — keep a particle set; `latent` returns the
  posterior mean, `belief()` the full set.

`tests.py` includes a random-walk adaptor that is not gradient-based, purely to
prove the driver never assumed gradients.

### The frozen-network guarantee

This is Stage 2's defining constraint, and a violation would be easy to miss —
the model would just get better and the experiments would look great. It is
enforced in the base class, not left to each subclass, and checked three ways:

1. `_assert_frozen()` at construction — no parameter may have `requires_grad`.
2. The optimiser is constructed over **exactly one tensor**, the latent. There is
   no code path from it to a network weight.
3. `assert_network_unchanged()` compares a checksum over every weight against the
   value captured before adaptation, and fails if any gradient accumulated. It
   runs after **every** experiment run, not only in tests.

`tests.py` also verifies the guard actually fires when a weight is tampered with —
a check that always passes is worthless.

### Two decisions that changed the results

**A decaying step size is required, not a nicety.** With a constant learning
rate, adaptation starting from an already-good initialisation ended up *worse
than not adapting at all* (0.95×): the noise the updates injected exceeded the
information they extracted. `lr_t = lr / (1 + lr_decay·t)` — the Robbins–Monro
condition — fixed it, and adaptation now improves on the control from every
initialisation tested. Default `lr_decay = 3e-3`.

**A bounded sliding window, not single samples.** `window=1` is pure
single-sample online SGD, and at 50 Hz it is too noisy: at `lr ≥ 0.03` the belief
*degraded* over the stream (0.2–0.6× "gain"). A window of the 32 most recent
transitions is still fully online — bounded memory, constant cost per update, no
trajectory ever revisited — and is stable. This was measured, not assumed.

Both are configurable; `window=1` reproduces textbook online SGD.

### Fairness of the RLS comparison

Three choices, all of which favour RLS:

1. **A spring-aware 5-parameter regressor**, because the baseline's own
   3-parameter form is structurally misspecified for spring-loaded doors. Both
   are reported.
2. **Velocity gating** — updates are skipped below `|thd| = 0.02`, since at rest
   the equation of motion is an inequality and those rows are invalid. The
   baseline gates identically in `moving_mask`. The latent adaptor gets no
   equivalent help; it updates on every transition.
3. **Sub-stepped integration** at MuJoCo's own 0.002 s, so RLS is not charged for
   discretisation error the learned model never pays.

Stated rather than hidden: acceleration is not observable, so the regressor uses
a finite difference of observed velocities — honest for a 50 Hz stream, but
noisier than the exact `qacc` the offline Stage-1 script reads from MuJoCo. Both
methods see the same 50 Hz transitions in the same order; matching by interaction
*time* rather than sample count is what "same trajectory" means for a robot.

### The prequential protocol

`predict` is always called before `observe` folds the transition in, so every
reported error is a genuine one-step-ahead prediction. Without this, an estimator
could score itself on data it had just fitted. It is also what a robot actually
experiences: you must act before you learn from the outcome.

---

## Files

| file | role |
|---|---|
| `adaptor.py` | `OnlineAdaptor` / `OnlineLatentAdaptor` / `GradientLatentAdaptor` / `StaticLatentAdaptor` |
| `rls_adaptor.py` | `RLSAdaptor` — wraps the untouched `dyn.rls_step` in the same interface |
| `loop.py` | driver, `AdaptationLog`, transition streams, init strategies |
| `viz.py` | error curves, latent-trajectory figures, animation |
| `experiments.py` | the three experiments + visualisation, CLI |
| `config.py` | dataclass config, YAML overrides |
| `tests.py` | 37 self-checks |

Outputs in `runs/latent_mechanics/base/online/`: `exp1_error_curve.png`,
`exp2_init_comparison.png`, `exp3_method_comparison.png`,
`belief_trajectory.png` / `.mp4`, `belief_snapshots.png`, per-experiment CSVs,
`belief_trajectory.npz`, `summary.json`.

---

## The belief visualisation

PCA is fitted **once, on the training embeddings**, so the axes mean the same
thing as in the Stage-1 latent-space figure and stay fixed across every frame —
refitting per frame would make the motion meaningless.

For unseen door 48 (I=18.4, µ=1.74, b=0.56, k=6.37) starting from `z=0`, the
belief starts at the origin — a point in the middle of the latent shell where no
training door lives and the network was never evaluated — travels into the
high-inertia region, and settles next to training doors 25 (I=19.1, k=7.67) and
45 (I=18.5, k=6.51). It found mechanically similar doors without ever being told
a physical parameter. The step size decays from ~1e-1 to ~1e-3 as the belief
settles.

---

## Limitations

- The comparison plant is exactly linear-in-parameters, which is RLS's best case
  and the learned model's worst. Conclusions about relative merit do not transfer
  to regimes with unmodelled physics — testing those is the point of the next
  experiment, not a caveat to be waved away.
- Unseen doors are interpolation: same `door.xml` geometry, parameters inside the
  training ranges. Genuine extrapolation is untested.
- Joint-limit transitions are excluded throughout, matching Stage-1 training.
- Episodes are concatenated into one stream per door, so the door jumps back to
  closed at each boundary — visible as transients in the error curves.
- The gradient adaptor is ~318 µs/update on CPU, comfortably real-time at 50 Hz
  (20 ms budget), but 39× RLS's cost.
