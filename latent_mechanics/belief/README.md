# UKF belief update over a reduced latent subspace

Branch `belief-update/ukf-reduced`. A drop-in replacement for the
gradient-descent belief update, motivated by the measured latent geometry rather
than by preference — see [../geometry/README.md](../geometry/README.md).

```bash
python3.10 -m latent_mechanics.belief.test_ukf_reference   # Step-3 checkpoint
python3.10 -m latent_mechanics.belief.calibrate_noise      # measure R's floor
python3.10 -m latent_mechanics.belief.ablation             # the two defects + fixes
python3.10 -m latent_mechanics.belief.sweep --objects 60   # d and noise sweep
```

## What is here

| file | role |
|---|---|
| `basis.py` | Step 2. Frozen affine map `z = z_mean + V_d z_r`, persisted artifact |
| `ukf.py` | Step 3. Generic UKF over `(fx, hx)`; knows nothing about this project. Standard and iterated (IPLF) updates |
| `noise.py` | Step 4. `FixedNoise` / `InnovationAdaptiveNoise` / `ResidualAdaptiveNoise` |
| `adaptor.py` | Step 5. `UKFLatentAdaptor`, same interface as `GradientLatentAdaptor` |
| `calibrate_noise.py` | Measures the predictor's irreducible residual — the floor on `R` |
| `ablation.py` | The two measurement-path defects, one change at a time |
| `sweep.py` | d = 4,5,6 × noise model, scored against per-object oracle ceilings |
| `test_ukf_reference.py` | filterpy validation, IPLF correctness, noise-model checks |

## Two defects, found by asking a question prediction error cannot answer

Every UKF number originally reported here was measured with `alpha=0.3` and the
innovation-form adaptive `R`. Both were wrong, and **prediction error was nearly
blind to it** — it moves by ~10% across settings that swing the latent's physical
decodability from `R² = −0.11` to `R² = +0.42`. The sweep, which scored only
prediction, therefore selected settings that were quietly destroying the latent.
What exposed it was checking whether the *estimated latent still decodes the
physical parameters*.

**1. The unscented transform was invalid over the prior.** The prior is the whole
training cloud (per-axis sd up to 2.71); the predictor is locally linear in `z`
only out to `|dz| ≈ 0.25`. Sigma points landed 2.6× (median) to 8.0× (max) beyond
that. Against 20k-sample Monte Carlo the transform mispredicted the measurement
by **2.62** in normalised units where the true residual at the filter's own mean
was **1.31** — the innovation was 2:1 transform error over signal. `alpha=0.3,
kappa=0` compounded it: at `d=6` it gives `Wm[0] = −10.11`, `Wc[0] = −7.20`, so
the transform's mean and covariance are differences of large numbers. `alpha` is
not a free "keep the points close" knob; with `kappa=0` it sets the weights too.

*Fix:* non-negative weights (`alpha=1.0`) plus an **iterated posterior
linearisation** update (IPLF, García-Fernández et al. 2015). Each iteration
re-draws the sigma points around the current *posterior* and redoes the update
*from the prior* with that better linearisation. Shrinking `P0` would also work
and is deliberately **not** used — it buys accuracy by declaring less prior
uncertainty than we actually have, which defeats the purpose of filtering.

**2. Adaptive `R` collapsed onto the uninformative channel.** `R̂ = C_ν − Pzz` is
a *difference*, and it went indefinite on **30–41%** of steps, clipped to a scalar
`1e-6` floor each time. That floor sits **452× below** the smallest eigenvalue of
the predictor's measured irreducible residual, and being scalar it cannot express
the 0.77 correlation between the two channels. The collapsed direction was
`d_theta` on **all six families** — the channel **4–13× less sensitive to `z`**,
because it is essentially kinematic (`Δθ ≈ dt·ω`). The filter placed near-infinite
confidence in the measurement carrying almost no mechanics; gains ran to **146**.

*Fix:* the residual form `R̂ = (1/N) Σ εεᵀ + Pzz_post`, a **sum** of PSD terms, so
it cannot go negative and never needs rescuing. Floored at the measured matrix
`IRREDUCIBLE_R = [[1.13e-3, 1.06e-2], [1.06e-2, 1.68e-1]]` (120 objects, 119,667
transitions; regenerate with `calibrate_noise.py`) in the Loewner order. The floor
is a modelling statement, not a numerical guard: the filter may never believe the
predictor is a better sensor than it provably is.

## Correctness

Validated against `filterpy.kalman.UnscentedKalmanFilter` + `MerweScaledSigmaPoints`
(1.4.5) to **1e-10 on every intermediate quantity** — sigma points, weights,
prior mean/covariance, innovation, innovation covariance, Kalman gain, posterior
— over 20 steps of a nonlinear system, at three `(n, α, β, κ)` settings.

One substantive finding from that validation. filterpy reuses the
`fx`-propagated sigma points for the measurement update instead of redrawing
from the prior, so those points carry covariance `F P Fᵀ` while `P_prior` is
`F P Fᵀ + Q`. The gain is computed as if there were no process noise. **On a
linear system the filterpy convention therefore does not reproduce the exact
Kalman filter unless `Q = 0`; regenerating the sigma points does, to 1e-12.**
Both are implemented and both behaviours are asserted in the tests.
`regenerate_sigma_points` defaults to `False` in the core (reference parity) and
`True` in `UKFConfig` (correctness for our identity-`fx`, `Q > 0` setting).

The iterated update has no filterpy counterpart, so it is pinned by two
properties instead. On an **affine** `h` the statistical linear regression is
exact, so every iteration produces the same linearisation and the result must
equal the textbook Kalman update for *any* iteration count — asserted at 1, 2 and
5 iterations to 1e-9. If iterating changed the answer there, the measurement
would be getting folded in more than once, which is the standard way to build an
overconfident iterated filter. On a nonlinear `h` the tests assert that iterating
*reduces* the post-update residual while **not** shrinking `P` below the one-shot
bound.

## Design decisions and their evidence

**Reduce the dimension.** Effective latent dimensionality is 2.36 of 16, so a
full 16×16 covariance would estimate 136 parameters for a ~2.4-dimensional
object and need 33 sigma points per update.

**Filter in normalised measurement space.** `h(z)` is `model.raw_output` (the
normalised state delta), not the raw next state. Raw units are radians for a
door and metres for a drawer; a shared `R` in raw units cannot be right for both.

**`fx` is the identity.** Object mechanics do not evolve, so prediction is just
`P ← P + Q`. `Q` is not a physical process noise but an explicit statement of how
willing the filter is to revise a settled belief — the direct analogue of the
gradient adaptor's learning rate.

**Adaptive `R` is the point, not an extra.** The geometry report found only ~47%
of prediction error is removable by any `z`; the rest is model error. In filter
terms that residual *is* measurement noise, it differs per object, and it cannot
be set a priori. Letting the filter measure its own noise floor is also the
principled form of the "when not to adapt" gate that Stages 4 and 5 both
identified as missing: when residuals are dominated by model error, `R` grows,
the gain shrinks, and the belief stops chasing noise without needing a heuristic.
This argument was right and the *implementation* of it was the bug — see the two
defects above. The same 47% figure is what `calibrate_noise.py` now measures from
the other side, as a covariance, and uses as `R`'s floor.

**One batched forward pass per iteration.** All 2d+1 sigma points go through the
predictor together. With IPLF this is one batched call per iteration; `iter_tol`
exits early, so the mean is **2.41 of 3** iterations, and an update costs ~610 µs
against ~350 µs for a gradient step. The filter is no longer cheaper than the
module it replaces — that is the honest price of a valid transform.

## Results (60 unseen objects, six families)

Scored as **ratio to the per-family oracle ceiling** from the geometry report.

| config | ×ceiling | ≤2× | NaN | resets | µs/update |
|---|---|---|---|---|---|
| no adaptation | 2.14× | 43% | 0 | 0 | 50 |
| gradient descent (current module) | 1.72× | 55% | 0 | 0 | 261 |
| **UKF d=4, adaptive R** | 1.17× | 67% | 0 | 0 | 224 |
| **UKF d=5, adaptive R** | 1.12× | 68% | 0 | 0 | 232 |
| **UKF d=6, adaptive R** | **0.97×** | 68% | 0 | 0 | 239 |
| UKF d=4, fixed R | 1.92× | 50% | 0 | 0 | 168 |
| UKF d=5, fixed R | 1.80× | 53% | 0 | 0 | 175 |
| UKF d=6, fixed R | 1.92× | 53% | 0 | 0 | 190 |
| d=5, window 20 | 1.51× | 55% | — | — | 229 |
| d=5, window 50 | 1.12× | 68% | — | — | 247 |
| d=5, window 100 | 0.98× | 72% | — | — | 268 |

Per-family ratio to ceiling:

| config | door | nonlin | soft-close | drawer | bifold | laptop |
|---|---|---|---|---|---|---|
| no adaptation | 1.90 | 5.25 | 2.11 | 11.81 | 0.95 | 1.68 |
| gradient descent | 3.16 | 6.89 | 1.61 | 1.30 | 0.95 | 2.03 |
| UKF d=6 adaptive | 1.01 | 5.09 | 0.65 | 0.22 | 0.97 | 1.45 |

> **Caveat on ratios below 1.** The ceilings are per-family *medians* measured
> over each object's full stream; the numbers here are per-object and scored on
> the final quarter. A ratio under 1 means an easier-than-median object on an
> easier-than-average window, not performance beyond the optimal latent.

Adaptive R is doing the heavy lifting — fixed R is roughly at the
gradient-descent level while adaptive R is 1.6–1.9× better than it. Zero NaNs and
zero filter resets across every configuration and all 60 objects; the covariance
trace settles two orders of magnitude below the fixed-R case, i.e. the filter
becomes confident rather than drifting.

`nonlinear_hinge` remains poor for every method (≈5× ceiling), consistent with
earlier stages.

## Not decided here

d, the adaptive-R window and floor, whether Q should adapt, and whether an
Iterated EKF is worth building are all flagged in the brief as user decisions.
Numbers are above; no choice has been baked in beyond the dataclass defaults.
