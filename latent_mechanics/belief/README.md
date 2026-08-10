# UKF belief update over a reduced latent subspace

Branch `belief-update/ukf-reduced`. A drop-in replacement for the
gradient-descent belief update, motivated by the measured latent geometry rather
than by preference — see [../geometry/README.md](../geometry/README.md).

```bash
python3.10 -m latent_mechanics.belief.test_ukf_reference   # Step-3 checkpoint
python3.10 -m latent_mechanics.belief.sweep --objects 60   # d and noise sweep
```

## What is here

| file | role |
|---|---|
| `basis.py` | Step 2. Frozen affine map `z = z_mean + V_d z_r`, persisted artifact |
| `ukf.py` | Step 3. Generic UKF over `(fx, hx)`; knows nothing about this project |
| `noise.py` | Step 4. `FixedNoise` / `InnovationAdaptiveNoise` (Mehra / RAUKF) |
| `adaptor.py` | Step 5. `UKFLatentAdaptor`, same interface as `GradientLatentAdaptor` |
| `sweep.py` | d = 4,5,6 × adaptive/fixed R, scored against the oracle ceiling |
| `test_ukf_reference.py` | filterpy validation |

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

**One batched forward pass per update.** All 2d+1 sigma points go through the
predictor together, so a UKF step costs about the same as a gradient step.

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
