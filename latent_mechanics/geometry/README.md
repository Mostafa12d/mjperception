# What structure does the learned latent mechanics space have?

*An empirical investigation to decide which family of online belief inference is
justified by the representation — before designing one.*

```bash
python3.10 -m latent_mechanics.geometry.report
```

Read-only with respect to all existing model weights. The one artefact created is
an additional checkpoint trained on all six families (Stages 1–5 never produced
one), using the unchanged Stage-1 pipeline.

---

## Bottom line

**An IMM over mechanism categories is not supported by this representation.**

The latent does contain real cluster structure — that part of the IMM
motivation survives. But the clusters are **bins along a continuous mechanical
scale axis**, not distinct mechanism types, and they dissolve when that axis is
removed. Two independent tests:

| test | result |
|---|---|
| ARI(6 clusters, true **family**) | **+0.398** |
| ARI(6 clusters, **inertia sextile**) | **+0.398** |
| excess silhouette over matched null, raw `z` | +0.256 |
| excess silhouette over matched null, **log-inertia regressed out** | **+0.055** |

Produced by **Step 3b** of the pipeline (`analyses.scale_dominance`, printed by
`report.py` and written to `geometry_report.json` under `scale_dominance`, figure
`scale_dominance.png`). Regression-tested in `geometry.tests`.

The clusters predict "how heavy is it" exactly as well as they predict "what kind
of thing is it", and 78% of the apparent structure is the scale axis alone.
Mechanical scale is continuous in the world, so discretising it is a modelling
choice, not a property of the data — and one that gets worse as object diversity
grows.

**What the evidence does support: a continuous, locally-Gaussian belief on a
reduced-dimension latent, with an inflated and preferably self-estimated
measurement-noise model.** Details and the case against each alternative are in
§7.

---

## 1. Summary of the latent representation

| property | finding |
|---|---|
| dimensionality | **16** (`ModelConfig.embed_dim`) |
| storage | `DoorEmbeddingTable`, an `nn.Embedding(num_objects, 16)` |
| persistence | checkpoint key `embedding_state`, separate from `model_state` |
| per object | **exactly one row per training instance**; no per-timestep history |
| deterministic or stochastic | **deterministic point estimate** — no variance, no distribution; the representation currently carries no notion of its own uncertainty |
| how it enters the network | plain concatenation: `net([norm(s), norm(a), z])`. No gating, no attention, no multiplicative interaction |
| objects available | 17 checkpoints, 20–120 objects each; new all-family table has **120** |
| categories | door, nonlinear\_hinge, soft\_close, drawer, laptop, bifold (+ `door_narrow`, a training-only narrow-range door) |

Two consequences worth stating up front. First, because `z` is a bare point with
no covariance, *every* candidate filter below is a strict addition to the
representation, not a modification of it. Second, because `z` is concatenated
rather than gated, the network has no architectural bias toward discrete modes —
whatever cluster structure exists was not designed in.

**Effective dimensionality is 2.36 of 16.** PC1 carries 63% of the variance, and
five components reach 90%. The 16-dimensional latent is using roughly two and a
half directions. Any filter maintaining a full 16×16 covariance would be
estimating ~136 parameters to describe a ~2.4-dimensional object.

---

## 2. Latent geometry

![geometry](../../runs/latent_mechanics/geometry/geometry_projections.png)

PCA, UMAP and t-SNE agree, which matters — the structure is not a projection
artefact:

- **laptop** forms a cleanly separated island in all three (95% of its nearest
  neighbours are laptops)
- **drawer** is partially separated (90% purity)
- **door, nonlinear\_hinge, soft\_close, bifold** form one heavily overlapping
  continuum (door purity **25%** — its neighbours are mostly *other families*)

Median nearest-neighbour distance is **1.08**; median pairwise distance **3.65**.

Note what predicts separation: laptop median log₁₀(inertia) = **−2.16**, drawer
**+1.55**, and the four overlapping families sit at **+0.22 to +1.10**. Families
separate exactly insofar as their mechanical scale differs.

---

## 3. Evidence for and against multimodality

![multimodality](../../runs/latent_mechanics/geometry/multimodality.png)

With 120 points in 16 dimensions, a full-covariance component costs 152
parameters and is not identifiable; the sweep uses diagonal covariance. **Every
statistic is compared against a matched unimodal null** — synthetic Gaussians
with the same mean, covariance, N and d — because model selection reliably
invents clusters in this regime.

| statistic | real latents | matched unimodal null | verdict |
|---|---|---|---|
| BIC-optimal K | 7 | **4.0** (mean) | BIC is unreliable here; the null also "finds" clusters |
| held-out log-lik optimal K | 4 | **4.5** (mean) | **no evidence beyond the null** |
| best silhouette | **0.572** | 0.315 mean, 0.342 max | **real structure**, p < 0.001 |
| ARI vs true families (K=6) | +0.398 | — | moderate; clusters ≠ categories |

So the two rigorous criteria disagree, and the disagreement is informative:

- **Cross-validated likelihood — the criterion that makes no large-sample
  assumption — cannot distinguish the latents from a single Gaussian.** K=4 is
  chosen for the real data and K=4.5 for the null.
- **Silhouette says the structure is real** (p < 0.001), so there is genuine
  non-Gaussian geometry.

Both can be true: the cloud has real anisotropic, partly-separated structure that
is nevertheless poorly described as a mixture of well-separated Gaussian modes.
That is exactly what the projections show — two outliers and a continuum.

And the structure is scale, not category (table in the summary above). Removing a
single scalar — log inertia — takes the excess silhouette from +0.254 to +0.055.

---

## 4. Continuous or discrete?

![continuity](../../runs/latent_mechanics/geometry/continuity.png)

Interpolating `z` between object pairs and scoring on both objects' real data:

| pair type | median barrier ratio | pairs with a barrier > 1.2× |
|---|---|---|
| within family | **1.01** | 17% |
| across families | **1.16** | 50% |

A barrier ratio near 1.0 means every interpolated latent is a valid mechanics
hypothesis for *something* — the path between two objects passes through
usable territory. Within a family that is essentially always true. Across
families, half of the pairs show a modest barrier, and those are dominated by
paths crossing the scale gap into the laptop island.

**Reading: locally continuous everywhere, with mild non-convexity between widely
separated scale regimes.** There are no hard walls. The largest observed barrier
is a factor of ~1.2, not the orders of magnitude a genuinely disjoint mode
structure would produce.

---

## 5. Local linearity

![linearity](../../runs/latent_mechanics/geometry/linearity_attribution.png)

First-order approximation `f(z₀+δ) ≈ f(z₀) + J δ`, relative error:

| ‖δz‖ | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| relative error | **1.1%** | **1.9%** | 5.9% | 13% | 41% |

Jacobian: ‖∂f/∂z‖_F median 0.010, p05→p95 spread **110×**, condition number
median 12.6 (p95 406), relative variation across operating points 2.13.

Two conclusions pull in opposite directions and both matter:

**Linearisation is excellent at update scale.** A Stage-2 online step moves the
belief by ~10⁻³–10⁻², where the first-order model is accurate to well under 1%.
An EKF-style update is entirely defensible.

**Linearisation fails at inter-object scale.** Nearest-neighbour spacing is 1.08,
where the error is 41%. And the fitted oracle latents sit **3.2–6.6** away from
the prior. A single linearisation cannot carry the belief from its prior to the
right object — that needs iteration (IEKF), sigma points (UKF), or many small
steps. The Jacobian's 110× spread and condition numbers reaching 406 also mean a
fixed process-noise scale will be badly wrong somewhere.

---

## 6. Where does the error actually come from?

Fitting an **oracle latent** offline on all of an unseen object's data gives the
best any belief update could ever achieve. The gap between prior error and oracle
error is what inference can win; the oracle error itself is model error that no
filter can touch.

| family | prior | online-adapted | oracle (ceiling) | removable by `z` |
|---|---|---|---|---|
| drawer | 4.06e+0 | 2.13e-1 | 1.64e-1 | **96%** |
| nonlinear\_hinge | 3.83e-2 | 5.21e-2 | 1.50e-2 | 57% |
| soft\_close | 4.40e-2 | 3.04e-2 | 1.88e-2 | 57% |
| laptop | 1.05e-1 | 1.21e-1 | 6.15e-2 | 39% |
| door | 2.46e-2 | 2.40e-2 | 1.67e-2 | 28% |
| bifold | 2.25e-2 | 1.90e-2 | 2.01e-2 | **25%** |

**Median across objects: only 47% of the prior error is removable by any choice
of `z`.**

This is the single most consequential number for filter design. Slightly over
half of the residual is irreducible — network approximation error, unmodelled
physics (the bifold's hidden second link), and under-sampling (the laptop). For
a Bayesian filter this is *measurement noise that is large, state-dependent, and
not zero-mean*. A filter with an optimistic fixed `R` will become
overconfident and stop listening to data. Whatever is built must either estimate
its own measurement noise online or be robust to a heavy-tailed residual.

Note also that the online adaptor gets nowhere near the ceiling, and on two
families is *worse than the prior* — consistent with Stage 5.

---

## 7. Candidate belief representations

| method | core assumption | supported here? | cost | robustness | scaling with more object types |
|---|---|---|---|---|---|
| **Single Gaussian (point + covariance)** | unimodal, roughly elliptical belief | **Yes.** Held-out likelihood cannot beat a single Gaussian; effective dim 2.4 | trivial | good if `R` inflated | good — no structural change needed |
| **EKF** | + local linearity of `f` in `z` | **Yes at update scale** (1–2% error); no at inter-object scale | O(d²), d=16 → cheap | fragile to the 110× Jacobian spread; needs adaptive process noise | good |
| **Iterated EKF** | + relinearise per update | **Better fit than EKF** — directly addresses the 3.2–6.6 prior-to-oracle distance | a few× EKF | noticeably better on large corrections | good |
| **UKF** | unimodal; no Jacobian needed | **Yes**, and avoids the ill-conditioned Jacobian entirely | 2d+1 = 33 forward passes | best of the Gaussian family here | good |
| **IMM** | belief is a mixture of **K discrete, persistent, well-separated modes** | **No.** Clusters = scale bins, not categories (ARI equal for both); structure vanishes without the scale axis; held-out likelihood no better than unimodal | K× a base filter | mode-mismatch risk; needs a transition matrix with no physical meaning here | **worsens** — see §8 |
| **Particle filter** | arbitrary belief shape | Over-powered for a 2.4-effective-dim unimodal cloud; would help only for the mild non-convexity | ~10²–10³ forwards/step | very robust; handles the heavy-tailed residual naturally | poor in raw 16-D, fine on a reduced latent |
| **Continuous Bayesian latent inference** | continuous posterior, explicit likelihood | **Well matched** to the geometry; the 53% irreducible error is exactly what a proper likelihood should model | depends on parameterisation | good | good |
| **Learned recurrent belief update** | amortised inference; enough training tasks to learn it | Attractive but **unsupported today**: 120 objects is far too few, and Stage 5 showed generalisation is already data-starved | cheap at test, expensive to train | unknown | best in the long run, if data grows |
| **Neural Bayesian / variational filtering** | learned posterior family, calibrated uncertainty | Same as above, plus it could *learn* the state-dependent noise floor — the right long-term answer to §6 | high training cost | potentially best calibration | best, but needs far more objects |

---

## 8. Scalability: would K discrete modes get better or worse?

Suppose the dataset grows to scissors, staplers, pliers, cabinets, refrigerators,
microwaves, oven doors, folding tables, grippers, articulated toys, deformable
hinges.

**Discrete modes get *less* appropriate, for three reasons this data already
shows.**

**The axis that separates modes is continuous.** The only strong separation here
is mechanical scale, spanning log₁₀(inertia) from −2.2 to +1.6. Scissors and
staplers land between laptop and door; refrigerator doors and folding tables land
above drawers. New objects **fill the gaps** rather than adding new islands. The
laptop looks discrete only because nothing in this suite occupies the four orders
of magnitude beneath a door.

**K would have to grow with the object inventory.** An IMM's cost is K filters
plus a K×K transition matrix, and neither the mode count nor the transitions have
a physical interpretation once "mode" means "inertia decile". With 11 more
categories you would be choosing K by cross-validation on a quantity the data
says is continuous.

**Category is the wrong grouping anyway.** Door and soft-close doors share a
category boundary but overlap almost completely (door 1-NN purity 25%), while a
heavy drawer and a heavy door are neighbours. Any mode structure worth having
would be over *mechanical regime*, and regimes are ordered and continuous.

The one caveat: a genuinely discrete **structural** variable might appear later —
number of DOF, presence of contact, deformable vs rigid. The bifold hints at it
(hidden second link) but does not separate in the latent (80% purity, and it sits
mid-cloud). If deformable hinges or multi-contact mechanisms produce a
*qualitatively* different observation model rather than a different parameter
value, a small discrete belief over *model class* — not object category — becomes
defensible. That is a hypothesis to test when such objects exist, not one this
data supports.

---

## 9. Recommendation

**Use a continuous, locally-Gaussian filter on a reduced-dimension latent. Do not
use IMM over mechanism categories.**

Concretely, justified point by point by the evidence above:

1. **Reduce the working dimension.** Effective dim is 2.36 of 16; filter in a
   4–6 dimensional PCA subspace of the trained embedding table. This makes the
   covariance identifiable and cuts UKF sigma points from 33 to ~11.
2. **Prefer a UKF or an iterated EKF over a plain EKF.** The Jacobian's 110×
   spread and condition numbers to 406 make a single linearisation risky, and the
   prior-to-oracle distance (3.2–6.6) is far outside the radius where one
   linearisation is valid (~0.25).
3. **Inflate and adapt the measurement noise.** Only 47% of the error is
   explainable by `z`; treat the rest as a state-dependent noise floor and
   estimate it online. This is also the principled version of the "when not to
   adapt" gate that Stages 4 and 5 both identified as missing — a filter with an
   honest `R` naturally stops updating when residuals are dominated by model
   error.
4. **If you want discrete structure, make it 2–3 coarse *scale regimes*, not six
   categories** — and treat it as a hierarchical prior (discrete over regime,
   continuous within), not an IMM over object types. The evidence for even this
   is modest, and §8 argues it will weaken as the inventory grows.
5. **Revisit learned/variational inference when the object count is an order of
   magnitude larger.** It is the best long-term answer to the calibration problem
   in §6, and is simply not trainable at 120 objects.

### Why this is worth publishing

The interesting result is not the recommendation but the method: the same
representation yields *opposite* answers under BIC (K=7) and cross-validated
likelihood (no better than unimodal), and only the matched null exposes why. The
"clusters" that a naive analysis would use to justify an IMM turn out to be a
single continuous scalar — and one ablation, regressing out log-inertia, is
enough to show it. That is a transferable recipe for motivating an inference
model from measured latent geometry rather than intuition.

---

## Limitations

- 120 objects in 16 dimensions is a small sample for mixture selection; this is
  precisely why the null baseline carries the argument rather than BIC.
- One trained model, one seed. Latent geometry could vary across initialisations;
  the family-purity and scale findings should be replicated over seeds.
- Six families spanning a scale range chosen by me. The "gaps fill in" prediction
  in §8 is an extrapolation, testable by adding intermediate-scale mechanisms.
- Oracle latents are fitted in full batch; the *reachable* optimum for a causal
  online filter may be worse.
- The laptop island is partly an artefact of sampling rate (Stage 4: its time
  constant is below the 20 ms step), so some of the observed discreteness is a
  benchmark property, not an object property.
