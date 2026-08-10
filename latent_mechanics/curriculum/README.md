# Stage 5 — Mechanics prior scaling

**Research question.** Does online mechanics adaptation reduce the amount of
offline mechanical diversity required for generalisation?

**Short answer: no, not within the diversity this benchmark contains.** Diversity
clearly buys a better *prior* and makes adaptation steadily *less harmful*, but
online adaptation never becomes net-beneficial at any of the seven levels.
Extrapolating the trend puts break-even at roughly **9–10 mechanism families** —
50–70% more mechanical variety than the six-family suite provides.

```bash
python3.10 -m latent_mechanics.curriculum.tests   # 47 self-checks
python3.10 -m latent_mechanics.curriculum.study   # ~6 min, 7 trainings
```

---

## The scaling curves

![scaling](../../runs/latent_mechanics/curriculum/scaling_curves.png)

Fixed budget of **48 training instances at every level**; fixed evaluation suite
of **60 unseen instances** across six families, identical for all seven models.
Errors are normalised by true motion (1.0 = no better than predicting no change).

| level | families | before | after | gain | failures | 
|---|---|---|---|---|---|
| L1 narrow doors | 1 | 3.18e-2 | 1.30e-1 | 0.38× | 88% |
| L2 wide doors | 1 | 3.40e-2 | 7.80e-2 | 0.55× | 77% |
| L3 + nonlinear | 2 | 2.44e-2 | 5.94e-2 | 0.72× | 75% |
| L4 + drawer | 3 | 2.44e-2 | 4.75e-2 | 0.61× | 77% |
| L5 + soft-close | 4 | 2.27e-2 | 4.40e-2 | 0.69× | 68% |
| L6 + bifold | 5 | 2.39e-2 | 5.39e-2 | 0.60× | 72% |
| L7 + laptop | 6 | 2.19e-2 | 4.82e-2 | **0.85×** | **57%** |
| *RLS on the same suite* | — | — | *6.1e-3* | — | — |

**What improves.** The zero-shot prior gets better with diversity: 3.18e-2 →
2.19e-2, a 1.45× reduction with no extra data. Adaptation becomes much less
destructive: 1.30e-1 → 4.82e-2, 2.7×. Harmful adaptations fall from 88% to 57%,
a trend that is statistically real (**−4.2 percentage points per family,
p = 0.014**).

**What does not.** Gain stays below 1.0 at every level. Adaptation is a net loss
on the median test instance throughout. The upward trend (+0.053 per family) has
p = 0.076 — suggestive, not significant on seven points.

### The quantitative answer

Extrapolating the two fitted trends:

| criterion | families required |
|---|---|
| harmful adaptations below 50% | **8.7** |
| median gain reaches break-even (1.0×) | **10.1** |

So: **roughly nine to ten mechanism families at this training budget**, against
the six available. This is a linear extrapolation from seven points and should be
read as an order-of-magnitude statement, not a precise threshold — but the
direction and rough scale are consistent across both metrics.

---

## Why adaptation fails: the prior-quality relationship

The clearest signal in the stage is not on the diversity axis at all. Pooling all
**420** instance-level results:

**Spearman(prior error, adaptation gain) = +0.334.**

Adaptation helps in proportion to how badly the prior was already doing:

| prior error quartile | median gain | harmful |
|---|---|---|
| 0.006 – 0.017 (best prior) | 0.39× | 86% |
| 0.017 – 0.025 | 0.57× | 77% |
| 0.025 – 0.066 | 0.66× | 66% |
| 0.066 – 0.451 (worst prior) | **0.83×** | 65% |

When the prior already fits, optimising the latent almost always makes things
worse — the gradient is chasing noise, and the belief wanders off the manifold
the network understands. The single best result in the whole stage is L7's
drawer: prior error 1.94e-1, gain **4.75×**, **0% harmful**. That is a badly
mismatched prior with real signal to exploit.

This reconciles Stage 5 with Stage 4, which reported gains of 1.3–11.6×. Stage 4
held out an *entire family* and evaluated only on that family — always the
badly-mismatched regime. Stage 5 evaluates on all six families including ones in
training, where the prior is already good and there is nothing to gain. Stage 4
also trained on 100 instances against Stage 5's 48. Neither result is wrong; they
measure different regimes, and Stage 5's is the more representative one.

The practical implication is the same one Stage 4 flagged: the adaptor has no
notion of when *not* to adapt. A gate on prior residual, or an uncertainty-scaled
step size, is the missing ingredient — Stage 2's unused `belief()` covariance
slot is where it belongs. Out of scope here by instruction.

---

## Representation analysis

![latents](../../runs/latent_mechanics/curriculum/latent_evolution.png)

| level | families | log-inertia R² | friction R² (within family) | geometry ρ | d_eff |
|---|---|---|---|---|---|
| L1 | 1 | 0.956 | **0.879** | 0.306 | 4.37 |
| L2 | 1 | 0.977 | 0.800 | 0.579 | 3.42 |
| L3 | 2 | 0.922 | 0.571 | 0.586 | 3.92 |
| L4 | 3 | 0.847 | 0.719 | 0.582 | 3.33 |
| L5 | 4 | 0.916 | 0.252 | 0.648 | 3.24 |
| L6 | 5 | 0.725 | 0.160 | 0.523 | 3.40 |
| L7 | 6 | 0.910 | **0.106** | **0.685** | 3.06 |

**There is a real trade-off here, and the fixed budget is what exposes it.**

*Global* geometry improves with diversity: the correlation between latent
distance and mechanics distance rises from 0.31 to 0.69. The space genuinely
becomes better organised by mechanics — objects that behave alike end up close
together, which is exactly the property an online optimiser needs in order to
travel usefully.

*Local* resolution collapses: within-family friction readout falls from 0.88 to
0.11. At a fixed 48-instance budget, going from one family to six cuts per-family
coverage from 48 instances to 8, and the latent can no longer resolve friction
differences *within* a family.

So diversity makes the prior broader and smoother while making it blurrier. That
is a coherent explanation for the headline result: the broader prior is why
zero-shot error improves and harmful adaptation declines, and the blurrier prior
is why adaptation still cannot find a better latent than the one it starts from.
Whether the two can be had at once is a budget question this stage cannot answer,
because holding the budget fixed is precisely what makes it a diversity study.

Effective dimensionality drifts down (4.4 → 3.1) — the latent is not collapsing,
but it is not expanding to accommodate new mechanisms either. With 16 dimensions
available and only ~3 used, capacity is not the binding constraint.

---

## Failures, documented

**1. Adaptation is net-harmful at every curriculum level.** The headline negative
result. 57–88% of test instances end worse than if the latent had been left alone.

**2. Within-family precision is sacrificed for cross-family breadth.** Friction
readout 0.88 → 0.11. At a fixed budget these appear to be in direct competition.

**3. The convergence metric is uninformative here and is reported as such.**
"Steps to reach 1.5× the run's own final error" returned 0 at every level: when
adaptation makes things worse, the final error is large and the rolling curve
starts below the threshold. The metric assumes convergence to something better,
which mostly did not happen. It is left in the CSVs but is not plotted.

**4. The laptop remains unfixable, as in Stage 4.** Prior error ~0.1 and gain
0.84× at L7, even with laptops in the training mixture. Its mechanical time
constant is shorter than the 20 ms sampling interval, so no amount of training
diversity helps — this is a sampling-rate problem, not a prior problem.

**5. RLS still wins by ~4× on the same fixed suite** (6.1e-3 versus L7's best
2.19e-2 before adaptation). Consistent with Stages 2–4: these plants remain
largely linear-in-parameters.

**6. A leak that nearly invalidated the stage.** `build_dataset_npz` originally
partitioned train/held-out *by family name*. Levels 2–7 train on doors while the
fixed suite also contains doors, so the evaluation doors were being promoted into
training — the "unseen" suite would have been partly memorised. Caught by a test
asserting that a held-out instance of a *trained* family stays held out. The
packer now takes an explicit `heldout_pops` argument, and the first full run was
discarded and repeated. Any earlier numbers from this stage should be ignored.

---

## Design

**Fixed budget is the whole point.** Every level trains on 48 instances × 5
episodes, so total transitions stay near-constant (58k at L7) and movement along
the x-axis is diversity alone. Without this, "more diversity helps" would be
indistinguishable from "more data helps". A test asserts the budget is identical
across levels.

**Fixed evaluation suite.** 10 unseen instances of each of the six real
mechanisms, generated once from a dedicated seed (999, far from the training seed
100), cached, and inserted as the held-out split of every level's dataset. Tests
assert it is deterministic, complete, and never leaks into training.

**Levels 1 and 2 vary parameter spread, not mechanism** — `door_narrow` has
collapsed friction/inertia/stiffness ranges — so the curriculum separates
parameter diversity from mechanism diversity before adding any new mechanism.

**Shared instance pools.** An instance used at several levels is the *same*
instance with the same trajectories, so between-level differences are not
sampling noise.

**Nothing frozen was touched.** Architecture (16-d latent, 2×256 MLP, 71 426
weights), optimiser, adaptation algorithm and RLS are all imported unchanged; a
test asserts the parameter count and the `door` family spec.

---

## Limitations

- Seven points, one seed per level. The failure-rate trend is significant
  (p = 0.014); the gain trend is not (p = 0.076). Multiple seeds would tighten
  both, and the break-even extrapolation especially.
- Break-even at ~9–10 families is a linear extrapolation well beyond the data.
- Only one budget (48 instances) was tested, so the diversity/precision trade-off
  is characterised at a single point on the budget axis. Repeating the curriculum
  at 96 and 192 instances would separate "diversity needs more data" from
  "diversity has a ceiling", and is the obvious next experiment.
- Mechanism families are a coarse diversity axis; within-family parameter spread
  is a second axis that only L1→L2 probes.
