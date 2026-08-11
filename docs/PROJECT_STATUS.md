# Project status — online latent mechanics adaptation

*Last verified against `main` @ `956ddd9`. All seven test suites passing.*

---

## One-paragraph summary

The project learns a latent "mechanics belief" `z` for articulated objects and
adapts it online against a frozen dynamics predictor, benchmarked throughout
against the pre-existing RLS system-identification baseline. Five evaluation
stages are complete, plus a latent-geometry investigation and a UKF belief-update
branch that is now merged. **The core method works and, with the UKF, beats both
the previous gradient-descent belief update and the no-adaptation control on
every mechanism family tested.** RLS still wins on clean, well-modelled plants;
the learned method's structural advantage is degraded sensing. A foundation
audit has since found and fixed five correctness issues and flagged two more
that are measured but not yet applied.

---

## Repository state

| branch | head | status |
|---|---|---|
| `main` | `956ddd9` live viewer using UKF | current |
| `fixes/foundation-audit` | `ccf5852` | **fully merged into main** |
| `tools/live-viewer` | `9d614c3` | merged |
| `belief-update/ukf-reduced` | `bbe72e4` | merged |

Working tree clean. ~13k lines across eight modules under `latent_mechanics/`,
plus `baseline/`, `iiwa/`, `perception/`, `scenes/` and `tools/`.

The repository was reorganised into packages on 2026-08-11: the loose scripts
that sat in the root now live under `baseline/`, `iiwa/` and `perception/`,
scene XMLs under `scenes/`, and everything is invoked as
`python3.10 -m <package>.<module>` from the repo root. No behaviour changed and
all seven suites still pass; see [RUNNING.md](RUNNING.md) section 0.

| module | role |
|---|---|
| `baseline/` | line A — the RLS baseline; `run_door_dynamics_validation.py` is the shared simulator |
| `iiwa/` | line B — KUKA iiwa 14 with FK + wrist F/T sensing |
| `perception/` | line D — RGB-D, FlowBot3D, OMIP |
| `scenes/` | MuJoCo XMLs and the `scene_path()` resolver |
| `latent_mechanics/` | Stage 1 — model, embedding table, dataset, training |
| `online/` | Stage 2 — adaptor interface, gradient adaptor, RLS adaptor |
| `mismatch/` | Stage 3 — perturbations, sensor pipeline, robustness study |
| `mechanisms/` | Stage 4 — six mechanism families, cross-family transfer |
| `curriculum/` | Stage 5 — diversity scaling |
| `geometry/` | latent-structure investigation |
| `belief/` | UKF belief update (reduced basis, adaptive noise) |
| `provenance.py` | checkpoint hashing / stage labelling (audit A5) |
| `tools/live_viewer.py` | read-only MuJoCo viewer with live belief update |

---

## What each stage established

### Stage 1 — the representation works
One MLP `(state, action, z) → next_state`, one learned embedding per object.
Validation angle RMSE **3.98e-05 rad**, i.e. 0.6% of actual motion. The latent is
load-bearing: a wrong embedding costs **3.8×** the error, a zero embedding 8.7×.
A leave-one-out probe recovers friction (R²=0.90), inertia (0.83) and stiffness
(0.80) from `z` alone — parameters never given as input. Damping (0.02) is not
encoded, consistent across every later stage.

### Stage 2 — online adaptation works; RLS still wins on ideal plants
Median **5.9×** improvement over a no-adaptation control, with the network
provably frozen. But RLS is **24× more accurate and 39× cheaper**, because the
simulated door is exactly linear-in-parameters and RLS recovers the true values
to four significant figures. Two findings only visible because of the control:
constant-step-size adaptation from a good init is a *net loss* (fixed by
Robbins–Monro decay), and pure single-sample SGD degrades the belief.

### Stage 3 — the crossover is sensing, not physics
Under every *physical* mismatch (Stribeck, position-dependent friction,
nonlinear compliance, drift) RLS degrades faster than the latent but never
loses. Under *sensing* degradation it collapses:

| mismatch | latent degrades | RLS degrades |
|---|---|---|
| encoder noise | 41× | **1873×** |
| quantisation | 5.4× | **464×** |
| dropped samples | **1.2×** | 59× |
| sensor latency | 1.8× | 4.7× (RLS keeps the lead) |

Cause is structural: RLS must form `θ̈ = Δθ̇/dt`, so noise lands in its regressor
and biases the estimate; the learned model never differentiates. The crossover is
at **14-bit encoder resolution** — ordinary hardware.

### Stage 4 — it is a mechanics code, if trained on diversity
Six families behind one interaction interface. Trained on doors only, adaptation
*hurt* on 4 of 5 unseen families; trained on five families it helped on 5 of 6
(1.3–11.6×). The latent's dominant axis is mechanical scale (log-inertia
R²=0.92). Documented failures: laptops fail for everyone including RLS (time
constant is 0.19–0.9× the sampling interval — a sampling failure, not a
representation one), and partial observability (bifold) is unsolved.

### Stage 5 — the negative result
At a **fixed training budget**, adaptation is net-harmful at every diversity
level (gain 0.38× → 0.85×, harmful cases 88% → 57%). Break-even extrapolates to
~9–10 families. Root cause measured: adaptation helps only when the prior is
already wrong (Spearman +0.334), and the gradient adaptor cannot tell the
difference. A real trade-off surfaced: global latent geometry improves with
diversity (ρ 0.31 → 0.69) while within-family precision collapses (friction R²
0.88 → 0.11).

### Geometry investigation — what structure the latent has
Effective dimensionality **2.36 of 16**. Against a matched unimodal null, BIC
selects K=7 but the null also selects K=4, and cross-validated likelihood cannot
distinguish the latents from a single Gaussian. Silhouette *is* significant
(0.572 vs null max 0.342, p<0.001) — but the structure is **scale, not
category**: `ARI(clusters, family)` and `ARI(clusters, inertia-sextile)` are both
+0.398, and residualising log-inertia collapses the excess silhouette from
+0.256 to +0.055. Conclusion: **an IMM over mechanism categories is not
supported**; a continuous, locally-Gaussian filter is.

### UKF belief update — merged, and it works
Validated against `filterpy` 1.4.5 to **1e-10 on every intermediate quantity**.
Re-measured on current `main`, 60 unseen objects, ratio to per-family oracle
ceiling:

| method | ×ceiling | within 2× | µs/update |
|---|---|---|---|
| no adaptation | 2.14× | 43% | 50 |
| gradient descent (replaced) | 1.72× | 55% | 271 |
| **UKF d=6, adaptive R** | **0.87×** | **72%** | 249 |
| UKF, fixed R | 1.92× | 53% | 177 |

It beats no-adaptation on **all six families**, where gradient descent was worse
than not adapting on three — so it fixes Stage 5's central negative result. Under
friction drift it is the only method that never becomes harmful. **The credit
belongs to adaptive R, not the UKF machinery**: fixed-R UKF performs at the
gradient-descent level.

Settings locked: `d=6`, `window=100`, `adapt_Q=off`,
`regenerate_sigma_points=True`, all-families basis.

---

## Foundation audit

Five issues found and fixed (A1–A5), two measured but **not applied** (B1–B3).

| id | issue | status |
|---|---|---|
| A1 | curriculum population draws used `hash(str)`, which Python salts per process — a seed did not reproduce a population | fixed via sha256-derived seeds; cached pools unaffected |
| A2 | `transitions_from_log` flagged near-limit against the *door's* range for every family; flagged 0% of a drawer episode actually against its stop | fixed at source, per-mechanism range; **no data changed** (callers had recomputed correctly) |
| A3 | "% moving" used an absolute 0.02 threshold on both m/s and rad/s | fixed, unit-invariant; diagnostics only |
| A4 | the geometry README's headline scale numbers came from an ad-hoc session script, not reproducible from the pipeline | committed as `scale_dominance`; corrected +0.254 → +0.256 |
| A5 | no record of which frozen predictor produced which result | `provenance.py`; every load logs stage + sha256 |

**A4 also strengthened the geometry conclusion.** Residualising against *observed*
scale (log RMS step size and action — what a filter can actually measure, rather
than ground-truth inertia) drives the excess silhouette to **+0.016 at p=0.15**,
i.e. statistically indistinguishable from unimodal. The case against discrete
modes is now stronger than when the report was written.

### Outstanding: B1 — the stages do not share a predictor

`python3.10 -m latent_mechanics.provenance` shows **nine distinct frozen
predictors** across the stages:

| stage | predictor | rows |
|---|---|---|
| Stage 1, Stage 3, **mismatch study** | `base/best.pt` | 48 (doors only) |
| Stage 4 | 8 distinct checkpoints | 20–100 |
| Stage 5 | 7 distinct (one per level) | 48 |
| geometry + UKF | `all_families/best.pt` | 120 |

Stage 4 and 5 *should* differ — the training mixture is the independent variable.
The problem is that **the entire Stage-3 robustness study was run against the
doors-only model**, while the geometry and UKF conclusions rest on the
all-families model. The sensing-crossover result is the project's strongest
claim, and it has not been reproduced on the predictor everything downstream
uses. **This is the highest-priority open item.**

### Outstanding: B2, B3 — harnesses built, not applied

- **B2** `online/hparam_sweep.py` — re-validates the gradient adaptor's
  `lr / window / lr_decay / n_inner_steps` on an arbitrary predictor. Those values
  were tuned in Stage 2 on the doors-only model and never re-tuned, so the
  gradient-descent baseline the UKF is compared against is running
  hyperparameters chosen for a different predictor.
- **B3** `mechanisms/time_resolution.py` — per-family control of `frame_skip` and
  integrator substeps, to test whether the laptop failure dissolves at a higher
  sampling rate. `FRAME_SKIP_OVERRIDES` is empty by default, so importing it
  changes nothing.

Neither is applied; both await sign-off before anything is retrained.

---

## Known limitations

- **Single seed throughout.** Stage 5's failure-rate trend is significant
  (p=0.014); the gain trend is not (p=0.076). Nothing has been replicated.
- **Simulation only.** No perception, no real hardware. Sensor degradation in
  Stage 3 is injected synthetically into otherwise-perfect states.
- **`nonlinear_hinge` is poorly predicted by every method** (~4.8× ceiling),
  consistently. That is the predictor's limit, not any filter's.
- **Partial observability unsolved** — a static latent cannot represent a hidden
  dynamic state (bifold). Needs memory, i.e. an architecture change.
- **Latent ratios below 1.0** in the UKF tables are a scoring artefact
  (family-median ceilings, tail-quarter scoring), not performance beyond optimal.
- **Scale and category are confounded** in the mechanism suite; disentangling
  needs a drawer with door-like inertia.

---

## Open decisions

1. **B1 — rerun the Stage-3 robustness study on the all-families predictor?**
   Recommended. Without it the sensing-crossover claim and the UKF results rest
   on different models.
2. **B2 — re-tune the gradient-descent baseline** before the UKF comparison goes
   in a paper. The fixed-R ablation is the clean control and says the same thing,
   but a re-tuned baseline would make it airtight.
3. **B3 — apply per-family time resolution** to test the laptop hypothesis.
4. **UKF follow-ups**: `R` floor and `smoothing` are unswept starting points.
5. **Iterated EKF** as a cheaper alternative — flagged in the geometry report,
   deliberately not built.
6. **Multi-seed replication** before submission.

---

## Recommended next step

Unchanged from the earlier recommendation, and now better supported: **close the
vision loop.** Estimate joint angle from the simulated camera instead of reading
it from MuJoCo. The infrastructure already exists (`rgbd_camera.py`,
`camera_scene.xml`, and the Stage-1 script still prints "Vision: deferred"), and
Stage 3 has turned it from a loose end into the payoff experiment: vision
supplies quantisation, noise, latency and dropout simultaneously and for real,
in the one regime where the learned method structurally beats RLS.

Do B1 first — it is cheap and the vision experiment should be built on a single
consistent predictor.
