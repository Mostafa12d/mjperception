# Stage 4 — Cross-mechanism generalisation

**Hypothesis under test.** Not "the embedding represents doors" but "the
embedding represents interaction mechanics."

Six mechanism families behind one interaction interface. The dynamics predictor,
the latent representation and the adaptation algorithm are imported from Stages
1–2 and used unchanged; only the environment and the *training mixture* differ.

```bash
python3.10 -m latent_mechanics.mechanisms.tests     # 61 self-checks
python3.10 -m latent_mechanics.mechanisms.study     # ~7 min, 8 trainings
```

| family | joint | units | inertia | what makes it different |
|---|---|---|---|---|
| `door` | revolute | rad, N·m | 10.2 | the original |
| `nonlinear_hinge` | revolute | rad, N·m | 10.5 | Stribeck + position-dependent friction |
| `soft_close` | revolute | rad, N·m | 12.5 | damper that engages near closed |
| `drawer` | **prismatic** | **m, N** | 35.7 | different physical dimension |
| `laptop` | revolute | rad, N·m | **0.007** | 1500× less inertia |
| `bifold` | revolute | rad, N·m | 1.7 | **two links, only one observed** |

---

## Headline: diversity is what makes the representation transfer

Normalised error on held-out families (1.0 = no better than predicting no motion).
"Gain" is no-adaptation ÷ latent adaptation, so >1 means adaptation helped.

| trained on | held-out family | no-adapt | latent | RLS | gain |
|---|---|---|---|---|---|
| **doors only** | nonlinear_hinge | 2.41e-2 | 2.91e-2 | 2.6e-3 | **0.83×** |
| | soft_close | 4.99e-2 | 3.36e-2 | 2.2e-3 | 1.49× |
| | drawer | 3.73e-2 | 6.08e-2 | 3.4e-4 | **0.61×** |
| | laptop | 8.22e-2 | 1.23e-1 | 7.8e-2 | **0.67×** |
| | bifold | 2.01e-2 | 2.89e-2 | 3.8e-3 | **0.70×** |
| **doors + drawers** | laptop | 9.05e-2 | 1.33e-1 | 7.8e-2 | **0.68×** |
| | bifold | 1.73e-2 | 1.98e-2 | 3.8e-3 | 0.87× |
| **5 families (LOFO)** | drawer | 2.49e+0 | 2.16e-1 | 3.4e-4 | **11.6×** |
| | nonlinear_hinge | 1.27e-1 | 2.31e-2 | 2.6e-3 | **5.5×** |
| | door | 6.24e-2 | 2.00e-2 | 5.5e-4 | **3.1×** |
| | soft_close | 3.79e-2 | 2.46e-2 | 2.2e-3 | 1.5× |
| | bifold | 2.97e-2 | 2.22e-2 | 3.8e-3 | 1.3× |
| | laptop | 8.39e-2 | 1.10e-1 | 7.8e-2 | **0.76×** |

**Experiment 1 — a doors-only representation does not transfer.** Adaptation
*actively hurts* on four of five held-out families. Optimising the latent moves
it somewhere the network was never trained to interpret, and the result is worse
than leaving it at the prior. This is the single clearest negative result of the
stage.

**Experiment 2 — adding drawers did not help laptops.** 0.67× → 0.68×,
unchanged. Interaction diversity per se is not what matters; *relevant* diversity
is. A drawer teaches nothing about a 0.007 kg·m² hinge.

**Experiment 3 — with five families, adaptation works on held-out families.**
Gains of 1.3× to 11.6× on five of six. The same algorithm, the same architecture,
the same latent dimensionality — only the training mixture changed. That is the
positive result the hypothesis predicts: the representation becomes a *mechanics*
code when it is trained on enough mechanical variety, and it stays a *door* code
when it is not.

RLS remains more accurate than the latent everywhere except the laptop, which is
consistent with Stages 2–3: these plants are still mostly linear-in-parameters.

---

## What the latent actually encodes

![structure](../../runs/latent_mechanics/mechanisms/latent_structure.png)

| measurement | value |
|---|---|
| log-inertia readout (LOO R², all families) | **+0.92** |
| within-family friction readout | +0.69 |
| within-family stiffness readout | +0.47 |
| within-family inertia readout | +0.36 |
| within-family damping readout | −0.10 |
| family separability (1-NN) | 0.78 (chance 0.20) |
| separability among the 3 *door-scale* families | 0.70 (chance 0.33) |

**The dominant axis is mechanical scale, not category.** PC1 alone is 69% of the
variance and tracks log-inertia at R² = 0.92 across families spanning four orders
of magnitude. In the UMAP the laptop forms an isolated island and the drawer its
own cluster, while the three door-scale families — which are *different
categories* — overlap heavily.

So the answer is closer to "mechanics" than "category", but with an important
caveat: **this suite confounds the two.** Families differ in scale *and* in type
simultaneously, so the residual 0.70 separability among door-scale families could
be behavioural (they genuinely have different friction laws and one is
underactuated) rather than categorical. Disentangling them properly needs
mechanisms of *matched* mechanical scale and different joint type — a drawer with
door-like inertia and force scale. That is the obvious next benchmark.

Damping remains unencoded (−0.10), consistent with every earlier stage: it is
weakly observable at these velocities.

---

## Failures, documented

**1. The laptop fails for everyone, including RLS — and the reason is sampling,
not representation.**

| family | I / b [s] | as a multiple of the 20 ms model step |
|---|---|---|
| door | 6.8 – 508 | 339 – 25 000× |
| drawer | 2.5 – 29.8 | 124 – 1 491× |
| bifold | 1.2 – 23.3 | 58 – 1 163× |
| **laptop** | **0.004 – 0.019** | **0.19 – 0.9×** |

The laptop's mechanical time constant is *shorter than the interval between
observations*. Within one model step it has already reached terminal velocity, so
`(q, q̇)` sampled at 50 Hz cannot resolve its dynamics at all. RLS scores 7.8e-2
on it — its worst result in the entire project — which confirms this is not a
weakness of the learned representation. The pipeline has an implicit timescale
assumption, and a mechanism three orders of magnitude away in inertia violates it.

Honest caveat: this specific failure mode is partly of my making. I raised the
laptop's damping to (0.3, 1.5) during tuning because at realistic low damping the
screen slammed into its stop and 80% of transitions were discarded. Both
configurations are pathological at 50 Hz; the low-damping one fails through data
loss, the high-damping one through under-sampling. A fair laptop experiment needs
a higher sampling rate, not a different model.

**2. Adaptation can be worse than no adaptation.** Whenever the training mixture
does not cover the test mechanism's regime, optimising the latent reliably makes
things worse (0.61–0.87× across seven such cases). The latent optimiser has no
notion of staying in-distribution — it will happily walk to a region the network
cannot interpret. A trust region or an uncertainty-aware step size is the obvious
missing ingredient, and Stage 2's `belief()` covariance slot is where it would go.

**3. Unit mismatch is catastrophic before adaptation.** A model trained without
drawers scores 2.49 normalised error on drawers — 2.5× worse than predicting no
motion at all. Adaptation recovers it to 0.216 (11.6×, the largest gain measured)
but still 600× worse than RLS. Metres and radians are simply different regimes,
and a single shared normalisation cannot span them.

**4. Partial observability is not solved.** The bi-fold cabinet hides a second
link, so the observed state is non-Markov by construction. Adaptation gains only
1.3×, and RLS — equally blind to the hidden link — still beats the latent 5.9×.
Nothing in a *static* latent can represent a hidden dynamic state; that needs
memory in the state, which is an architecture change and therefore out of scope
here. This is the cleanest evidence for what information the mechanics belief is
missing.

---

## Design notes

**One interface, verified.** All six families produce `(state, action,
next_state)` of shape `(2, 1, 2)`, and `tests.py` asserts the simulator contains
no family-specific branch by inspecting its source. Actions are applied directly
as generalised force (`qfrc_applied[dof]`), the only formulation with the same
meaning for a hinge and a slide. This differs from the Stages 1–3 handle-force
path by 1.2e-3 rad over 3 s, so Stage-4 data is regenerated throughout and never
mixed with earlier datasets.

**Labels never reach the model.** Family and physical parameters are stored
alongside the data for analysis; a training sample contains only
`{door_id, state, action, next_state}`, and `door_id` is an opaque row index.
A test asserts this.

**Excitation scaling was necessary and is not cosmetic.** `sample_profile` floors
the bias at `max(frictionloss, 0.5)` — a door-calibrated constant that hands a
0.1 N·m laptop hinge a 3× oversized push. Each family therefore has a
`force_unit`; excitation is generated in door units and converted, so every
family sits on the same side of that floor and receives the same *shape* of
excitation.

**Errors are normalised by true motion per family.** A drawer moves 0.5 m and a
door 2.3 rad; raw RMSE is not comparable across families even in principle.

---

## Limitations

- Scale and category are confounded across families, as discussed above.
- Only one instance count (20/family) and one sampling rate (50 Hz) were tested;
  the laptop result argues the rate should itself be a variable.
- Adaptation uses the medoid initialisation throughout, which is drawn from the
  *training* mixture and so is a poor prior for a distant held-out family.
- No mechanism has more than two links, and none involves contact with a second
  object.
