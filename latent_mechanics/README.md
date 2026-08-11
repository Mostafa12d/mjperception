# Latent mechanics embeddings — stage 1

A learned dynamics model that represents many different doors with one network
plus a per-door **latent mechanics vector**:

```
(state, action, z)  ->  next_state
     state  = [door_angle, door_velocity]      (rad, rad/s)
     action = applied hinge torque             (N·m)
     z      = learned vector describing this door's hidden mechanics
```

This is the first building block of a research direction that will eventually be
compared against the RLS online system-identification baseline already in the
repository. **Stage 1 is not online adaptation.** Every training door simply owns
one row of an embedding table, learned jointly with the network by gradient
descent. Stage 2 will keep the network, throw the table away, and optimise a
fresh `z` online for an unseen door.

The RLS baseline (`baseline/run_door_dynamics_validation.py` and everything importing it)
is **not modified by any of this**. This package imports from it.

---

## Quick start

```bash
python3.10 -m latent_mechanics.tests                                    # self-checks
python3.10 -m latent_mechanics.data_gen  --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.train     --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.evaluate  --checkpoint runs/latent_mechanics/base/best.pt
tensorboard --logdir runs/latent_mechanics
```

End to end this is about 35 s of simulation, 55 s of CPU training, and a few
seconds of evaluation. No GPU needed.

---

## Results on the shipped config

48 training doors × 8 episodes × 6 s, 50 Hz model rate, embed_dim 16, 60 epochs.
Metrics are on the validation split (**unseen episodes from the same doors**).

| metric | value |
|---|---|
| one-step angle RMSE | 3.98e-05 rad (0.0023°) |
| one-step velocity RMSE | 1.02e-03 rad/s |
| normalised error (angle / velocity) | 0.006 / 0.114 |
| rollout angle RMSE @ 0.5 s | 2.85e-03 rad |
| rollout angle RMSE @ 2.0 s | 1.70e-02 rad |

"Normalised error" divides by the RMS one-step change in that dimension. It is
the number that matters: a model that predicted `next_state = state` would score
1.0. 0.006 means the angle channel explains >99% of the actual motion.

**The embedding is doing real work** (`evaluate.py` section 3) — same network,
different latent:

| latent | angle RMSE | vs correct |
|---|---|---|
| correct | 3.98e-05 | 1.0× |
| another door's | 1.51e-04 | 3.8× |
| zero | 3.46e-04 | 8.7× |

That gap is exactly the headroom stage-2 online adaptation will try to recover
for a door it has never seen.

**The embedding organises by physics it was never shown** (`evaluate.py`
section 4) — leave-one-door-out linear probe from `z` to each true parameter:

| parameter | R² |
|---|---|
| frictionloss | 0.90 |
| inertia / mass / density | 0.83 |
| spring stiffness | 0.80 |
| spring rest angle | 0.49 |
| viscous damping | 0.02 |

Damping is the honest negative. At the velocities these episodes reach, `b·θ̇` is
small next to Coulomb friction `μ·sign(θ̇)`, so it is barely observable — the same
collinearity the RLS baseline flags in its own 3-parameter ablation. It is a
property of the data, not of the method; exciting higher velocities would be the
fix.

---

## Components

### `config.py`
Nested dataclasses (`doors`, `excitation`, `sim`, `model`, `train`, `eval`) with
YAML overrides. A YAML file only lists what it changes. Unknown keys raise rather
than being silently ignored. The full config is saved into every checkpoint.

### `door_sampler.py` — what a "door" is
One door is one draw of hidden mechanics, held fixed across all of its episodes:
panel density (→ hinge inertia, sampled log-uniform), Coulomb friction, viscous
damping, torsional spring stiffness, spring rest angle.

Models are built by calling the baseline's `dyn.load_model(...)` and then adding
the torsional spring, which that loader does not expose, via `jnt_stiffness` /
`qpos_spring`. Ground-truth inertia comes from the baseline's
`true_hinge_inertia`, so these numbers are directly comparable to RLS estimates.

Training doors get ids `0 … n_train-1`; held-out doors continue above that. The
embedding table is only `n_train` rows, so a stray lookup of a held-out door
fails loudly instead of silently borrowing another door's latent.

### `excitation.py` — torque profiles
Four kinds: `multisine`, `steps`, `chirp`, and `swing`. Amplitudes scale with the
door's own friction so heavy-stiction doors still break away.

Every profile is a **zero-order hold on the model timestep grid**. This is not
cosmetic: a transition `(s_t, a_t) → s_{t+1}` is only well defined if the torque
is constant across the whole interval, and MuJoCo integrates 10 substeps inside
one model step.

`swing` exists because of a MuJoCo detail. Episodes all start from the closed
door, and the obvious fix — randomising the start angle via `model.qpos0` —
**silently corrupts the data**: MuJoCo treats `qpos0` as the reference
configuration at which the body sits in its XML pose, so shifting it decouples
the joint coordinate from the door's geometric angle and breaks the handle moment
arm. Coverage of the closing direction therefore has to come from the torque
signal, which is what `swing` (a slow, large oscillation) provides. It takes
closing-direction samples from ~1% to ~12%.

### `data_gen.py` — dataset generation
**Trajectory generation is not reimplemented.** Every episode is rolled out by
`dyn.simulate`, the baseline's integrator loop, with its handle-F/T → hinge-torque
reconstruction and oracle kinematics. This module samples doors and excitation,
calls `simulate`, and slices the 500 Hz log into model-rate transitions.

Episode length is made configurable by `episode_length()`, a context manager that
temporarily sets `dyn.T_END` / `dyn.N_STEPS` and always restores them. That is how
the baseline's loop gets reused at a configurable horizon without editing the file.

The slicing alignment is the subtle part and is worth stating precisely.
`simulate` logs the state *after* step `i`, and `tau[i]` is the torque applied
over `[i·dt, (i+1)·dt)`. So logged index `j` holds the state at time `(j+1)·dt`,
and a `K`-step transition starting there consumes torques `j+1 … j+K`. Starting at
`j = K-1` and striding by `K` makes that span coincide exactly with one
zero-order-hold block. The code asserts the hold really is constant.

> The recorded action is the mean over the hold, with a tolerance rather than an
> exact-equality check. The commanded torque is an exact ZOH, but the *logged*
> hinge torque wobbles by ~1e-6 relative inside a hold, because `simulate`
> reconstructs it from site kinematics that `mj_step` leaves one integration step
> stale. Far below any modelling error, but not exactly zero.

Saved arrays: `door_id`, `episode_id`, `state`, `action`, `next_state`, `t`,
`split`, `near_limit`, plus episode offsets, the true parameter table, and the
config.

### `dataset.py` — `DoorTransitionDataset`
A `torch.utils.data.Dataset` yielding `{door_id, state, action, next_state}`,
preserving which door every sample came from.

**Splits are episode-level, never per-transition.** Consecutive 20 ms transitions
are near-duplicates, so a randomly split "validation" sample would sit between two
training samples and measure nothing. Validation holds out whole *episodes* from
the *same* doors — the right test for stage 1, where the question is whether a
door's latent generalises to new interactions with that door.

Three splits exist: `train`, `val`, and `heldout_door` (doors with no embedding
row at all, generated now so stage 2 has genuinely unseen mechanics ready).

`exclude_near_limit` (default **on**) drops transitions touching a joint limit.
This one flag was worth 64× in validation loss. Limit contact adds a constraint
torque that is not part of the action, making those steps near-unpredictable, and
because they are ~50× larger than a typical step they contributed **90% of the
squared error** — the model spent all its capacity hedging on impacts and had no
incentive to use the latent at all. The RLS baseline masks the same samples in
`moving_mask`. The `.npz` always stores every transition, so the flag can be
flipped without regenerating.

### `model.py` — `MechanicsDynamicsModel` and `DoorEmbeddingTable`
Two deliberately separate objects.

**`MechanicsDynamicsModel`** is a small MLP (default `[256, 256]`, SiLU, ~71k
weights). No transformers, RNNs, attention, or diffusion. It takes `z` as a plain
tensor and **has no embedding table, no door ids, and no notion of how many doors
exist.** That is the stage-2 contract, and `tests.py` enforces it.

Two details that matter:

- **Delta prediction** (`predict_delta`, default on). The head outputs
  `next_state − state`, not `next_state`. Over a 20 ms step the state barely
  changes, so predicting the state directly makes the identity map a near-optimal
  solution and the model learns nothing about mechanics.
- **Normalisation lives inside the model** as registered buffers, so a checkpoint
  is self-contained: feed raw SI units, get raw SI units back. Stage-2 code does
  not have to carry dataset statistics around. Stats are always computed on the
  training split so every split is measured on the same scale.

Stage-2 helpers: `freeze()` and `new_latent()`.

**`DoorEmbeddingTable`** is an `nn.Embedding` initialised small and near zero, so
every door starts from the same prior and the trained latent cloud stays compact.
Training then spreads the rows onto a shell (norm ≥ 1.8) whose centroid contains
no door, so the origin is *not* a good stage-2 starting point — see
[Starting point for stage 2](#starting-point-for-stage-2).

### `train.py`
Joint optimisation of network and embeddings from one MSE loss, in two parameter
groups: the network at `lr`, the latents at `embedding_lr` (10× higher, because a
given door appears in only a fraction of batches) with stronger weight decay.
Warmup + cosine schedule, gradient clipping, best/last checkpoints, optional
early stopping, `history.json`.

TensorBoard logs train/val loss, raw-unit RMSE per dimension, learning rates,
periodic rollout metrics, and **latent-space geometry** (`embed/norm_mean`,
`embed/spread`, a histogram). Watch those: norms collapsing toward zero means the
doors are not being distinguished; norms exploding means a stage-2 embedding
starting at zero has a long way to travel.

**On the loss.** `loss_space: normalized` (default) computes MSE on the
next-state error with each dimension divided by its std. It is still MSE between
predicted and ground-truth next state — angle and velocity differ by more than an
order of magnitude in scale here, and unnormalised the velocity term would own the
entire gradient. `loss_space: raw` gives the literal unscaled version.

### `rollout.py`
`rollout` / `rollout_episode` feed predictions back in open-loop from the first
state — what you plot. `horizon_errors` does something more informative: it rolls
**every** valid start index forward `H` steps in one batch and averages the error
at step `H`, which is unbiased across the trajectory, unlike a single rollout from
`t=0` whose error is dominated by wherever the first big mistake happened. Windows
passing through a joint limit are excluded, matching the training filter.

One-step MSE flatters a dynamics model; over 20 ms almost any smooth function
looks right. Multi-step error is where a wrong latent shows up, so this is the
metric stage 2 should be judged on.

### `evaluate.py`
Four sections: one-step accuracy (overall, per door, worst doors with their true
parameters), rollout error vs horizon, the **latent ablation**, and the **linear
probe**. Writes `per_door_metrics.csv`, `horizon_metrics.csv`, `embeddings.npy`,
`summary.json`, and four figures.

The ablation is the sanity check the direction depends on, and it is wired to warn
loudly: if shuffling embeddings costs less than 1.5×, the network has learned an
average door and stage 2 would have nothing to optimise.

### `visualize.py`
`rollouts.png` (open-loop overlays), `horizon_error.png`, `per_door_error.png`
(error against each true parameter — shows which regimes are hard), and
`latent_space.png` (PCA of the table coloured by true physics).

### `tests.py`
33 self-checks: baseline integrity, model interface, **the stage-2 contract**,
checkpoint round-trip, dataset structure and split disjointness, transition
fidelity against a fresh MuJoCo run, and rollout shapes.

---

## Stage 2: the interface this was built for

Nothing needs to change in the model. The whole pipeline is:

```python
from latent_mechanics.model import load_checkpoint

# 1. new unseen door — its data is already in the dataset as split "heldout_door"
model, _, cfg, _ = load_checkpoint("runs/latent_mechanics/base/best.pt",
                                   with_embeddings=False)   # table not needed
model.freeze()                                              # 2. freeze network

z = model.new_latent(1)                                     # 3. init embedding
opt = torch.optim.Adam([z], lr=1e-2)                        # 4. optimise ONLY z

for state, action, next_state in stream_of_transitions:
    loss = F.mse_loss(model(state, action, z), next_state)
    opt.zero_grad(); loss.backward(); opt.step()            # 5. prediction improves
```

Held-out doors are already generated and tagged, `new_latent()` already starts at
the average-door prior, normalisation already travels inside the checkpoint, and
`tests.py::test_stage2_contract` already verifies that gradients reach `z` while
every network weight stays frozen.

**This has been run once as a design check** — not as a stage-2 method, which is
still to be built. Taking the shipped checkpoint, held-out door 48, and fitting
only `z` on its first 500 transitions (~10 s of interaction), then measuring on
that door's *remaining* transitions:

| | angle RMSE | velocity RMSE |
|---|---|---|
| `z = 0` (average-door prior) | 2.37e-04 | 4.55e-03 |
| after optimising `z` | 9.74e-05 | 9.19e-04 |

2.4× better on angle, 5× on velocity, with every network weight frozen. So the
latent is adaptable on a door the network has never seen, which is the premise
stage 2 rests on. What stage 2 still needs: doing this *online* from a streaming
interaction rather than a batch, and a fair head-to-head against RLS.

### Starting point for stage 2

Repeating that sweep across **all 8** held-out doors (fit on 2 episodes ≈ 12 s,
measured on the other 6), against the trained-door reference of 3.98e-05 rad:

| latent for an unseen door | angle RMSE | vs reference |
|---|---|---|
| `z = 0` | 3.16e-04 | 7.9× |
| an arbitrary *trained* door's `z` | 9.20e-05 | 2.3× |
| fitted `z` | 7.26e-05 | 1.8× |

**Zeros is the worst of the three — worse than borrowing a random real door.**
The trained latents spread onto a shell (norms 1.8–4.1) whose centroid is the
origin, and the nearest real door is 1.22 away, comparable to the typical
door-to-door spacing of 1.41. So `z = 0` is a hole in the latent space that the
network was never evaluated at during training. `new_latent()` still defaults to
zeros, but stage 2 should pass `init` explicitly — the table's medoid, or the row
of whichever training door best explains the first few transitions.

The residual 1.8× at the fitted optimum is the network's own generalisation gap
to new mechanics; no amount of latent optimisation closes it.

The natural comparison against the RLS baseline: RLS estimates interpretable
parameters `(Î, μ̂, b̂)` online from a linear regressor; this estimates an
uninterpretable `z` online from a learned nonlinear model. Both consume the same
`(θ, θ̇, τ)` stream from the same simulator, so they can be run on identical
trajectories and compared on the same multi-step prediction metric.

---

## Known limitations

- **Joint-limit dynamics are deliberately out of scope.** The model is trained on
  free motion only; a rollout that runs into a limit will overshoot (visible in
  `rollouts.png`, door 1 after t≈4.1 s). Modelling contact would be a separate
  piece of work.
- **Damping is weakly identifiable** at the velocities sampled — see the probe
  table above.
- All doors share one geometry (`door.xml`). `door_small.xml` is compatible and
  can be added to `doors.model_paths`, but it changes the handle moment arm, so
  re-check the recorded-action scale if you do.
- Episodes always start from the closed door, for the `qpos0` reason above.
- `tensorboard --logdir` may fail on this machine: the installed TensorBoard 2.13
  predates protobuf 6.x. Event files are written correctly regardless; upgrading
  TensorBoard fixes the viewer.
