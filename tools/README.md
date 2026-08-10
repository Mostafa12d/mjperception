# `live_viewer.py` — watch a mechanism move under the project's own excitation

A standalone, read-only viewer. It builds one articulated object, drives it with
the same synthetic excitation used to generate training data, and shows the
numbers updating live. No robot arm, no controller, no contact actuation: the
action is a generalised force applied straight to the joint's DOF, exactly as
`mechanisms/rollout.py` does it.

Nothing in the simulator, object generator, training pipeline, or any
checkpoint is modified. This tool only reads.

## Running it

**On macOS the interactive viewer must be launched with `mjpython`, not
`python`.** MuJoCo needs the main thread for the Cocoa event loop. The binary
ships with the same interpreter:

```
/Library/Frameworks/Python.framework/Versions/3.10/bin/mjpython tools/live_viewer.py --family door
```

If you launch it with plain `python3.10` the tool catches the error and prints
the `mjpython` command to use instead.

### One line per family

```bash
MJP=/Library/Frameworks/Python.framework/Versions/3.10/bin/mjpython

$MJP tools/live_viewer.py --family door              # revolute, the default
$MJP tools/live_viewer.py --family door_narrow       # door, parameter ranges collapsed
$MJP tools/live_viewer.py --family drawer            # prismatic: metres, ~10x force scale
$MJP tools/live_viewer.py --family laptop --speed 0.5   # tiny inertia, friction-dominated
$MJP tools/live_viewer.py --family bifold            # two links, only the first is observed
$MJP tools/live_viewer.py --family nonlinear_hinge   # + Stribeck / position-dependent friction
$MJP tools/live_viewer.py --family soft_close        # + damper that engages near closed
```

`laptop` is worth slowing down (`--speed 0.5`): its hinge reaches its stop
quickly, which is also why most of its transitions get excluded from the belief
feed (see below).

### Useful flags

```bash
# half speed, stop after 30 simulated seconds
$MJP tools/live_viewer.py --family door --speed 0.5 --duration 30

# force one excitation profile instead of sampling the mix
$MJP tools/live_viewer.py --family door --profile swing

# a different sampled instance of the family, and a different excitation seed
$MJP tools/live_viewer.py --family drawer --instance 3 --seed 7

# let episodes run on without resetting to the closed state
$MJP tools/live_viewer.py --family door --no-reset
```

### Without a window

Runs the identical physics and console readout with no viewer — useful over ssh,
or as a quick check that a family still builds:

```bash
python3.10 tools/live_viewer.py --family laptop --no-viewer --duration 5
```

## What you see

A console line, refreshed ~5x/second (`--readout-hz`), and the same fields drawn
in the viewer window as an overlay (`--no-overlay` to turn off):

```
t   2.40 s  episode 0 (multisine)  q  +0.3184 rad  qdot  +0.9120 rad/s  tau   +1.182 N*m
```

Units follow the joint type and are labelled, because the state interface
deliberately does not rescale between them: **rad / rad/s / N·m** for revolute
families, **m / m/s / N** for `drawer`. `tau_extra` appears for the two families
that carry unmodelled physics (`nonlinear_hinge`, `soft_close`) and shows the
perturbation torque, which is *not* part of the action.

## Belief pipeline (optional)

Pass a trained predictor and the readout also shows the online belief update
running:

```bash
$MJP tools/live_viewer.py --family door \
    --checkpoint runs/latent_mechanics/base/best.pt

$MJP tools/live_viewer.py --family drawer \
    --checkpoint runs/latent_mechanics/mechanisms/runs/exp2_doors_drawers/best.pt
```

Extra fields: `|z|` (latent norm), `tr(P)` (belief uncertainty), `1-step rmse`
(prequential one-step prediction error against the true simulated next state,
split into position and velocity), and `updates`.

Three things to know about it:

- **It is fed at the model's cadence, not the integrator's.** The simulation
  runs at 500 Hz; transitions go to the belief module every `frame_skip` = 10
  steps, i.e. at 50 Hz. The transition convention matches
  `data_gen.transitions_from_log` exactly — the action is the zero-order-hold
  torque across the block, and the first boundary of each episode only seeds the
  previous state.
- **Near-limit transitions are skipped**, matching `SimConfig.exclude_near_limit`
  in training. The readout counts them (`+N skipped`). On `laptop` this is most
  of them, which is honest rather than a defect.
- **`tr(P)` reads `n/a` on this branch.** See below.

Backends: `--belief auto` (default) uses the UKF if `latent_mechanics.belief` is
importable and the gradient-descent baseline otherwise; `--belief gd` / `ukf`
force one. Other knobs: `--init {zero,mean,medoid,random_trained}` (default
`zero`, matching `OnlineConfig.default_init`), `--lr`, `--device`.

At exit the tool calls `assert_network_unchanged()` and prints confirmation, so
"the predictor was frozen" is checked rather than assumed.

## What Step 3 needed from the belief interface

Two mismatches with what a caller might reasonably assume, both worth knowing
before you run it:

1. **The interface is not `(s, a, s') -> (z, P)`.** It is
   `observe(state, action, next_state) -> AdaptorStep`, where `AdaptorStep`
   carries `prediction / target / error / loss / latent / extras`. The belief
   `(mean, cov)` comes from a *separate* `belief()` call. So the viewer calls
   `observe` for the update and `belief()` for the display; the one-step error
   comes from `AdaptorStep.error`, which is genuinely prequential — the
   prediction is made with the belief held *before* the transition is folded in.

2. **On this branch there is no covariance to show.** `main` has
   `latent_mechanics/online/adaptor.py` (`GradientLatentAdaptor`,
   `StaticLatentAdaptor`) but **not** `latent_mechanics/belief/` — the UKF work,
   the `LatentBasis` PCA mapping, and `geometry/` are not merged here.
   `GradientLatentAdaptor.belief()` returns `{"mean": ..., "cov": None}`, so:
   - `tr(P)` displays `n/a` rather than a placeholder number, and
   - `z_pca` is omitted entirely, since the reduced-basis mapping is what would
     supply it.

   Both fields light up with no code change on a branch where the UKF is merged:
   `--belief auto` detects it, `belief()` then returns `cov_reduced`, and the
   basis supplies the first three PCA components. The viewer already reads
   `cov_reduced` in preference to `cov` — the full 16-D `cov` is rank-deficient
   by construction, so its trace would understate uncertainty.

If you want `tr(P)` and `z_pca` live, run this tool from a branch with
`latent_mechanics/belief/` present (e.g. `fixes/foundation-audit`); nothing in
the viewer needs to change.

## Fidelity check

The stepping loop was verified to be **bit-identical** to
`mechanisms.rollout.simulate_mechanism` over a full 6 s episode for all seven
families — same joint position, velocity and applied torque at every one of the
3000 integrator steps — and the model-rate samples it feeds the belief module
match `data_gen.transitions_from_log`'s slicing to float32 precision. The
excitation is drawn with the same per-episode seed formula
`rollout_mechanism` uses, so `--seed S --instance I` reproduces the exact torque
stream that instance saw during dataset generation.

Wall-clock pacing is accurate to ~1% at `--speed` 0.25, 1.0 and 4.0.
