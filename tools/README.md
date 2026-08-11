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
- **`tr(P)` needs the UKF.** The gradient baseline has no covariance, so it
  displays `n/a` there. See below.

Backends: `--belief auto` (default) uses the UKF if `latent_mechanics.belief` is
importable and the gradient-descent baseline otherwise; `--belief gd` / `ukf`
force one. Other knobs: `--init {zero,mean,medoid,random_trained}` (default
`zero`, matching `OnlineConfig.default_init`), `--lr`, `--device`.

With the UKF, use the checkpoint the basis was fit on, or the reduced
coordinates describe directions the predictor never learned to use:

```bash
$MJP tools/live_viewer.py --family door --belief ukf \
    --checkpoint runs/latent_mechanics/geometry/runs/all_families/best.pt
```

For any other checkpoint the basis is recomputed in memory from that
checkpoint's own table and never written to disk, so the tool stays read-only.

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

2. **Only the UKF has a covariance.** `GradientLatentAdaptor.belief()` returns
   `{"mean": ..., "cov": None}`, so under `--belief gd` the `tr(P)` field shows
   `n/a` and `z_pca` is omitted entirely — the reduced-basis mapping is what
   would supply it. Both fields are live under `--belief ukf`.

   The viewer reads `cov_reduced` in preference to `cov`: the full 16-D `cov` is
   rank-deficient by construction (`decode_covariance` returns `V^T P_r V`, rank
   `d`), so its trace would describe a chart the filter is not working in.

   `UKFLatentAdaptor` truncates whatever basis it is handed to `UKFConfig.dim`
   (6), so the viewer reads the basis back off the adaptor after construction
   and reports the chart actually in use — the persisted artifact has 8
   components, and reporting that would misstate the filter's dimension.

## Does the UKF actually work here?

Yes. Six simulated seconds per family, `--init zero`, all-families checkpoint:

| family | updates | skipped | tr(P) start → end | rmse q | rmse v |
|---|---|---|---|---|---|
| door | 299 | 0 | 7.78 → 0.073 | 9.96e-05 | 4.01e-03 |
| door_narrow | 299 | 0 | 6.41 → 0.063 | 2.67e-04 | 1.90e-02 |
| nonlinear_hinge | 299 | 0 | 4.73 → 0.100 | 6.51e-04 | 4.06e-02 |
| soft_close | 299 | 0 | 4.89 → 0.067 | 6.01e-04 | 3.79e-02 |
| drawer | 252 | 47 | 3.58 → 0.122 | 2.39e-03 | 1.32e-01 |
| bifold | 299 | 0 | 5.34 → 0.098 | 4.93e-04 | 3.26e-02 |
| laptop | 78 | 221 | 8.80 → 1.048 | 6.49e-03 | 4.52e-01 |

Uncertainty collapses by roughly two orders of magnitude everywhere the filter
gets a full stream of evidence. `laptop` is the informative exception: it spends
most of its time against a joint stop, so 221 of its 299 transitions are
excluded and it receives about a quarter of the evidence — and its `tr(P)`
correspondingly stalls an order of magnitude higher than everything else. That
is the filter honestly reporting that it has not been told enough, which is the
behaviour you would want to be able to see.

On the same door stream the UKF's one-step error is about 6x lower than the
gradient baseline's (`q` 9.96e-05 vs 5.95e-04, `v` 4.01e-03 vs 1.74e-02).

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
