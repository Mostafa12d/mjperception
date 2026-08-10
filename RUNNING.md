# Running the scripts in this repository

Everything here is MuJoCo-based work on **estimating the mechanics of articulated
objects** (mostly a hinged door) while interacting with them. Four largely
independent lines of work live side by side:

| line of work | what it is | entry points |
|---|---|---|
| **A. RLS baseline** | classical online system identification: `τ = I·θ̈ + μ·sign(θ̇) + b·θ̇`, estimated by recursive least squares, plus an adaptive-impedance controller that uses the running estimate | `run_door_*.py`, `run_friction_sweep.py` |
| **B. KUKA iiwa** | the same estimation/control loop, but driven through a 7-DoF arm with a simulated wrist F/T sensor instead of abstract hinge torques | `run_*iiwa*.py`, `view_*iiwa*.py` |
| **C. Latent mechanics** | a learned dynamics model with a per-object latent "mechanics vector", adapted online and compared head-to-head against A | `latent_mechanics/` (5 stages) |
| **D. Perception** | RGB-D capture, FlowBot3D articulation-flow prediction, and OMIP kinematic-structure estimation | `rgbd_camera.py`, `*flowbot3d*.py`, `run_door_kinematic_estimation.py` |

A and C are the two halves of the central comparison; B and D are the paths
toward making it hardware-realizable.

---

## 1. Setup

### Main environment (Python 3.10)

Everything except the OMIP script runs here.

```bash
cd mjperception
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import mujoco, torch; print(mujoco.__version__, torch.__version__)"
python test_kuka_sim.py        # opens a MuJoCo window with the iiwa; close to exit
```

> Interactive viewers use `mujoco_python_viewer`, **not** `mujoco.viewer`, so a
> plain `python` interpreter is enough on macOS — `mjpython` is not required.

### OMIP environment (Python 3.12)

Only [run_door_kinematic_estimation.py](run_door_kinematic_estimation.py) needs
this. `omip_core` is a locally built pybind11 `.so` compiled for CPython 3.12,
so it cannot be imported from the 3.10 environment.

```bash
python3.12 -m venv .venv-omip
.venv-omip/bin/pip install -r requirements-omip.txt
```

`omip_core` / `omip_mujoco_wrapper` are not pip-installable. They are located on
`sys.path` from a **sibling checkout of the omip repo** — i.e.
`HutchinsonGroup/omip` next to `HutchinsonGroup/mjperception`. If it lives
elsewhere:

```bash
export OMIP_REPO_ROOT=/path/to/omip
```

The extension must already be built, i.e. `omip_core/build/python/omip_core*.so`
must exist.

### FlowBot3D (external repo, own venv)

The FlowBot3D scripts do **not** import FlowBot3D. They shell out to its own
interpreter through [flowbot3d_bridge.py](flowbot3d_bridge.py), because
FlowBot3D pins `torch==1.13.1` with source-built torch-scatter/sparse/cluster,
which conflicts irreconcilably with this project's `torch==2.8.0`.

Expected layout, hard-coded at the top of
[flowbot3d_bridge.py:26-31](flowbot3d_bridge.py#L26-L31):

```
HutchinsonGroup/
├── mjperception/          # this repo
└── flowbot3d/
    ├── .venv/bin/python
    ├── scripts/flowbot3d_query_cli.py
    └── pretrained/model_nomask_vpa.ckpt
```

Edit those constants if your checkout differs. Without FlowBot3D, sections 5.2
onward will not run; nothing else in the repo is affected.

---

## 2. Quick start — the shortest useful path

```bash
# A. RLS baseline: prints the ablation + sweep tables (no files written)
python3.10 run_door_dynamics_validation.py

# C. Latent mechanics, end to end (~2 min total on CPU)
python3.10 -m latent_mechanics.data_gen  --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.train     --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.evaluate  --checkpoint runs/latent_mechanics/base/best.pt
python3.10 -m latent_mechanics.online.experiments --config configs/online_adaptation.yaml
```

---

## 3. Line A — RLS baseline (door, no arm)

These use oracle MuJoCo kinematics (`qpos`/`qvel`/`qacc`) and reconstruct hinge
torque from a handle force. Vision is deliberately not used; see
[park/vision_theta_interface.py](park/vision_theta_interface.py) for the parked
drop-in θ̂ API.

### 3.1 `run_door_dynamics_validation.py` — the reference implementation

**This is the file everything else imports.** `load_model`, `simulate`,
`rls_init`, `rls_step`, `hinge_torque_from_handle_force` and
`true_hinge_inertia` all live here, and both the iiwa scripts and the entire
`latent_mechanics/` package call into it rather than reimplementing the physics.
Changing it changes every downstream result.

```bash
python3.10 run_door_dynamics_validation.py
```

No arguments, no output files — it prints four things to stdout:

1. handle F/T → hinge torque reconstruction checked against oracle `qfrc_applied`
2. ablation: the Phase-0 2-parameter regressor vs a 3-parameter one with viscous damping
3. parameter sweeps over density, friction and excitation
4. online RLS with a forgetting factor

Runtime ≈ 30 s. Model: [door.xml](door.xml), 6 s episodes at dt = 0.002.

### 3.2 `run_door_adaptive_impedance.py` — adaptive control, two conditions

Closes the loop: the impedance controller consumes the live RLS estimate.

```bash
python3.10 run_door_adaptive_impedance.py
```

Runs both conditions back to back:

- **quasi-static creep** — nearly constant torque just above friction, so there
  is no intentional θ̈ and `Î` stays unobservable. This is the negative control.
- **excited** — smooth minimum-jerk open plus a small high-frequency torque
  dither, so the door opens cleanly *and* `Î` still converges.

Writes `adaptive_quasistatic.mp4`, `adaptive_excited.mp4`,
`adaptive_impedance_results.png`, `adaptive_impedance_dither_log.csv`.

### 3.3 `run_door_size_weight_sweep.py` — generalization of the tuned controller

Asks whether the controller tuned in 3.2 survives a change of plant with **no
retuning at all** (`Kp`, `Kd`, `TAU_MAX`, `DITHER_AMP`, `DITHER_FREQ`, `RLS_LAM`,
`TRACE_P_THRESH`, `RAMP_DURATION` are imported unchanged). Only the door changes:
[door_small.xml](door_small.xml), density swept light (200) / heavy (1400).

```bash
python3.10 run_door_size_weight_sweep.py
```

Writes `adaptive_small_{light,heavy}.mp4` and matching `*_dither_log.csv`.

### 3.4 `phase0_observability_demo.py` — the original motivating experiment

Pure NumPy, no MuJoCo. Shows why quasi-static motion cannot identify inertia:
least-squares fit of `[I, μ]` from noisy differentiated positions, compared
between a quasi-static and an excited torque profile, with regressor
conditioning reported.

```bash
python3.10 phase0_observability_demo.py
```

---

## 4. Line B — KUKA iiwa 14

Same estimator, hardware-realizable sensing: θ from proprioceptive forward
kinematics, τ from a simulated wrist F/T sensor read out of the weld
equality-constraint rows of MuJoCo's `efc` arrays. Notably this does *not*
include the door's own frictionloss, which is what lets the estimator recover
`I_hinge` and `μ` separately.

Scene: [door_iiwa_scene.xml](door_iiwa_scene.xml), arm assets in
[kuka_iiwa_14/](kuka_iiwa_14/).

```bash
python3.10 test_kuka_sim.py               # sanity: arm loads and steps
python3.10 run_door_iiwa_estimation.py    # estimation only, F/T + FK sensing
python3.10 run_iiwa_adaptive_impedance.py # + gains that scale with live Î
python3.10 run_friction_sweep.py          # μ_true ∈ {0.5,1,2,3,5,7} N·m, fixed wrong init
```

`run_iiwa_adaptive_impedance.py` writes `iiwa_adaptive_results.png`;
`run_friction_sweep.py` writes `friction_sweep_traces.png` and
`friction_sweep_summary.png`.

### Viewers

```bash
python3.10 view_door_iiwa.py              # excited profile (default)
python3.10 view_door_iiwa.py --mode qs    # quasi-static creep
python3.10 view_door_iiwa.py --video      # write mp4 instead of opening a window
python3.10 view_door_iiwa.py --both       # live window and mp4

python3.10 view_iiwa_adaptive.py          # adaptive controller, live estimates in the title bar
python3.10 view_iiwa_adaptive.py --mode qs
python3.10 view_iiwa_adaptive.py --video
```

Close the window to exit.

---

## 5. Line D — Perception

### 5.1 RGB-D camera

[rgbd_camera.py](rgbd_camera.py) is a library, not a script: it wraps
`mujoco.Renderer` for synchronized RGB + depth, derives pinhole intrinsics from
the camera's `fovy`, and back-projects depth to point clouds in camera or world
frame. Verify it against the standalone scene:

```bash
python3.10 view_camera_scene.py                # saves RGB + depth to media/
python3.10 view_camera_scene.py --interactive  # live rotatable 3D point cloud
```

### 5.2 FlowBot3D — articulation flow prediction

Requires the FlowBot3D setup from §1. All of these run one query and visualize
the predicted per-point flow field, the selected contact point, and the pull
direction.

```bash
# static prediction on the iiwa door scene, saved to media/
python3.10 view_flowbot3d_prediction.py [--qpos 0.5] [--scene SCENE.xml]

# same prediction, but in an orbit/zoom/pan viewer instead of a PNG
python3.10 view_flowbot3d_interactive.py [--scene SCENE.xml] [--qpos 0.5]

# live: the door swings under a scripted torque, re-queried ~1 Hz with the
# overlay updating on the moving door
python3.10 live_flowbot3d_view.py

# perception-only check on the desk drawer (no robot, no physics)
python3.10 run_desk_drawer_flowbot3d.py

# any standalone asset scene -- injects a camera via MjSpec, so the original
# file is never modified and no per-scene setup is needed
python3.10 view_flowbot3d_asset.py --scene /path/to/scene.xml
python3.10 view_flowbot3d_asset.py --scene /path/to/scene.xml --joint lid_hinge --qpos 1.0
```

`--qpos` sets the articulated joint's position before capture, which is how you
check that the prediction tracks the object's configuration rather than
memorizing one pose.

### 5.3 OMIP — kinematic-structure estimation from RGB-D

**Python 3.12 only.** Feeds rendered RGB+D through omip_core's
feature_tracker → rb_tracker → joint_tracker pipeline and logs the resulting
rigid-body poses and joint-type/parameter estimate.

```bash
.venv-omip/bin/python run_door_kinematic_estimation.py
.venv-omip/bin/python run_door_kinematic_estimation.py \
    --lead-in-s 0.5 --swing-s 5.0 --hold-after-s 1.83
```

| flag | default | meaning |
|---|---|---|
| `--lead-in-s` | 0.5 | static hold before motion, lets the feature tracker initialize |
| `--swing-s` | 5.0 | time to swing closed → 90° |
| `--hold-after-s` | 1.83 | static hold after the swing |

Writes `media/door_kinematic_estimation.mp4`,
`media/door_kinematic_estimation_summary.png`,
`media/door_kinematic_estimation_3d.png`, and
`door_kinematic_estimation_log.csv`.

The scene is generated in Python by
[door_kinematic_scene.py](door_kinematic_scene.py) rather than being a static
XML, for two reasons documented there: the door panel needs a literal
checkerboard of small box geoms (MuJoCo's `builtin="checker"` texture only tiles
correctly on `plane` geoms, so a textured box gives the corner detector nothing
to track), and the camera must be enclosed in walls so no pixel renders at the
far plane, which would corrupt triangulation.

**Known limitation, not a bug in this script:** the revolute-joint EKF has an
open convergence gap on hinged-door trajectories and may report
prismatic/disconnected even with a well-tracked rigid body. The script reports
whatever the pipeline actually outputs.

---

## 6. Line C — Latent mechanics (`latent_mechanics/`)

Five stages, each with its own README containing the full results, design
rationale and limitations. **Read those for the science**; this section covers
only how to run them and what depends on what.

| stage | package | README | question |
|---|---|---|---|
| 1 | `latent_mechanics/` | [README](latent_mechanics/README.md) | can one network + a per-door latent represent many doors? |
| 2 | `latent_mechanics/online/` | [README](latent_mechanics/online/README.md) | can the latent be adapted online on an unseen door, vs RLS? |
| 3 | `latent_mechanics/mismatch/` | [README](latent_mechanics/mismatch/README.md) | when does latent adaptation beat explicit estimation? |
| 4 | `latent_mechanics/mechanisms/` | [README](latent_mechanics/mechanisms/README.md) | does the latent encode *doors* or *mechanics*? |
| 5 | `latent_mechanics/curriculum/` | — | does offline mechanical diversity buy online adaptability? |

### Dependency graph

```
Stage 1  data_gen ──► train ──► evaluate
              │          │
              │          └──► runs/latent_mechanics/base/best.pt
              │                    │
              └──► data/door_mechanics.npz
                         │          │
                         └────┬─────┘
                              ▼
                    Stage 2  online.experiments
                              │
                              ▼
                    Stage 3  mismatch.study      (needs the Stage-1 checkpoint)

Stage 4  mechanisms.study     self-contained: trains its own 8 models
Stage 5  curriculum.study     self-contained: trains its own 7 models
```

Stages 4 and 5 do **not** consume `best.pt` or `data/door_mechanics.npz`. They
regenerate their own data and train their own models, because their whole point
is varying the training mixture.

### Stage 1 — representation

```bash
python3.10 -m latent_mechanics.tests                                     # 33 self-checks
python3.10 -m latent_mechanics.data_gen  --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.train     --config configs/latent_mechanics.yaml
python3.10 -m latent_mechanics.evaluate  --checkpoint runs/latent_mechanics/base/best.pt
tensorboard --logdir runs/latent_mechanics
```

≈ 35 s simulation + 55 s CPU training + a few seconds of evaluation. No GPU
needed. Config: [configs/latent_mechanics.yaml](configs/latent_mechanics.yaml)
(48 training doors × 8 episodes × 6 s, 50 Hz, `embed_dim` 16, 60 epochs).

CLI overrides, for quick experiments without editing the YAML:

```bash
python3.10 -m latent_mechanics.data_gen --out data/other.npz --seed 7
python3.10 -m latent_mechanics.train --run-name wide --embed-dim 32 --epochs 100 --device cpu
python3.10 -m latent_mechanics.evaluate --checkpoint <ckpt> --data <npz> --out-dir <dir>
```

Outputs land in `runs/latent_mechanics/<run_name>/`: `best.pt`, `last.pt`,
`config.yaml`, `history.json`, `tb/`, and `eval/` with `per_door_metrics.csv`,
`horizon_metrics.csv`, `embeddings.npy`, `summary.json` and four figures.

**What to check first in the output:** the latent ablation in `evaluate.py`
section 3. If shuffling the embeddings costs less than 1.5× the error, the
network has learned an "average door" and is ignoring the latent — every later
stage is then meaningless. The script warns loudly when this happens.

### Stage 2 — online adaptation

```bash
python3.10 -m latent_mechanics.online.tests                              # 37 self-checks
python3.10 -m latent_mechanics.online.experiments --config configs/online_adaptation.yaml
python3.10 -m latent_mechanics.online.experiments --only 3               # just the RLS comparison
python3.10 -m latent_mechanics.online.experiments --no-animation         # skip the mp4
```

≈ 66 s on CPU. Config:
[configs/online_adaptation.yaml](configs/online_adaptation.yaml). Requires
`runs/latent_mechanics/base/best.pt` and `data/door_mechanics.npz` from Stage 1.

`--only` accepts `1`, `2`, `3` or a comma-separated subset:

1. prediction error vs interaction count on an unseen door
2. latent initialisation: zero / random-trained / mean / medoid
3. latent adaptation vs RLS on identical doors and streams

Outputs in `runs/latent_mechanics/base/online/`: `exp{1,2,3}_*.png`,
`belief_trajectory.png` / `.mp4` / `.npz`, `belief_snapshots.png`,
per-experiment CSVs, `summary.json`.

### Stage 3 — model mismatch

```bash
python3.10 -m latent_mechanics.mismatch.tests                       # 46 self-checks
python3.10 -m latent_mechanics.mismatch.study                       # ~6 min, 8 sweeps
python3.10 -m latent_mechanics.mismatch.study --only stribeck,drift # a subset
python3.10 -m latent_mechanics.mismatch.study --doors 2 --episodes 2 --no-figures  # fast smoke run
```

Config: [configs/mismatch.yaml](configs/mismatch.yaml). Sweep definitions live in
`latent_mechanics/mismatch/config.py` (`default_sweeps`) — one `Sweep` entry per
mismatch mechanism. Adding a mechanism means writing a `PlantPerturbation`
subclass and one `Sweep`; nothing is hard-coded in the simulator or the driver.

Sweep names for `--only`: `encoder_noise`, `quantization`, `dropout`, `latency`,
`stribeck`, `position_friction`, `compliance`, `drift`.

Outputs in `runs/latent_mechanics/base/mismatch/`: `overview.png`, `sweep_*.png`,
`belief_*.png`, CSVs and a LaTeX table.

**Read `holdout_nrmse`, not `angle_nrmse_final`.** The prequential error on a
corrupted stream is confounded — a stale or dropped reading makes the
instantaneous prediction wrong for every method equally, which once produced a
spurious "crossover" that was really two failed methods tying. The hold-out
metric freezes the learned belief and scores it on clean episodes.

### Stage 4 — cross-mechanism generalisation

```bash
python3.10 -m latent_mechanics.mechanisms.tests    # 61 self-checks
python3.10 -m latent_mechanics.mechanisms.study    # ~7 min, 8 trainings
python3.10 -m latent_mechanics.mechanisms.study --only 1,2
python3.10 -m latent_mechanics.mechanisms.study --instances 8 --episodes 2 --epochs 5 --no-analysis
```

| flag | default | meaning |
|---|---|---|
| `--out-dir` | `runs/latent_mechanics/mechanisms` | output directory |
| `--only` | `1,2,3` | which of the three experiments to run |
| `--instances` | 24 | mechanism instances per family |
| `--episodes` | 6 | episodes per instance |
| `--epochs` | 40 | training epochs per model |
| `--no-analysis` | off | skip the latent-structure analysis and its figures |

Six mechanism families: `door`, `nonlinear_hinge`, `soft_close`, `drawer`
(prismatic, metres/newtons), `laptop` (1500× less inertia), `bifold` (two links,
only one observed). Experiment 3 is leave-one-family-out and trains six models,
which is where most of the runtime goes.

`--no-analysis` also avoids the optional `umap-learn` / `scikit-learn`
dependency; without them the analysis falls back to PCA.

### Stage 5 — diversity curriculum

No README yet; the module docstrings in
[levels.py](latent_mechanics/curriculum/levels.py) and
[study.py](latent_mechanics/curriculum/study.py) are the reference.

```bash
python3.10 -m latent_mechanics.curriculum.tests
python3.10 -m latent_mechanics.curriculum.study
python3.10 -m latent_mechanics.curriculum.study --levels 1,4,7
python3.10 -m latent_mechanics.curriculum.study --epochs 5 --instances 8 --no-figures
```

Seven levels of increasing mechanical diversity, trained on a **fixed instance
budget** (48 instances, 5 episodes each, 30 epochs) so that "more diversity" is
never confounded with "more data":

| level | families |
|---|---|
| 1 | `door_narrow` |
| 2 | `door` |
| 3 | + `nonlinear_hinge` |
| 4 | + `drawer` |
| 5 | + `soft_close` |
| 6 | + `bifold` |
| 7 | + `laptop` |

Every level is evaluated on one fixed suite of unseen mechanisms (seed 999,
deliberately far from any training seed), reporting per test instance: `before`
(no adaptation), `after` (final quarter, with adaptation), `gain`, `steps` to
converge, and whether adaptation made things *worse*.

Defaults live in `CurriculumConfig`
([levels.py:71-88](latent_mechanics/curriculum/levels.py#L71-L88)); the CLI flags
override them. Output: `runs/latent_mechanics/curriculum/`.

---

## 7. Tests

Each stage ships self-checks that run as modules and exit non-zero on failure.
They cover baseline integrity, the frozen-network contract, split disjointness,
checkpoint round-trips, and — in Stage 3 — that the perturbed integrator with no
perturbations reproduces `dyn.simulate` **exactly**.

```bash
python3.10 -m latent_mechanics.tests             # 33
python3.10 -m latent_mechanics.online.tests      # 37
python3.10 -m latent_mechanics.mismatch.tests    # 46
python3.10 -m latent_mechanics.mechanisms.tests  # 61
python3.10 -m latent_mechanics.curriculum.tests
```

Run these first when something looks surprising. Lines A, B and D have no
automated tests — `test_kuka_sim.py` is a load-and-step smoke check, not a test
suite.

---

## 8. Outputs, and what is version-controlled

[.gitignore](.gitignore) excludes all generated artifacts: `*.png`, `*.jpg`,
`*.mp4`, `media/`, `runs/`, `data/*.npz`, `MUJOCO_LOG.TXT`. Everything below can
be regenerated by rerunning the commands above.

| path | written by |
|---|---|
| `media/` | perception scripts, OMIP |
| `data/door_mechanics.npz` | Stage 1 `data_gen` |
| `runs/latent_mechanics/base/` | Stages 1–3 |
| `runs/latent_mechanics/mechanisms/` | Stage 4 |
| `runs/latent_mechanics/curriculum/` | Stage 5 |
| `adaptive_*.mp4`, `adaptive_*.png`, `*_dither_log.csv` | line-A control scripts (repo root) |
| `iiwa_adaptive_results.png`, `friction_sweep_*.png` | line-B scripts (repo root) |
| `door_kinematic_estimation_log.csv` | OMIP script (repo root) |

The line-A/B scripts write into the repo root rather than `media/`, which is a
historical inconsistency, not a decision.

---

## 9. Scenes and models

| file | contents |
|---|---|
| [door.xml](door.xml) | the standard door; the plant for line A and Stage 1 |
| [door_small.xml](door_small.xml) | half-size panel; used by the size/weight sweep. Compatible with Stage 1's `doors.model_paths`, but it changes the handle moment arm — re-check the recorded action scale if you add it |
| [door_scene.xml](door_scene.xml) | door with floor/walls for rendering |
| [door_iiwa_scene.xml](door_iiwa_scene.xml) | door + KUKA iiwa 14 welded at the handle |
| [kuka_iiwa_14/](kuka_iiwa_14/) | arm meshes and `scene.xml` |
| [camera_scene.xml](camera_scene.xml), [rgbd_camera.xml](rgbd_camera.xml) | standalone RGB-D camera test scenes |
| [desk_drawer_scene.xml](desk_drawer_scene.xml), [desk_drawer/](desk_drawer/) | prismatic drawer asset for FlowBot3D |
| generated by [door_kinematic_scene.py](door_kinematic_scene.py) | checkerboard door in an enclosed room, for OMIP |

---

## 10. Troubleshooting

**`tensorboard --logdir` fails.** The installed TensorBoard 2.13 predates
protobuf 6.x. Event files are written correctly regardless; upgrading TensorBoard
fixes the viewer, and nothing in the pipeline depends on it.

**`ImportError: omip_core`.** You are on Python 3.10, or `OMIP_REPO_ROOT` is
wrong, or the extension is not built. Check which CPython the `.so` targets:
`file $OMIP_REPO_ROOT/omip_core/build/python/*.so`.

**FlowBot3D scripts hang or fail with a subprocess error.** Check the three paths
at the top of [flowbot3d_bridge.py](flowbot3d_bridge.py): the venv interpreter,
the query CLI, and the checkpoint. The bridge passes point clouds through temp
`.npz` files, so a silent failure is usually a bad path rather than a bad point
cloud.

**Viewer windows crash with a `texid` / `texuniform` attribute error.**
`mujoco_viewer` 0.1.4 sets `mjvGeom` fields that this MuJoCo version dropped.
[view_flowbot3d_interactive.py](view_flowbot3d_interactive.py) monkey-patches
around it; reuse `_patched_add_marker_to_scene` if you write a new viewer.

**A latent-mechanics result looks surprising.** Run that stage's `tests` module
first. Stage 3 in particular asserts bit-exact agreement with the Stage-1
integrator, which is the fastest way to rule out a silent divergence.
