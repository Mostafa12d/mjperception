# Task: Integrate omip_core / omip_mujoco_wrapper into this MuJoCo workspace

## Background

`omip_core` is a ROS-free C++ port (with Python bindings) of OMIP
(Online Multimodal Interactive Perception) — a pipeline that estimates
kinematic structure (rigid bodies + joint types/parameters: rigid,
prismatic, revolute, disconnected) from RGB-D + camera-intrinsics
observations over time. It lives in a separate repo:

- Repo: `HutchinsonGroup/omip/omip_core` (fill in — the repo containing `omip_core/`
  and `omip_mujoco_wrapper/`; it's a fork of `tu-rbo/omip` at
  `Mostafa12d/omip`)
- `omip_core/` — the ROS-free C++ library + pybind11 bindings (built via
  CMake into `omip_core/build/python/omip_core.<abi>.so`)
- `omip_mujoco_wrapper/` — a thin Python package with:
  - `omip_mujoco_wrapper/orchestrator.py` — `OmipOrchestrator`: owns the
    3 pipeline stages (feature_tracker, rb_tracker, joint_tracker) and
    exposes one method, `process_frame(rgb_bgr, depth, timestamp_ns) ->
    FrameResult` (`.rigid_bodies`, `.kinematic_structure`). This is
    MuJoCo-agnostic — it only takes plain numpy arrays.
  - `omip_mujoco_wrapper/driver.py` — `MuJoCoDriver`, a reference
    implementation of the MuJoCo-facing half (rendering RGB+depth,
    computing camera intrinsics from fovy) — **not required**, just a
    reference for the two things you'll adapt into your own loop.
  - `omip_mujoco_wrapper/scene_utils.py` — only relevant to the packaged
    demo scene (a drawer). Not needed here since you already have your
    own scene.
- `PORTING_NOTES.md` (in the omip repo root) — the full porting log,
  including a Phase 6 section documenting exactly what was tried, what
  worked, and what didn't when building the reference demo. **Read this
  before debugging anything that looks wrong** — several non-obvious
  MuJoCo rendering gotchas and one open modeling limitation are already
  documented there (see Known limitations below).

The full port (Phases 0-6: ROS removal, pybind11 bindings, MuJoCo demo)
is done and committed on the omip repo's `master` branch. This task is
**not** more porting work — it's wiring the already-working
`OmipOrchestrator` into a *different*, pre-existing MuJoCo workspace that
this prompt's reader does not have prior context on.

## This workspace

- Has its own MuJoCo simulation loop already running (stepping/forwarding
  an `MjModel`/`MjData` pair via the `mujoco` Python package).
- Has a door (hinge joint) and a camera already defined in its MJCF.
- Does **not** yet render RGB-D or call into omip_core at all.

## Goal

Add omip_core's kinematic-structure estimation to this workspace's
existing simulation loop, without restructuring the scene or the
existing loop more than necessary. Concretely: each simulation frame,
render RGB+depth from the existing camera, feed it to an
`OmipOrchestrator`, and surface the resulting rigid-body poses and
kinematic-structure (joint type + parameters) estimate — e.g. print it,
log it, or expose it to whatever this workspace already does per frame.

## Ground rules

1. **Do not modify anything under `omip_core/` or the ported algorithm
   code.** It's a line-by-line-auditable port of the original OMIP
   estimator; if something looks numerically off, that's a finding to
   report, not something to "fix" by editing the ported Filter classes.
   Tuning knobs are all exposed via `OmipOrchestrator`'s constructor /
   the individual `set*` methods on the 3 stage objects it owns — use
   those, not source edits.
2. **Don't restructure this workspace's existing scene or simulation
   loop more than adding the rendering + orchestrator calls requires.**
   This is an integration task, not a rewrite.
3. **Ask before installing new dependencies system-wide** or making any
   change outside this workspace's own directory (e.g. don't touch the
   omip repo itself unless explicitly asked to).
4. If `omip_core` isn't built yet in the referenced omip repo, that's a
   prerequisite step (see Setup), not something to route around.

## Setup (prerequisites, one-time)

1. Confirm `omip_core` is built in the omip repo: check for
   `omip_core/build/python/omip_core*.so`. If missing:
   ```
   cd <path-to-omip-repo>/omip_core
   mkdir -p build && cd build
   cmake .. && cmake --build . -j
   ```
   Needs `pybind11` + Python dev headers findable by CMake (e.g.
   `brew install pybind11` on macOS) — see
   `omip_core/CMakeLists.txt`'s `OMIP_CORE_BUILD_PYTHON_BINDINGS` option.
2. In *this* workspace's Python environment, make `omip_core` and
   `omip_mujoco_wrapper` importable. Simplest: add both to `sys.path`
   (or `pip install -e <path-to-omip-repo>/omip_mujoco_wrapper`, which
   auto-adds `omip_core`'s build output via its `__init__.py` bootstrap —
   see that file). Confirm with:
   ```python
   import omip_core
   from omip_mujoco_wrapper.orchestrator import OmipOrchestrator
   ```

## Integration steps

1. **Find the existing loop's model/data/camera-name/timestep.** Locate
   where this workspace already creates its `mujoco.MjModel` and
   `mujoco.MjData`, what its simulation step size or target FPS is, and
   the name of the already-defined camera.

2. **Add a renderer and compute camera intrinsics once, outside the
   loop** (see `omip_mujoco_wrapper/driver.py`'s `camera_intrinsics()`
   for the exact formula — fx=fy derived from the camera's `fovy`,
   cx/cy at image center):
   ```python
   import math
   import mujoco
   import omip_core as oc

   renderer = mujoco.Renderer(model, height=<H>, width=<W>)
   cam_id = model.camera("<your_camera_name>").id
   fovy = model.cam_fovy[cam_id]
   fy = <H> / (2 * math.tan(math.radians(fovy) / 2))
   fx, cx, cy = fy, <W> / 2, <H> / 2

   intrinsics = oc.CameraIntrinsics()
   intrinsics.width, intrinsics.height = <W>, <H>
   intrinsics.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
   intrinsics.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
   ```

3. **Create one `OmipOrchestrator`, once, outside the loop:**
   ```python
   from omip_mujoco_wrapper.orchestrator import OmipOrchestrator

   loop_period_ns = 1e9 / <your_fps>
   orchestrator = OmipOrchestrator(loop_period_ns)
   orchestrator.set_camera_info(intrinsics)
   ```

4. **Inside the existing loop**, after whatever step/forward call
   already runs, render + process:
   ```python
   renderer.update_scene(data, camera="<your_camera_name>")
   rgb = renderer.render()                 # HxWx3 uint8, RGB
   bgr = rgb[:, :, ::-1].copy()            # omip expects BGR channel order

   renderer.enable_depth_rendering()
   renderer.update_scene(data, camera="<your_camera_name>")
   depth = renderer.render()               # HxW float32, meters
   renderer.disable_depth_rendering()

   result = orchestrator.process_frame(bgr, depth, frame_idx * loop_period_ns)
   # result.rigid_bodies: RigidBodyPosesAndVels
   # result.kinematic_structure: KinematicStructure (joint type/params per RB pair)
   ```
   Surface `result` however fits this workspace (print, log, plot,
   assert against expected values, etc.) — that surfacing logic is this
   task's to design, since it depends on what the rest of the workspace
   already does per frame.

5. **Verify feature-trackability of the door before trusting any
   output.** `cv::goodFeaturesToTrack` (inside `omip_core`'s
   feature_tracker) needs real visual texture/corners on the door's
   surface — a flat single-color geom gives almost nothing to track but
   silhouette edges. If the door renders as a flat color, either add a
   texture/material with visible pattern, or (if editing the MJCF is
   off-limits) flag this as a blocker rather than silently proceeding
   with near-zero tracked features on the door.

## Known limitations (already discovered — read before re-debugging these)

From `PORTING_NOTES.md`'s Phase 6 section, building the reference demo
in the omip repo:

- **Revolute (hinge) joint classification is an open gap.** A hinged
  door was the *first* thing tried for the reference demo. After real
  tuning effort (system-noise covariances, rotation speed/range, RANSAC
  thresholds), the revolute joint filter's EKF did not converge to a
  correctly-classified estimate for that synthetic trajectory — the
  pipeline instead reported prismatic or disconnected. This was
  investigated as a possible porting bug and ruled out (the ported
  `RevoluteJointFilter` code matches the original faithfully); it's an
  EKF convergence/calibration issue for certain trajectories, not (yet)
  understood precisely. **Since this workspace's object is also a door
  (hinge/revolute), expect the same difficulty** — don't be surprised if
  the joint comes back classified as something other than revolute, and
  don't assume the integration is broken if so. If you get it working
  correctly here, that closes a real open item — document what was
  different (motion speed/range, camera framing, noise parameters, scene
  scale) in this workspace's own notes, since it'd be useful to feed back.
- **Empty background renders at a bogus far-plane depth** (tens of
  meters) unless every pixel in frame is covered by real, finite
  geometry — this can corrupt feature triangulation for features
  detected off the object of interest. Check depth statistics
  (`depth.max()`) for suspiciously large values; if present, either
  extend the scene's background geometry to fill the camera's view, or
  tighten `<visual><map znear=".." zfar=".."/>` to the scene's real
  depth range for better precision (won't fix out-of-geometry rays, but
  helps quantization within range).
- `MultiRBTracker`/`MultiJointTracker` construction parameters in
  `OmipOrchestrator.__init__` are tuned to match the original ROS
  packages' documented cfg-yaml defaults, not to any particular scene —
  they may need retuning for this workspace's specific object
  scale/speed/camera distance, the same way the reference demo needed
  its own tuning pass (see PORTING_NOTES.md Phase 6 for the specific
  knobs that mattered: `estimation_error_threshold`,
  `static_motion_threshold`, `new_rbm_error_threshold`,
  `max_error_to_reassign_feats`, `max_interframe_jump`).

## Definition of done

- The existing simulation loop runs unchanged in its own structure, with
  RGB-D rendering + `orchestrator.process_frame(...)` added per frame.
- Running it produces a `KinematicStructure` with at least one joint
  entry once the door has moved enough (matching
  `min_num_frames_for_new_rb`/`min_joint_age_for_ee` — expect ~20-30
  frames of real motion before a joint estimate appears, same as the
  reference demo).
- Whatever the door's joint gets classified as (correctly revolute, or
  not — see Known limitations), that result is reported clearly, not
  silently discarded or misrepresented as more confident/correct than it
  is.
