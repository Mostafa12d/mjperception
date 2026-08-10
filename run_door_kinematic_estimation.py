"""Integrates omip_core's kinematic-structure estimator into this
workspace's door simulation, per omip-integration-prompt.md.

Each simulated frame: render RGB+depth from a MuJoCo camera looking at a
hinge-jointed door (door_kinematic_scene.py), feed it to an
OmipOrchestrator (feature_tracker -> rb_tracker -> joint_tracker, all
ported omip_core C++ code via pybind11 -- see the prompt's Ground rule 1:
none of that gets modified here, only its documented constructor/set*
tuning knobs), and log + visualize the resulting rigid-body poses and
joint-type/parameter estimate.

Must run under Python 3.12 (omip_core's build is a cpython-312 extension):
    .venv-omip/bin/python run_door_kinematic_estimation.py

Known, already-documented limitation (see the omip repo's PORTING_NOTES.md
Phase 6 section, and omip-integration-prompt.md's "Known limitations"):
the revolute-joint EKF has an open convergence gap on hinged-door
trajectories -- it may report prismatic/disconnected instead of revolute
even with a well-tracked rigid body. This script reports whatever the
pipeline actually outputs; it does not paper over that gap.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from door_kinematic_scene import DoorKinematicSceneConfig, build_door_kinematic_scene_xml, linear_ramp
from omip_geometry import (
    cam_optical_to_world,
    dir_cam_optical_to_world,
    project_point,
    se3_exp,
    world_to_cam_optical,
)

# --- omip_core / omip_mujoco_wrapper bootstrap -----------------------------
# omip_core's .so is a locally-built pybind11 extension with no installable
# wheel (see PORTING_NOTES.md Phase 6.4), so it's found via sys.path rather
# than pip -- same approach omip_mujoco_wrapper/__init__.py itself uses.
# Assumes the omip repo is a sibling of this workspace's directory
# ("HutchinsonGroup/omip" next to "HutchinsonGroup/mjperception");
# override with the OMIP_REPO_ROOT env var if that's not the case.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OMIP_REPO_ROOT = os.environ.get(
    "OMIP_REPO_ROOT", os.path.normpath(os.path.join(_THIS_DIR, "..", "omip"))
)
for _p in (
    os.path.join(OMIP_REPO_ROOT, "omip_core", "build", "python"),
    os.path.join(OMIP_REPO_ROOT, "omip_mujoco_wrapper"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import omip_core as oc
    from omip_mujoco_wrapper.orchestrator import OmipOrchestrator
except ImportError as e:
    raise ImportError(
        f"Could not import omip_core / omip_mujoco_wrapper from '{OMIP_REPO_ROOT}'. "
        "Set OMIP_REPO_ROOT to the omip repo's path, and make sure omip_core is built "
        "(omip_core/build/python/omip_core*.so must exist) and that you're running this "
        "under the Python version that .so was built for (check with "
        "`file omip_core/build/python/*.so`)."
    ) from e

FPS = 30.0
DEFAULT_LEAD_IN_S = 0.5    # static hold before motion starts (lets the feature tracker init)
DEFAULT_SWING_S = 5.0      # time to swing from closed to TARGET_ANGLE_RAD
DEFAULT_HOLD_AFTER_S = 1.83  # static hold after motion ends
# Camera framing keeps the door's textured face reasonably visible up to
# about 90 deg of rotation (door_kinematic_scene.py) -- see that module's
# DoorKinematicSceneConfig docstring for why the swing stops here rather
# than at door_small.xml's full ~120 deg range.
TARGET_ANGLE_RAD = math.radians(90.0)

MEDIA_DIR = os.path.join(_THIS_DIR, "media")
VIDEO_PATH = os.path.join(MEDIA_DIR, "door_kinematic_estimation.mp4")
LOG_CSV_PATH = os.path.join(_THIS_DIR, "door_kinematic_estimation_log.csv")
SUMMARY_PNG_PATH = os.path.join(MEDIA_DIR, "door_kinematic_estimation_summary.png")
KINEMATICS_3D_PNG_PATH = os.path.join(MEDIA_DIR, "door_kinematic_estimation_3d.png")

CSV_FIELDS = [
    "frame", "t_s", "true_angle_deg", "n_features", "n_rigid_bodies", "rb_ids",
    "n_joints", "most_likely", "prism_p", "rev_p", "rigid_p", "discon_p",
    "rev_joint_value_rad", "prism_joint_value_m",
    # rev_position/rev_orientation as returned by omip_core: expressed in the
    # parent rigid body's local frame (RRBF), not camera or world frame.
    "rev_pos_x", "rev_pos_y", "rev_pos_z",
    "rev_ori_x", "rev_ori_y", "rev_ori_z",
    # Same axis, composed with the parent RB's pose_wc and converted to
    # world coordinates -- directly comparable to the true hinge axis
    # (world origin, direction (0,0,1)).
    "rev_pos_world_x", "rev_pos_world_y", "rev_pos_world_z",
    "rev_ori_world_x", "rev_ori_world_y", "rev_ori_world_z",
]


def pose_from_rb(rb) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) such that p_camera_optical = R @ p_local_at_birth + t, from
    RigidBodyPoseAndVel.pose_wc.twist -- see omip_geometry.py's module
    docstring for why this isn't just reading off a translation field."""
    tw = rb.pose_wc.twist
    twist6 = np.array([tw.rx, tw.ry, tw.rz, tw.vx, tw.vy, tw.vz])
    return se3_exp(twist6)


def estimate_axis_in_camera_frame(joints, rb_transforms: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Composes JointModel.rev_position/rev_orientation (parent-RB-frame)
    with the parent RB's estimated pose to get the estimated joint axis in
    the camera-optical frame -- computed regardless of which joint type
    ended up winning, so it can be visualized even when misclassified."""
    if not joints:
        return None
    j = joints[0]
    R_p, t_p = rb_transforms.get(j.parent_rb_id, (np.eye(3), np.zeros(3)))
    rev_pos_local = np.asarray(j.rev_position, dtype=np.float64).reshape(-1)
    rev_ori_local = np.asarray(j.rev_orientation, dtype=np.float64).reshape(-1)
    axis_point_cam = R_p @ rev_pos_local + t_p
    axis_dir_cam = R_p @ rev_ori_local
    norm = np.linalg.norm(axis_dir_cam)
    if norm > 1e-9:
        axis_dir_cam = axis_dir_cam / norm
    return axis_point_cam, axis_dir_cam


def camera_intrinsics(model: mujoco.MjModel, cam_name: str, width: int, height: int):
    """fx=fy from the camera's fovy, cx/cy at image center -- the standard
    formula for MuJoCo cameras configured via fovy alone (see
    omip_mujoco_wrapper/driver.py's camera_intrinsics(), which this mirrors
    since driver.py itself is a reference, not something imported).
    Returns (oc.CameraIntrinsics, fx, fy, cx, cy)."""
    cam_id = model.camera(cam_name).id
    fovy_deg = float(model.cam_fovy[cam_id])
    fy = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    intrinsics = oc.CameraIntrinsics()
    intrinsics.width = width
    intrinsics.height = height
    intrinsics.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    intrinsics.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return intrinsics, fx, fy, cx, cy


def make_record(frame_idx: int, true_angle_rad: float, n_features: int, rb_ids: list[int], joints,
                 axis_world: tuple[np.ndarray, np.ndarray] | None) -> dict:
    record = {
        "frame": frame_idx,
        "t_s": frame_idx / FPS,
        "true_angle_deg": math.degrees(true_angle_rad),
        "n_features": n_features,
        "n_rigid_bodies": len(rb_ids),
        "rb_ids": ";".join(str(i) for i in rb_ids),
        "n_joints": len(joints),
        "most_likely": "", "prism_p": "", "rev_p": "", "rigid_p": "", "discon_p": "",
        "rev_joint_value_rad": "", "prism_joint_value_m": "",
        "rev_pos_x": "", "rev_pos_y": "", "rev_pos_z": "",
        "rev_ori_x": "", "rev_ori_y": "", "rev_ori_z": "",
        "rev_pos_world_x": "", "rev_pos_world_y": "", "rev_pos_world_z": "",
        "rev_ori_world_x": "", "rev_ori_world_y": "", "rev_ori_world_z": "",
    }
    if joints:
        j = joints[0]
        rev_pos = np.asarray(j.rev_position).reshape(-1)
        rev_ori = np.asarray(j.rev_orientation).reshape(-1)
        record.update({
            "most_likely": str(j.most_likely_joint),
            "prism_p": j.prism_probability,
            "rev_p": j.rev_probability,
            "rigid_p": j.rigid_probability,
            "discon_p": j.discon_probability,
            "rev_joint_value_rad": j.rev_joint_value,
            "prism_joint_value_m": j.prism_joint_value,
            "rev_pos_x": rev_pos[0], "rev_pos_y": rev_pos[1], "rev_pos_z": rev_pos[2],
            "rev_ori_x": rev_ori[0], "rev_ori_y": rev_ori[1], "rev_ori_z": rev_ori[2],
        })
        if axis_world is not None:
            pos_w, dir_w = axis_world
            record.update({
                "rev_pos_world_x": pos_w[0], "rev_pos_world_y": pos_w[1], "rev_pos_world_z": pos_w[2],
                "rev_ori_world_x": dir_w[0], "rev_ori_world_y": dir_w[1], "rev_ori_world_z": dir_w[2],
            })
    return record


_MAX_DRAW_COORD = 5000.0
_RB_COLORS = [(255, 210, 60, 255), (80, 220, 120, 255), (120, 180, 255, 255), (255, 140, 220, 255)]


def _clip_pixel(p: tuple[float, float]) -> tuple[float, float] | None:
    x, y = p
    if abs(x) > _MAX_DRAW_COORD or abs(y) > _MAX_DRAW_COORD:
        return None
    return (x, y)


def _draw_dashed_line(draw: ImageDraw.ImageDraw, p0, p1, color, width=2, dash=7, gap=5) -> None:
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return
    direction = seg / length
    t = 0.0
    while t < length:
        t_end = min(t + dash, length)
        a = p0 + direction * t
        b = p0 + direction * t_end
        draw.line([tuple(a), tuple(b)], fill=color, width=width)
        t += dash + gap


def build_overlay_frame(
    rgb: np.ndarray,
    record: dict,
    fx: float, fy: float, cx: float, cy: float,
    rb_transforms: dict,
    axis_cam: tuple[np.ndarray, np.ndarray] | None,
    true_axis_pts_cam: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    # Ground-truth hinge axis (always exact -- from the known MuJoCo scene
    # geometry + camera pose, not an estimate).
    p0 = project_point(fx, fy, cx, cy, true_axis_pts_cam[0])
    p1 = project_point(fx, fy, cx, cy, true_axis_pts_cam[1])
    if p0 is not None and p1 is not None:
        p0c, p1c = _clip_pixel(p0), _clip_pixel(p1)
        if p0c and p1c:
            draw.line([p0c, p1c], fill=(0, 230, 230, 255), width=3)
            draw.text((p0c[0] + 6, p0c[1]), "true axis", fill=(0, 230, 230, 255))

    # omip's live estimate of the joint axis, composed from rev_position/
    # rev_orientation (parent-RB frame) via the parent RB's estimated pose
    # -- drawn regardless of which joint type currently "wins", since that's
    # exactly what's useful to see when the classification is wrong.
    if axis_cam is not None:
        point, direction = axis_cam
        e0 = project_point(fx, fy, cx, cy, point - 0.5 * direction)
        e1 = project_point(fx, fy, cx, cy, point + 0.5 * direction)
        if e0 is not None and e1 is not None:
            e0c, e1c = _clip_pixel(e0), _clip_pixel(e1)
            if e0c and e1c:
                _draw_dashed_line(draw, e0c, e1c, (255, 60, 220, 255), width=3)
                draw.text((e0c[0] + 6, e0c[1] + 12), "est. axis", fill=(255, 60, 220, 255))

    # Tracked rigid-body origins (estimated poses from pose_wc), so the
    # rigid-body "coordinate frames" the pipeline is tracking are visible,
    # not just the joint estimate between them.
    for i, (rb_id, (_, t_cam)) in enumerate(sorted(rb_transforms.items())):
        p = project_point(fx, fy, cx, cy, t_cam)
        if p is None:
            continue
        pc = _clip_pixel(p)
        if pc is None:
            continue
        color = _RB_COLORS[i % len(_RB_COLORS)]
        r = 5
        draw.ellipse([pc[0] - r, pc[1] - r, pc[0] + r, pc[1] + r], outline=color, width=2)
        draw.text((pc[0] + 7, pc[1] - 6), f"rb{rb_id}", fill=color)

    lines = [
        f"frame {record['frame']:3d}  t={record['t_s']:.2f}s  true_angle={record['true_angle_deg']:6.1f} deg",
        f"features={record['n_features']}  rigid_bodies={record['n_rigid_bodies']} (ids={record['rb_ids']})",
    ]
    if record["most_likely"]:
        lines.append(
            f"joint: {record['most_likely']}  "
            f"rev={record['rev_p']:.2f} prism={record['prism_p']:.2f} "
            f"rigid={record['rigid_p']:.2f} discon={record['discon_p']:.2f}"
        )
        lines.append(f"rev_joint_value={record['rev_joint_value_rad']:.3f} rad")
    else:
        lines.append("joint: (none estimated yet)")

    pad = 4
    line_h = 14
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle([0, 0, rgb.shape[1], box_h], fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255, 255))
    return np.array(img)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-in-s", type=float, default=DEFAULT_LEAD_IN_S,
                         help="static hold before the door starts moving (seconds)")
    parser.add_argument("--swing-s", type=float, default=DEFAULT_SWING_S,
                         help="time to swing the door from closed to 90 deg (seconds)")
    parser.add_argument("--hold-after-s", type=float, default=DEFAULT_HOLD_AFTER_S,
                         help="static hold after the door reaches 90 deg (seconds)")
    args = parser.parse_args()

    open_start_frame = round(args.lead_in_s * FPS)
    open_end_frame = open_start_frame + round(args.swing_s * FPS)
    num_frames = open_end_frame + round(args.hold_after_s * FPS)

    os.makedirs(MEDIA_DIR, exist_ok=True)

    scene_cfg = DoorKinematicSceneConfig()
    xml = build_door_kinematic_scene_xml(scene_cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=scene_cfg.height, width=scene_cfg.width)
    hinge_adr = model.joint("hinge").qposadr[0]

    intrinsics, fx, fy, cx, cy = camera_intrinsics(model, "cam_est", scene_cfg.width, scene_cfg.height)

    # The camera is static (only the door moves), so its world pose can be
    # read once -- needed to convert omip's camera-frame estimates into
    # world coordinates for the 3D plot, and to project the known-exact
    # true hinge axis into the image for the 2D overlay.
    mujoco.mj_forward(model, data)
    cam_id = model.camera("cam_est").id
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    true_axis_world = (
        np.array([0.0, 0.0, scene_cfg.panel_center_z - scene_cfg.panel_half_h]),
        np.array([0.0, 0.0, scene_cfg.panel_center_z + scene_cfg.panel_half_h]),
    )
    true_axis_pts_cam = (
        world_to_cam_optical(true_axis_world[0], cam_pos, cam_mat),
        world_to_cam_optical(true_axis_world[1], cam_pos, cam_mat),
    )

    loop_period_ns = 1e9 / FPS
    orchestrator = OmipOrchestrator(loop_period_ns, number_features=200)
    orchestrator.set_camera_info(intrinsics)

    print(f"omip repo: {OMIP_REPO_ROOT}")
    print(f"Scene: door hinge, range {scene_cfg.hinge_range} rad, camera {scene_cfg.width}x{scene_cfg.height} "
          f"fovy={scene_cfg.fovy_deg} deg, depth clip [{scene_cfg.znear}, {scene_cfg.zfar}] m")
    print(f"Running {num_frames} frames ({num_frames / FPS:.1f}s simulated), door opening "
          f"0 -> {math.degrees(TARGET_ANGLE_RAD):.0f} deg over frames "
          f"[{open_start_frame}, {open_end_frame}]...\n")

    records: list[dict] = []
    video_frames: list[np.ndarray] = []
    depth_warned = False
    early_snapshot = None
    last_snapshot = None

    for frame_idx in range(num_frames):
        true_angle = linear_ramp(frame_idx, open_start_frame, open_end_frame, 0.0, TARGET_ANGLE_RAD)
        data.qpos[hinge_adr] = true_angle
        mujoco.mj_forward(model, data)

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera="cam_est")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="cam_est")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        depth_max = float(depth.max())
        if not depth_warned and depth_max > 0.95 * scene_cfg.zfar:
            print(f"WARNING frame {frame_idx}: depth.max()={depth_max:.2f} m is within 5% of zfar="
                  f"{scene_cfg.zfar} m -- some pixels may be hitting the far clip plane (empty "
                  "background), which can corrupt feature triangulation. See PORTING_NOTES.md Phase 6.")
            depth_warned = True

        timestamp_ns = frame_idx * loop_period_ns
        result = orchestrator.process_frame(bgr, depth, timestamp_ns)

        n_features = orchestrator.feature_tracker.getState().size()
        rb_ids = [rb.rb_id for rb in result.rigid_bodies.rb_poses_and_vels]
        joints = result.kinematic_structure.joints

        if frame_idx == 0 and n_features < 10:
            print(f"WARNING frame 0: only {n_features} features tracked -- the door panel may not "
                  "have enough visual texture/corners for cv::goodFeaturesToTrack. See "
                  "omip-integration-prompt.md step 5.")

        rb_transforms = {rb.rb_id: pose_from_rb(rb) for rb in result.rigid_bodies.rb_poses_and_vels}
        axis_cam = estimate_axis_in_camera_frame(joints, rb_transforms)
        axis_world = None
        if axis_cam is not None:
            axis_world = (
                cam_optical_to_world(axis_cam[0], cam_pos, cam_mat),
                dir_cam_optical_to_world(axis_cam[1], cam_mat),
            )

        record = make_record(frame_idx, true_angle, n_features, rb_ids, joints, axis_world)
        records.append(record)
        video_frames.append(build_overlay_frame(
            rgb, record, fx, fy, cx, cy, rb_transforms, axis_cam, true_axis_pts_cam))

        snapshot = dict(frame=frame_idx, true_angle_deg=math.degrees(true_angle),
                         rb_transforms=rb_transforms, axis_cam=axis_cam, record=record)
        last_snapshot = snapshot
        if early_snapshot is None and len(rb_ids) >= 2 and frame_idx >= 10:
            # A few frames after the door first separates into its own
            # rigid body, so the estimate has had a moment to settle.
            early_snapshot = snapshot

        if frame_idx % 20 == 0 or frame_idx == num_frames - 1:
            joint_str = (
                f"{record['most_likely']} (rev={record['rev_p']:.2f} prism={record['prism_p']:.2f} "
                f"rigid={record['rigid_p']:.2f} discon={record['discon_p']:.2f})"
                if joints else "none yet"
            )
            print(f"  frame {frame_idx:3d}/{num_frames}: true_angle={math.degrees(true_angle):6.1f} deg  "
                  f"features={n_features:3d}  rigid_bodies={rb_ids}  joint={joint_str}")

    print(f"\nSaving video to {VIDEO_PATH} ...")
    imageio.mimsave(VIDEO_PATH, video_frames, fps=FPS, format="FFMPEG")

    with open(LOG_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved per-frame log to {LOG_CSV_PATH}")

    _save_summary_plot(records, open_start_frame, open_end_frame)
    print(f"Saved summary plot to {SUMMARY_PNG_PATH}")

    _save_3d_kinematics_plot(early_snapshot, last_snapshot, cam_pos, cam_mat, true_axis_world, scene_cfg)
    print(f"Saved 3D coordinate-frame plot to {KINEMATICS_3D_PNG_PATH}")

    return _print_final_report(records)


def _save_summary_plot(records: list[dict], open_start_frame: int, open_end_frame: int) -> None:
    frames = [r["frame"] for r in records]
    true_deg = [r["true_angle_deg"] for r in records]
    with_joint = [r for r in records if r["most_likely"]]

    fig, (ax_angle, ax_prob) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_angle.plot(frames, true_deg, color="black", label="true hinge angle")
    ax_angle.axvline(open_start_frame, color="gray", linestyle="--", linewidth=0.8)
    ax_angle.axvline(open_end_frame, color="gray", linestyle="--", linewidth=0.8)
    ax_angle.set_ylabel("true angle [deg]")
    ax_angle.set_title("Door kinematic-structure estimation (omip_core)")
    ax_angle.legend(loc="upper left")

    if with_joint:
        jf = [r["frame"] for r in with_joint]
        ax_prob.plot(jf, [r["rev_p"] for r in with_joint], label="P(revolute)")
        ax_prob.plot(jf, [r["prism_p"] for r in with_joint], label="P(prismatic)")
        ax_prob.plot(jf, [r["rigid_p"] for r in with_joint], label="P(rigid)")
        ax_prob.plot(jf, [r["discon_p"] for r in with_joint], label="P(disconnected)")
        ax_prob.legend(loc="upper left")
    else:
        ax_prob.text(0.5, 0.5, "no joint estimated in any frame", ha="center", va="center",
                     transform=ax_prob.transAxes)
    ax_prob.set_ylabel("joint-type probability")
    ax_prob.set_xlabel("frame")
    ax_prob.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(SUMMARY_PNG_PATH, dpi=150)
    plt.close(fig)


_RB_PLOT_COLORS = {0: "goldenrod", 2: "seagreen"}


def _draw_frame_tripod(ax, origin: np.ndarray, axes_cols: np.ndarray, length: float) -> None:
    """axes_cols: 3x3, each column a world-frame unit direction (local x,y,z)."""
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        d = axes_cols[:, i] * length
        ax.quiver(*origin, *d, color=color, linewidth=1.5, arrow_length_ratio=0.25)


def _plot_snapshot(ax, snapshot: dict, cam_pos: np.ndarray, cam_mat: np.ndarray,
                    true_axis_world: tuple[np.ndarray, np.ndarray], bound: float) -> None:
    _draw_frame_tripod(ax, np.zeros(3), np.eye(3), length=0.2)
    ax.text(0, 0, -0.1, "world", fontsize=8)

    ax.plot(*zip(true_axis_world[0], true_axis_world[1]), color="black", linewidth=3, label="true hinge axis")

    _draw_frame_tripod(ax, cam_pos, cam_mat, length=0.25)
    ax.scatter(*cam_pos, color="black", marker="^", s=60, label="camera")

    for rb_id, (_, t_cam) in sorted(snapshot["rb_transforms"].items()):
        origin_world = cam_optical_to_world(t_cam, cam_pos, cam_mat)
        color = _RB_PLOT_COLORS.get(rb_id, "purple")
        ax.scatter(*origin_world, color=color, s=50, label=f"rb{rb_id} origin (estimated)")

    axis_cam = snapshot["axis_cam"]
    if axis_cam is not None:
        point_cam, dir_cam = axis_cam
        pos_w = cam_optical_to_world(point_cam, cam_pos, cam_mat)
        dir_w = dir_cam_optical_to_world(dir_cam, cam_mat)
        a, b = pos_w - 0.5 * dir_w, pos_w + 0.5 * dir_w
        if np.abs(a).max() <= bound and np.abs(b).max() <= bound:
            ax.plot(*zip(a, b), color="magenta", linewidth=2, linestyle="--", label="estimated axis")
        else:
            ax.text2D(0.02, 0.02,
                       f"estimated axis is off-scale (not drawn):\n"
                       f"pos=({pos_w[0]:.1f}, {pos_w[1]:.1f}, {pos_w[2]:.1f}) m\n"
                       f"dir=({dir_w[0]:.2f}, {dir_w[1]:.2f}, {dir_w[2]:.2f})",
                       transform=ax.transAxes, fontsize=7, color="magenta")

    rec = snapshot["record"]
    joint_desc = rec["most_likely"] or "none"
    ax.set_title(
        f"frame {snapshot['frame']}  (true angle={snapshot['true_angle_deg']:.0f} deg)\n"
        f"joint={joint_desc}  P(rev)={rec['rev_p'] if rec['rev_p'] != '' else 0:.2f}",
        fontsize=9,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound / 2, bound)
    # Equal per-axis scale (1m in x == 1m in y == 1m in z visually) --
    # mplot3d doesn't do this by default, so without it the true axis
    # (vertical) and the horizontal camera/door spread aren't comparable.
    ax.set_box_aspect((2 * bound, 2 * bound, 1.5 * bound))
    # Default (elev=30, azim=-60) looks nearly down the origin->camera
    # direction here, which foreshortens everything into a cluster; this
    # angle actually separates world/camera/door visually.
    ax.view_init(elev=22, azim=-115)


def _save_3d_kinematics_plot(early_snapshot, last_snapshot, cam_pos: np.ndarray, cam_mat: np.ndarray,
                              true_axis_world: tuple[np.ndarray, np.ndarray],
                              scene_cfg: DoorKinematicSceneConfig) -> None:
    """Shows the true hinge axis (exact) against omip's estimated rigid-body
    frames and joint axis (composed from rev_position/rev_orientation +
    the parent RB's pose_wc -- see omip_geometry.py), all in world
    coordinates, at two points in the run: shortly after the door first
    separates into its own tracked rigid body, and at the end."""
    snapshots = [("shortly after 2nd RB detected", early_snapshot), ("final frame", last_snapshot)]
    snapshots = [(label, s) for label, s in snapshots if s is not None]
    if not snapshots:
        return

    bound = max(abs(scene_cfg.cam_pos[0]), abs(scene_cfg.cam_pos[1])) + 0.7  # covers camera + door, snug
    fig = plt.figure(figsize=(6.5 * len(snapshots), 6))
    for i, (label, snapshot) in enumerate(snapshots):
        ax = fig.add_subplot(1, len(snapshots), i + 1, projection="3d")
        _plot_snapshot(ax, snapshot, cam_pos, cam_mat, true_axis_world, bound)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle("Estimated rigid-body frames + joint axis vs. ground truth (world frame)")
    fig.tight_layout()
    fig.savefig(KINEMATICS_3D_PNG_PATH, dpi=150)
    plt.close(fig)


def _print_final_report(records: list[dict]) -> int:
    last = records[-1]
    print("\n=== Final kinematic structure estimate ===")
    print(f"Ground truth: revolute (hinge) joint at world origin, axis (0, 0, 1), "
          f"swung 0 -> {math.degrees(TARGET_ANGLE_RAD):.0f} deg.")

    if not last["most_likely"]:
        print("No joint was ever estimated -- the door's rigid body was likely never separated "
              "from the static environment, or the joint tracker never received enough motion. "
              "See PORTING_NOTES.md Phase 6 and the CSV log's n_rigid_bodies/n_features columns "
              "to see where in the pipeline this happened.")
        return 1

    print(f"Most likely joint type : {last['most_likely']}")
    print(f"  P(revolute)     = {last['rev_p']:.3f}   rev_joint_value = {last['rev_joint_value_rad']:.3f} rad")
    print(f"  P(prismatic)    = {last['prism_p']:.3f}   prism_joint_value = {last['prism_joint_value_m']:.3f} m")
    print(f"  P(rigid)        = {last['rigid_p']:.3f}")
    print(f"  P(disconnected) = {last['discon_p']:.3f}")
    print(f"  estimated revolute axis position    = ({last['rev_pos_x']:.3f}, {last['rev_pos_y']:.3f}, {last['rev_pos_z']:.3f})")
    print(f"  estimated revolute axis orientation  = ({last['rev_ori_x']:.3f}, {last['rev_ori_y']:.3f}, {last['rev_ori_z']:.3f})")

    if "Revolute" not in last["most_likely"]:
        print(
            "\nNOTE: this did NOT classify as revolute, despite the true joint being a hinge. "
            "This matches the open, documented convergence gap in omip_core's RevoluteJointFilter "
            "for hinge trajectories (see omip repo PORTING_NOTES.md Phase 6, and "
            "omip-integration-prompt.md's 'Known limitations') -- it is not being silently reported "
            "as more confident/correct than it is."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
