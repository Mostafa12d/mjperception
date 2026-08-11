"""
Perception-only verification: capture RGB-D from door_iiwa_scene.xml's
"rgbd" camera, build a point cloud of the door body only (segmentation
drops floor/wall/robot), query FlowBot3D for the best contact point + pull
direction, and visualize it. No robot control involved — this validates
perception independently before it's wired into any control loop.

Usage:
    python3.10 view_flowbot3d_prediction.py [--qpos ANGLE_RAD]
"""
from __future__ import annotations

import argparse
import os

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from flowbot3d_bridge import query_flowbot3d
from rgbd_camera import RGBDCamera

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_iiwa_scene.xml")
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
OUT_PNG = os.path.join(MEDIA_DIR, "flowbot3d_prediction.png")
OUT_SCENE_PNG = os.path.join(MEDIA_DIR, "flowbot3d_scene.png")


def door_pointcloud(
    model: mujoco.MjModel, data: mujoco.MjData, cam: RGBDCamera, n_points: int = 1200
) -> np.ndarray:
    """World-frame point cloud restricted to the door body's geoms, subsampled
    to n_points to roughly match the point-cloud density FlowBot3D was
    trained on (~1200 pts/object) — a raw depth image gives 10-100x that many
    points, which is a real distribution shift for the PointNet++ backbone."""
    depth = cam.capture(data)[1]
    zfar = float(model.vis.map.zfar)
    valid = np.isfinite(depth) & (depth < zfar)

    points_cam = cam.depth_to_points(depth, max_depth=zfar)
    points_world = cam.points_to_world(points_cam, data)

    seg = cam.capture_segmentation(data)  # (H, W, 2) int32: [geom_id, mjtObj]
    geom_id = seg[..., 0][valid]
    is_geom = seg[..., 1][valid] == mujoco.mjtObj.mjOBJ_GEOM

    door_id = model.body("door").id
    on_door = is_geom & (geom_id >= 0) & (model.geom_bodyid[np.clip(geom_id, 0, None)] == door_id)
    points_world = points_world[on_door]

    if len(points_world) > n_points:
        idx = np.random.default_rng(0).choice(len(points_world), size=n_points, replace=False)
        points_world = points_world[idx]

    return points_world


def _add_arrow(scn, start, end, width, rgba):
    i = scn.ngeom
    mujoco.mjv_initGeom(
        scn.geoms[i],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(scn.geoms[i], mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
    scn.ngeom = i + 1


def _add_sphere(scn, pos, radius, rgba):
    i = scn.ngeom
    mujoco.mjv_initGeom(
        scn.geoms[i],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scn.ngeom = i + 1


def render_scene_with_prediction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points: np.ndarray,
    flow: np.ndarray,
    contact_point: np.ndarray,
    pull_dir: np.ndarray,
    camera: str = "flowbot_view",
    height: int = 720,
    width: int = 960,
    n_field_arrows: int = 60,
    field_arrow_len: float = 0.1,
    contact_arrow_len: float = 0.12,
    surface_offset: np.ndarray = np.array([0.0, -0.03, 0.0]),
) -> np.ndarray:
    """Render the actual sim (robot + door) with the predicted flow field
    drawn directly on the door in 3D: a cyan quiver sampled across the whole
    surface (so the visualization doesn't depend on one point being visible/
    unoccluded), plus the chosen contact point (yellow sphere) and its pull
    direction (bigger red arrow) highlighted. Geoms are injected into the
    MuJoCo scene after update_scene().

    Annotations are nudged by `surface_offset` (default: 3cm toward the
    camera side, i.e. -y, matching this scene's door-panel/camera geometry)
    so they float just in front of the door face instead of z-fighting with
    it or, worse, dipping into the wall body behind the hinge and getting
    fully depth-occluded — arrows are kept short for the same reason.
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    scn = renderer.scene

    n = min(n_field_arrows, len(points))
    idx = np.random.default_rng(0).choice(len(points), size=n, replace=False)
    for p, f in zip(points[idx], flow[idx]):
        f_norm = f / (np.linalg.norm(f) + 1e-8)
        p = p + surface_offset
        _add_arrow(scn, p, p + f_norm * field_arrow_len, 0.006, [0, 0.9, 0.9, 0.85])

    contact_point = contact_point + surface_offset
    _add_arrow(
        scn,
        contact_point,
        contact_point + pull_dir * contact_arrow_len,
        0.02,
        [1, 0, 0, 1],
    )
    _add_sphere(scn, contact_point, 0.03, [1, 1, 0, 1])

    img = renderer.render().copy()
    renderer.close()
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", type=float, default=0.0, help="door hinge angle (rad)")
    parser.add_argument("--scene", default=SCENE_PATH, help="scene XML path")
    args = parser.parse_args()

    os.makedirs(MEDIA_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        # Start from the "grasp" keyframe (arm welded to the handle in a
        # consistent pose), then override the hinge angle. Note: the weld's
        # relpose is fixed at the keyframe's geometry, so away from qpos=0 the
        # rendered arm pose is no longer physically consistent with the door —
        # fine for testing perception (purely geometric), but use
        # live_flowbot3d_view.py to see a physically consistent, continuously
        # opening door with the arm actually following it.
        mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    data.qpos[model.joint("hinge").qposadr[0]] = args.qpos
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    points = door_pointcloud(model, data, cam)
    cam.close()
    print(f"Door point cloud: {points.shape[0]} points (hinge angle = {args.qpos:.3f} rad)")
    if points.shape[0] < 10:
        raise RuntimeError("Too few door points captured — check camera framing / segmentation.")

    contact_point, pull_dir, flow = query_flowbot3d(points)
    print(f"Contact point : {contact_point}")
    print(f"Pull direction: {pull_dir}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="tab:blue", alpha=0.5, label="door points")
    ax.scatter(*contact_point, c="red", s=80, label="contact point")
    ax.quiver(*contact_point, *(pull_dir * 0.2), color="red", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"FlowBot3D prediction (hinge={args.qpos:.2f} rad)")
    ax.legend()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}")

    scene_img = render_scene_with_prediction(model, data, points, flow, contact_point, pull_dir)
    imageio.imwrite(OUT_SCENE_PNG, scene_img)
    print(f"Saved: {OUT_SCENE_PNG}")


if __name__ == "__main__":
    main()
