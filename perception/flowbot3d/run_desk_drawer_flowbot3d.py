"""
Perception-only check: capture RGB-D from desk_drawer_scene.xml's "rgbd"
camera, build a point cloud of the desk_drawer body, query FlowBot3D for
the predicted contact point + pull direction, and visualize it. No robot,
no physics -- geometry only.

Usage:
    python3 run_desk_drawer_flowbot3d.py
"""
from __future__ import annotations

import os

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from perception.flowbot3d.flowbot3d_bridge import query_flowbot3d
from perception.rgbd_camera import RGBDCamera
from perception.flowbot3d.view_flowbot3d_prediction import render_scene_with_prediction
from scenes import REPO_ROOT, scene_path

SCENE_PATH = scene_path("desk_drawer_scene.xml")
MEDIA_DIR = os.path.join(REPO_ROOT, "media")
OUT_PNG = os.path.join(MEDIA_DIR, "desk_drawer_flowbot3d_prediction.png")
OUT_SCENE_PNG = os.path.join(MEDIA_DIR, "desk_drawer_flowbot3d_scene.png")


def desk_drawer_pointcloud(
    model: mujoco.MjModel, data: mujoco.MjData, cam: RGBDCamera, n_points: int = 1200
) -> np.ndarray:
    """World-frame point cloud restricted to the desk_drawer body's geoms,
    subsampled to n_points (~what FlowBot3D was trained on)."""
    depth = cam.capture(data)[1]
    zfar = float(model.vis.map.zfar)
    valid = np.isfinite(depth) & (depth < zfar)

    points_cam = cam.depth_to_points(depth, max_depth=zfar)
    points_world = cam.points_to_world(points_cam, data)

    seg = cam.capture_segmentation(data)
    geom_id = seg[..., 0][valid]
    is_geom = seg[..., 1][valid] == mujoco.mjtObj.mjOBJ_GEOM

    body_id = model.body("desk_drawer").id
    on_body = is_geom & (geom_id >= 0) & (model.geom_bodyid[np.clip(geom_id, 0, None)] == body_id)
    points_world = points_world[on_body]

    if len(points_world) > n_points:
        idx = np.random.default_rng(0).choice(len(points_world), size=n_points, replace=False)
        points_world = points_world[idx]

    return points_world


def main() -> None:
    os.makedirs(MEDIA_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    points = desk_drawer_pointcloud(model, data, cam)
    cam.close()
    print(f"desk_drawer point cloud: {points.shape[0]} points")
    if points.shape[0] < 10:
        raise RuntimeError("Too few points captured -- check camera framing/segmentation.")

    contact_point, pull_dir, flow = query_flowbot3d(points)
    print(f"Contact point : {contact_point}")
    print(f"Pull direction: {pull_dir}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="tab:blue", alpha=0.5, label="desk_drawer points")
    ax.scatter(*contact_point, c="red", s=80, label="contact point")
    ax.quiver(*contact_point, *(pull_dir * 0.2), color="red", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("FlowBot3D prediction: desk_drawer")
    ax.legend()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}")

    scene_img = render_scene_with_prediction(
        model, data, points, flow, contact_point, pull_dir, camera="view",
    )
    imageio.imwrite(OUT_SCENE_PNG, scene_img)
    print(f"Saved: {OUT_SCENE_PNG}")


if __name__ == "__main__":
    main()
