"""Run FlowBot3D on any standalone MuJoCo asset scene -- e.g. flowbot3d's own
data/*/assets/*/scene.xml files -- with no per-scene setup required.

These asset scenes have no camera and no robot. This script loads the scene
via mujoco.MjSpec (so the original file is never modified), injects an RGB-D
camera framed from the scene's own <statistic> center/extent, compiles it,
runs one FlowBot3D query on the whole object (every non-world-body geom,
e.g. base + lid together), and opens the same static/interactive viewer as
view_flowbot3d_interactive.py.

Usage:
    python3.10 view_flowbot3d_asset.py --scene /path/to/scene.xml
    python3.10 view_flowbot3d_asset.py --scene /path/to/scene.xml --qpos 1.0
    python3.10 view_flowbot3d_asset.py --scene /path/to/scene.xml --joint lid_hinge --qpos 1.0
Close the window to exit.
"""
from __future__ import annotations

import argparse

import mujoco
import mujoco_viewer
import numpy as np

from flowbot3d_bridge import query_flowbot3d
from rgbd_camera import RGBDCamera
from view_flowbot3d_interactive import (
    CONTACT_ARROW_LEN,
    FIELD_ARROW_LEN,
    N_FIELD_ARROWS,
    SURFACE_OFFSET,
    _connector,
)  # importing this module also installs its mujoco_viewer texid monkeypatch


def _add_rgbd_camera(spec: mujoco.MjSpec, fovy: float = 58.0, resolution=(640, 480)) -> None:
    """Inject a 3/4-angle 'rgbd' camera looking at the scene's <statistic>
    center, at a distance derived from its extent. A dead-on view risks a
    near-flat depth image (see door_iiwa_scene.xml's rgbd camera history);
    an oblique elevated view gives real depth parallax across the object."""
    center = np.array(spec.stat.center, dtype=np.float64)
    extent = float(spec.stat.extent) if spec.stat.extent > 0 else 1.0

    direction = np.array([0.9, -1.0, 0.55])
    direction /= np.linalg.norm(direction)
    cam_pos = center + direction * extent * 1.8

    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    z_local = -forward
    x_local = np.cross(world_up, z_local)
    x_local /= np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    mat = np.column_stack([x_local, y_local, z_local]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)

    cam = spec.worldbody.add_camera(name="rgbd")
    cam.pos = cam_pos
    cam.quat = quat
    cam.fovy = fovy
    cam.resolution = resolution


def _default_joint(model: mujoco.MjModel) -> str | None:
    """The one non-free joint in the model, if there's exactly one."""
    candidates = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    return candidates[0] if len(candidates) == 1 else None


def object_pointcloud(
    model: mujoco.MjModel, data: mujoco.MjData, cam: RGBDCamera, n_points: int = 1200
) -> np.ndarray:
    """World-frame point cloud of every geom NOT directly on the world body
    (id 0) -- i.e. the whole object (all its parts), excluding the floor."""
    depth = cam.capture(data)[1]
    zfar = float(model.vis.map.zfar)
    valid = np.isfinite(depth) & (depth < zfar)

    points_cam = cam.depth_to_points(depth, max_depth=zfar)
    points_world = cam.points_to_world(points_cam, data)

    seg = cam.capture_segmentation(data)
    geom_id = seg[..., 0][valid]
    is_geom = seg[..., 1][valid] == mujoco.mjtObj.mjOBJ_GEOM
    body_id = model.geom_bodyid[np.clip(geom_id, 0, None)]
    on_object = is_geom & (geom_id >= 0) & (body_id != 0)
    points_world = points_world[on_object]

    if len(points_world) > n_points:
        idx = np.random.default_rng(0).choice(len(points_world), size=n_points, replace=False)
        points_world = points_world[idx]

    return points_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="path to an asset scene.xml")
    parser.add_argument("--joint", default=None, help="articulated joint to set --qpos on")
    parser.add_argument("--qpos", type=float, default=None, help="joint position (rad or m)")
    args = parser.parse_args()

    spec = mujoco.MjSpec.from_file(args.scene)
    _add_rgbd_camera(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    if args.qpos is not None:
        joint_name = args.joint or _default_joint(model)
        if joint_name is None:
            raise RuntimeError(
                "Scene has multiple (or zero) articulated joints -- pass --joint explicitly."
            )
        data.qpos[model.joint(joint_name).qposadr[0]] = args.qpos
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    points = object_pointcloud(model, data, cam)
    cam.close()

    print(f"Object point cloud: {points.shape[0]} points")
    if points.shape[0] < 10:
        raise RuntimeError("Too few points captured — check camera framing / segmentation.")

    contact_point, pull_dir, flow = query_flowbot3d(points)
    print(f"contact={np.round(contact_point, 3)}  pull_dir={np.round(pull_dir, 3)}")

    scratch_scn = mujoco.MjvScene(model, maxgeom=10)

    markers = []
    n = min(N_FIELD_ARROWS, len(points))
    idx = np.random.default_rng(0).choice(len(points), size=n, replace=False)
    for p, f in zip(points[idx], flow[idx]):
        f_norm = f / (np.linalg.norm(f) + 1e-8)
        p = p + SURFACE_OFFSET
        pos, mat, size = _connector(
            scratch_scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
            p, p + f_norm * FIELD_ARROW_LEN,
        )
        markers.append(dict(
            type=mujoco.mjtGeom.mjGEOM_ARROW, pos=pos, mat=mat, size=size,
            rgba=[0, 0.9, 0.9, 0.85],
        ))

    c = contact_point + SURFACE_OFFSET
    pos, mat, size = _connector(
        scratch_scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.02,
        c, c + pull_dir * CONTACT_ARROW_LEN,
    )
    markers.append(dict(
        type=mujoco.mjtGeom.mjGEOM_ARROW, pos=pos, mat=mat, size=size,
        rgba=[1, 0, 0, 1],
    ))
    markers.append(dict(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=c, mat=np.eye(3).flatten(),
        size=[0.03, 0, 0], rgba=[1, 1, 0, 1],
    ))

    viewer = mujoco_viewer.MujocoViewer(model, data)
    print("Static FlowBot3D prediction — orbit/zoom freely, close the window to exit.")
    while viewer.is_alive:
        for m in markers:
            viewer.add_marker(**m)
        viewer.render()

    viewer.close()


if __name__ == "__main__":
    main()
