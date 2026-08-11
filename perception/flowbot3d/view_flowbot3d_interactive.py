"""Static, interactive FlowBot3D view on door_iiwa_scene.

Captures one point cloud, runs one FlowBot3D query, then opens a MuJoCo
viewer you can freely orbit/zoom/pan to inspect the predicted flow field
(cyan quiver), contact point (yellow sphere), and pull direction (red
arrow). No physics stepping, no repeat queries — just one static prediction.

Usage:
    python3.10 view_flowbot3d_interactive.py [--scene SCENE_XML] [--qpos ANGLE_RAD]
Close the window to exit.
"""
from __future__ import annotations

import argparse

import mujoco
import mujoco_viewer
import numpy as np

from perception.flowbot3d.flowbot3d_bridge import query_flowbot3d
from perception.rgbd_camera import RGBDCamera
from perception.flowbot3d.view_flowbot3d_prediction import SCENE_PATH, door_pointcloud

N_FIELD_ARROWS = 60
FIELD_ARROW_LEN = 0.1
CONTACT_ARROW_LEN = 0.12
SURFACE_OFFSET = np.array([0.0, -0.03, 0.0])


def _patched_add_marker_to_scene(self, marker):
    """mujoco_viewer 0.1.4 unconditionally sets g.texid/texuniform/texrepeat,
    fields dropped from mjvGeom in this mujoco version; guard them out."""
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError('Ran out of geoms. maxgeom: %d' % self.scn.maxgeom)

    g = self.scn.geoms[self.scn.ngeom]
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    if hasattr(g, "texid"):
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size[:] = np.ones(3) * 0.1
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)

    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == "label", "Only label is a string in mjtGeom."
            g.label = value if value is not None else ""
        elif hasattr(g, key):
            raise ValueError(f"mjtGeom has attr {key} but type {type(value)} is invalid")
        else:
            raise ValueError(f"mjtGeom doesn't have field {key}")

    self.scn.ngeom += 1


mujoco_viewer.MujocoViewer._add_marker_to_scene = _patched_add_marker_to_scene


def _connector(scratch_scn, geom_type, width, start, end):
    """pos/mat/size for a connector-type mjvGeom spanning start->end, via a
    scratch scene so this can feed mujoco_viewer's add_marker()."""
    g = scratch_scn.geoms[0]
    mujoco.mjv_initGeom(
        g, type=geom_type, size=np.zeros(3), pos=np.zeros(3),
        mat=np.eye(3).flatten(), rgba=np.zeros(4, dtype=np.float32),
    )
    mujoco.mjv_connector(g, geom_type, width, start, end)
    return g.pos.copy(), g.mat.copy(), g.size.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=SCENE_PATH, help="scene XML path")
    parser.add_argument("--qpos", type=float, default=None, help="door hinge angle (rad)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    if args.qpos is not None:
        data.qpos[model.joint("hinge").qposadr[0]] = args.qpos
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    points = door_pointcloud(model, data, cam)
    cam.close()  # done with offscreen rendering before the interactive viewer opens

    print(f"Door point cloud: {points.shape[0]} points")
    if points.shape[0] < 10:
        raise RuntimeError("Too few door points captured — check camera framing / segmentation.")

    contact_point, pull_dir, flow = query_flowbot3d(points)
    print(f"contact={np.round(contact_point, 3)}  pull_dir={np.round(pull_dir, 3)}")

    scratch_scn = mujoco.MjvScene(model, maxgeom=10)

    # Precompute every marker geom once — the scene never changes.
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
