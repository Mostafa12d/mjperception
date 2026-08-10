"""
Live, interactive MuJoCo viewer for FlowBot3D predictions on door_iiwa_scene.

The door opens under a gentle scripted hinge torque (the arm follows it
through the existing weld constraint, gravity-compensated so it doesn't
sag). Roughly once a second, the door's point cloud is re-captured and
re-queried through FlowBot3D, updating a live overlay drawn directly on the
door: a cyan quiver of the predicted flow field, plus the highlighted
contact point (yellow sphere) and pull direction (red arrow).

Usage:
    python3.10 live_flowbot3d_view.py
Close the window to exit.
"""
from __future__ import annotations

import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

from flowbot3d_bridge import query_flowbot3d
from rgbd_camera import RGBDCamera
from view_flowbot3d_prediction import SCENE_PATH, door_pointcloud

HINGE_TORQUE = 4.0  # gently opens the door; weld drags the arm along
QUERY_INTERVAL_SEC = 1.0
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
    """Compute the (pos, mat, size) a connector-type mjvGeom needs to span
    start->end, using a scratch scene so this can feed mujoco_viewer's
    add_marker() (which sets raw geom fields, unlike mjv_connector)."""
    g = scratch_scn.geoms[0]
    mujoco.mjv_initGeom(
        g, type=geom_type, size=np.zeros(3), pos=np.zeros(3),
        mat=np.eye(3).flatten(), rgba=np.zeros(4, dtype=np.float32),
    )
    mujoco.mjv_connector(g, geom_type, width, start, end)
    return g.pos.copy(), g.mat.copy(), g.size.copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    arm_dofs = np.arange(1, model.nv)
    cam = RGBDCamera(model, "rgbd")
    scratch_scn = mujoco.MjvScene(model, maxgeom=10)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    points = flow = contact_point = pull_dir = None
    last_query = 0.0

    print("Live FlowBot3D view — close the window to exit.")
    while viewer.is_alive:
        data.ctrl[:7] = data.qfrc_bias[arm_dofs]
        data.qfrc_applied[0] = HINGE_TORQUE
        mujoco.mj_step(model, data)

        now = time.time()
        if now - last_query > QUERY_INTERVAL_SEC:
            last_query = now
            pts = door_pointcloud(model, data, cam)
            if len(pts) >= 10:
                try:
                    contact_point, pull_dir, flow = query_flowbot3d(pts)
                    points = pts
                    print(
                        f"hinge={data.qpos[0]:.3f} rad  "
                        f"contact={np.round(contact_point, 3)}  "
                        f"pull_dir={np.round(pull_dir, 3)}"
                    )
                except RuntimeError as e:
                    print(f"flowbot3d query failed: {e}")

        viewer._markers = []
        if contact_point is not None:
            n = min(N_FIELD_ARROWS, len(points))
            idx = np.random.default_rng(0).choice(len(points), size=n, replace=False)
            for p, f in zip(points[idx], flow[idx]):
                f_norm = f / (np.linalg.norm(f) + 1e-8)
                p = p + SURFACE_OFFSET
                pos, mat, size = _connector(
                    scratch_scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
                    p, p + f_norm * FIELD_ARROW_LEN,
                )
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_ARROW, pos=pos, mat=mat, size=size,
                    rgba=[0, 0.9, 0.9, 0.85],
                )

            c = contact_point + SURFACE_OFFSET
            pos, mat, size = _connector(
                scratch_scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.02,
                c, c + pull_dir * CONTACT_ARROW_LEN,
            )
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_ARROW, pos=pos, mat=mat, size=size,
                rgba=[1, 0, 0, 1],
            )
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=c, mat=np.eye(3).flatten(),
                size=[0.03, 0, 0], rgba=[1, 1, 0, 1],
            )

        # RGBDCamera's offscreen mujoco.Renderer steals the current GL
        # context on every capture() call; reclaim it before drawing so
        # the on-screen viewer window doesn't render black.
        glfw.make_context_current(viewer.window)
        viewer.render()

    viewer.close()
    cam.close()


if __name__ == "__main__":
    main()
