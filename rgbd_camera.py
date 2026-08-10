"""
RGB-D capture helper for MuJoCo cameras defined via rgbd_camera.xml.

Wraps mujoco.Renderer to provide:
  - synchronized RGB + depth capture
  - pinhole intrinsics derived from the camera's fovy/resolution
  - depth -> point cloud back-projection (camera frame and world frame)

Camera frame convention (MuJoCo): +X right, +Y up, camera looks down -Z.
Depth values from mujoco.Renderer are perpendicular ("z-") depth along the
camera's optical axis, in meters — verified empirically: a fronto-parallel
plane at distance d renders depth=d at every pixel, not a per-ray Euclidean
distance. This matches the standard pinhole back-projection used below.

Usage:
    model = mujoco.MjModel.from_xml_path("camera_scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    rgb, depth = cam.capture(data)
    points_cam = cam.depth_to_points(depth)
    points_world = cam.points_to_world(points_cam, data)
"""

from __future__ import annotations

import mujoco
import numpy as np


class RGBDCamera:
    """RGB + depth capture and back-projection for a single named MuJoCo camera."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_name: str,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        self.model = model
        self.cam_name = cam_name
        self.cam_id = model.camera(cam_name).id

        # Default to the resolution baked into the MJCF camera element
        # (see rgbd_camera.xml) unless the caller overrides it.
        res_w, res_h = model.cam_resolution[self.cam_id]
        self.height = int(height) if height is not None else int(res_h)
        self.width = int(width) if width is not None else int(res_w)

        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.fovy_deg = float(model.cam_fovy[self.cam_id])

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "RGBDCamera":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render one synchronized (rgb, depth) pair for the current data state.

        Returns:
            rgb:   (H, W, 3) uint8
            depth: (H, W) float32, meters, perpendicular distance from the
                   camera's image plane (not Euclidean ray distance).
        """
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_name)
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        return rgb, depth

    def intrinsics(self) -> np.ndarray:
        """3x3 pinhole intrinsics matrix K, assuming square pixels.

        MuJoCo cameras configured via fovy alone (no physical sensorsize,
        as in rgbd_camera.xml) use the same focal length in pixels for both
        axes, with fovy setting the vertical field of view:

            fy = (height / 2) / tan(fovy / 2)
            fx = fy
            cx = width / 2, cy = height / 2
        """
        fovy_rad = np.deg2rad(self.fovy_deg)
        fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def depth_to_points(self, depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
        """Back-project a depth image to an (N, 3) point cloud in camera frame.

        Points with non-finite depth, or depth > max_depth (when given,
        e.g. the model's zfar), are dropped.
        """
        K = self.intrinsics()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        v, u = np.indices(depth.shape, dtype=np.float64)
        z = depth.astype(np.float64)

        valid = np.isfinite(z)
        if max_depth is not None:
            valid &= z < max_depth

        u, v, z = u[valid], v[valid], z[valid]

        x_cam = (u - cx) / fx * z
        # Image row increases downward; camera +Y is up, hence the flip.
        y_cam = -(v - cy) / fy * z
        # Camera looks down -Z; points in front of the camera have negative Z.
        z_cam = -z

        return np.stack([x_cam, y_cam, z_cam], axis=-1)

    def points_to_world(self, points_cam: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """Transform an (N, 3) camera-frame point cloud into world coordinates."""
        cam_pos = data.cam_xpos[self.cam_id]
        cam_rot = data.cam_xmat[self.cam_id].reshape(3, 3)
        return points_cam @ cam_rot.T + cam_pos

    def capture_segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """Render a segmentation image for the current data state.

        Returns:
            seg: (H, W, 2) int32 — channel 0 is the object id, channel 1 is
                 the mjtObj enum value (geoms are mujoco.mjtObj.mjOBJ_GEOM,
                 value 5). Background pixels are (-1, -1). Verified against
                 mujoco 3.10's Renderer.enable_segmentation_rendering().
        """
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.cam_name)
        seg = self._renderer.render().copy()
        self._renderer.disable_segmentation_rendering()
        return seg
