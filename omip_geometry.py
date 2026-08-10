"""Pure-math helpers for turning omip_core's pybind11 outputs into 3D poses
and 2D pixel coordinates, so the kinematic-structure estimate can be drawn
on top of the rendered video / a world-frame plot.

No omip_core import here on purpose -- this only needs the raw numbers
(twist coefficients, vectors, camera intrinsics), so it works the same
whether the source is omip_core's bindings or a plain tuple of floats.

The exact conventions below were confirmed by reading omip_core's C++
source directly (not guessed), since getting the frame conventions wrong
would silently produce a plausible-looking but meaningless visualization:

- `RigidBodyPoseAndVel.pose_wc.twist` (fields rx,ry,rz,vx,vy,vz) is exponential
  (se3) coordinates of that rigid body's *pose* (not a velocity), mapping
  points from the rigid body's local/birth frame into the camera frame:
  p_camera = T(twist) @ p_local. See omip_core/include/omip_core/LieGroup.hpp
  (se3Exp, so3Exp, so3Dexp) and omip_core/src/rb_tracker/RBFilter.cpp
  (predicted_location = _predicted_pose * location_at_birth). Vector layout
  is angular-first: coeffs = [rx,ry,rz, vx,vy,vz]. Translation is NOT just
  the linear part v -- it's so3Dexp(w)^T @ v (the SO(3) left-Jacobian
  transpose applied to v), per LieGroup.hpp's se3Exp.

- `JointModel.rev_position` / `rev_orientation` are expressed in the
  "Reference Rigid Body Frame" (RRBF) -- the parent rigid body's local
  frame, not the camera frame directly (omip_core/include/omip_core/
  joint_tracker/JointFilter.h). To get camera-frame coordinates, compose
  with the parent rigid body's pose_wc-derived transform:
      axis_point_cam = R_parent @ rev_position + t_parent
      axis_dir_cam   = R_parent @ rev_orientation   (rotation only, it's a direction)

- omip_core's camera-frame convention is the standard vision/OpenCV optical
  frame implied by the pinhole K/P matrices we hand it (+X right, +Y down,
  +Z forward into the scene) -- NOT MuJoCo's own camera-local convention
  (+X right, +Y up, camera looks down -Z; see rgbd_camera.py's docstring).
  `cam_optical_to_mujoco_local()` converts between them (it's its own
  inverse: applying it twice is the identity).
"""
from __future__ import annotations

import numpy as np


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: 3x3 rotation matrix for angular part w (axis*angle)."""
    theta = np.linalg.norm(w)
    what = _skew(w)
    if theta < 1e-8:
        # exp(what) = I + what + what^2/2 + O(theta^3)
        return np.eye(3) + what + what @ what * 0.5
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * what + b * (what @ what)


def so3_dexp(w: np.ndarray) -> np.ndarray:
    """SO(3) left-Jacobian V(w) used by se3Exp's translation term
    (LieGroup.hpp: t = so3Dexp(w).transpose() * v)."""
    theta = np.linalg.norm(w)
    what = _skew(w)
    if theta < 1e-8:
        # V(w) = I + what/2 + what^2/6 + O(theta^3)
        return np.eye(3) + 0.5 * what + what @ what / 6.0
    a = (1.0 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) + a * what + b * (what @ what)


def se3_exp(twist6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """twist6 = [rx,ry,rz,vx,vy,vz] (angular first) -> (R, t) such that
    p_camera = R @ p_local + t."""
    twist6 = np.asarray(twist6, dtype=np.float64).reshape(-1)
    w, v = twist6[0:3], twist6[3:6]
    R = so3_exp(w)
    t = so3_dexp(w).T @ v
    return R, t


def cam_optical_to_mujoco_local(p: np.ndarray) -> np.ndarray:
    """Flips between omip's optical convention (+Y down, +Z forward) and
    MuJoCo's camera-local convention (+Y up, +Z backward). Involutory: same
    formula converts either direction."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.array([p[0], -p[1], -p[2]])


def world_to_cam_optical(p_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """cam_mat is MuJoCo's 3x3 camera-local-to-world rotation (e.g.
    data.cam_xmat[cam_id].reshape(3,3)); cam_pos is data.cam_xpos[cam_id]."""
    p_local = cam_mat.T @ (np.asarray(p_world, dtype=np.float64) - cam_pos)
    return cam_optical_to_mujoco_local(p_local)


def dir_world_to_cam_optical(d_world: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    d_local = cam_mat.T @ np.asarray(d_world, dtype=np.float64)
    return cam_optical_to_mujoco_local(d_local)


def cam_optical_to_world(p_cam: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    p_local = cam_optical_to_mujoco_local(p_cam)
    return cam_mat @ p_local + cam_pos


def dir_cam_optical_to_world(d_cam: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    d_local = cam_optical_to_mujoco_local(d_cam)
    return cam_mat @ d_local


def project_point(fx: float, fy: float, cx: float, cy: float, p_cam: np.ndarray) -> tuple[float, float] | None:
    """Pinhole projection of a point already in omip's camera-optical frame
    (+Z forward). Returns None if the point is behind the camera (z <= 0),
    since it can't be meaningfully drawn."""
    x, y, z = p_cam
    if z <= 1e-6:
        return None
    return (fx * x / z + cx, fy * y / z + cy)
