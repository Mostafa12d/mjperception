"""
Sanity check for the standalone RGB-D camera scene (camera_scene.xml).

Loads the scene, captures a single RGB + depth frame from the "rgbd"
camera via RGBDCamera, saves both to media/, and prints numeric stats
so the camera can be validated before it's wired into anything else.

Usage:
    python3.10 view_camera_scene.py               # save static images to media/
    python3.10 view_camera_scene.py --interactive  # also open a live, mouse-
                                                    # rotatable 3D point cloud
                                                    # window (click-drag to
                                                    # rotate, scroll to zoom;
                                                    # close the window to exit)
"""

from __future__ import annotations

import argparse
import os

import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from rgbd_camera import RGBDCamera

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_scene.xml")
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
RGB_OUT = os.path.join(MEDIA_DIR, "camera_rgb.png")
DEPTH_OUT = os.path.join(MEDIA_DIR, "camera_depth.png")
POINTS_OUT = os.path.join(MEDIA_DIR, "camera_points_world.npy")
POINTCLOUD_OUT = os.path.join(MEDIA_DIR, "camera_pointcloud.png")
MAX_PLOT_POINTS = 20000


def build_pointcloud_figure(points: np.ndarray, colors: np.ndarray, cam_pos: np.ndarray) -> tuple[plt.Figure, int]:
    """Build a 3D-perspective + top-down colored point cloud figure.

    The left (3D) axes support click-drag rotation and scroll-to-zoom when
    shown in a live window via plt.show() — matplotlib enables this by
    default for any Axes3D, no extra configuration needed.
    """
    n = points.shape[0]
    if n > MAX_PLOT_POINTS:
        idx = np.random.default_rng(0).choice(n, size=MAX_PLOT_POINTS, replace=False)
        points, colors = points[idx], colors[idx]

    fig = plt.figure(figsize=(11, 5.5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.5, depthshade=False)
    ax3d.scatter(*cam_pos, c="black", marker="^", s=80, label="camera")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Point cloud (world frame) — drag to rotate")
    ax3d.legend(loc="upper right")

    ax_top = fig.add_subplot(1, 2, 2)
    ax_top.scatter(points[:, 0], points[:, 1], c=colors, s=1.5)
    ax_top.scatter(cam_pos[0], cam_pos[1], c="black", marker="^", s=80, label="camera")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_title("Top-down view (x-y)")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(loc="upper right")

    fig.tight_layout()
    return fig, points.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and visualize the RGB-D camera scene")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a live, mouse-rotatable 3D point cloud window instead of only saving a static PNG",
    )
    args = parser.parse_args()

    os.makedirs(MEDIA_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, "rgbd")
    rgb, depth = cam.capture(data)

    zfar = float(model.vis.map.zfar)
    finite = np.isfinite(depth)
    valid = finite & (depth < zfar)

    K = cam.intrinsics()
    points_cam = cam.depth_to_points(depth, max_depth=zfar)
    points_world = cam.points_to_world(points_cam, data)
    # Same mask depth_to_points applies internally, so this aligns 1:1 with
    # the rows of points_cam/points_world (row-major indexing order matches).
    point_colors = rgb[valid].astype(np.float64) / 255.0

    imageio.imwrite(RGB_OUT, rgb)

    depth_vis = np.where(valid, depth, np.nan)
    d_min = float(np.nanmin(depth_vis))
    d_max = float(np.nanmax(depth_vis))
    norm = (depth_vis - d_min) / max(d_max - d_min, 1e-6)
    norm = np.nan_to_num(norm, nan=1.0)
    depth_rgb = (cm.viridis(norm)[:, :, :3] * 255).astype(np.uint8)
    imageio.imwrite(DEPTH_OUT, depth_rgb)

    np.save(POINTS_OUT, points_world)

    cam_pos = data.cam_xpos[cam.cam_id].copy()
    fig, n_plotted = build_pointcloud_figure(points_world, point_colors, cam_pos)
    fig.savefig(POINTCLOUD_OUT, dpi=150)

    print(f"Scene       : {SCENE_PATH}")
    print(f"Camera      : \"rgbd\"  resolution={cam.width}x{cam.height}  fovy={cam.fovy_deg:.1f} deg")
    print(f"Intrinsics K:\n{K}")
    print(f"Depth stats : min={d_min:.4f}  max={d_max:.4f}  mean={float(np.nanmean(depth_vis)):.4f} m")
    print(f"Valid depth : {int(valid.sum())}/{depth.size} pixels ({100.0 * valid.sum() / depth.size:.1f}%)")
    print(f"Point cloud : {points_world.shape[0]} points (world frame), {n_plotted} plotted")
    print(f"Saved       : {RGB_OUT}")
    print(f"Saved       : {DEPTH_OUT}")
    print(f"Saved       : {POINTS_OUT}")
    print(f"Saved       : {POINTCLOUD_OUT}")

    cam.close()

    if args.interactive:
        print("\nOpening interactive window — click-drag to rotate, scroll to zoom.")
        print("Close the window to exit.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
