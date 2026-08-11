"""Builds the MJCF scene used by run_door_kinematic_estimation.py.

A dedicated scene, separate from door_small.xml, because the door panel
here needs a literal checkerboard of small boxes (not a single textured
box) for cv::goodFeaturesToTrack to find corners on. Per omip's
PORTING_NOTES.md (Phase 6): MuJoCo's builtin="checker" texture only tiles
correctly on `plane` geoms, not on `box` geoms, in the MuJoCo version that
pipeline was validated against -- so a single textured door panel would
render as a flat color or 1-axis stripes, giving the feature tracker almost
nothing to track. The checkerboard tiles are added as zero-density,
non-colliding overlays on top of the real door_small.xml-style panel, so
this scene's mass/inertia properties (for anyone reusing the geometry)
stay identical to door_small.xml's.

The second empirically-verified finding from the same section: an empty
background renders at MuJoCo's far-plane depth (tens of meters) wherever
no real geometry is in frame, which corrupts feature triangulation. This
scene wraps the camera in an enclosing box of flat-colored walls + ceiling
so every pixel the camera can see is finite, real geometry.
"""
from __future__ import annotations

import dataclasses


def _tile_grid_xml(name_prefix, half_width, half_height, center_x, center_z,
                    y_offset, n=8, color_a=(0.62, 0.48, 0.28), color_b=(0.88, 0.76, 0.52),
                    thickness=0.002):
    """Checkerboard of small box geoms spanning
    [center_x - half_width, center_x + half_width] x [center_z - half_height, center_z + half_height]
    in the door body's local X/Z plane, offset by y_offset along local Y
    (so it sits just off one face of the real panel geom without z-fighting).
    Zero density + no collision: purely a visual overlay for feature
    tracking, doesn't change the body's mass/inertia."""
    tile_w = (2 * half_width) / n
    tile_h = (2 * half_height) / n
    parts = []
    for i in range(n):
        for j in range(n):
            cx = center_x - half_width + tile_w * (i + 0.5)
            cz = center_z - half_height + tile_h * (j + 0.5)
            color = color_a if (i + j) % 2 == 0 else color_b
            parts.append(
                f'<geom name="{name_prefix}_{i}_{j}" type="box" '
                f'size="{tile_w / 2 * 0.98:.5f} {thickness:.5f} {tile_h / 2 * 0.98:.5f}" '
                f'pos="{cx:.5f} {y_offset:.5f} {cz:.5f}" '
                f'rgba="{color[0]} {color[1]} {color[2]} 1" '
                f'contype="0" conaffinity="0" density="0"/>'
            )
    return "\n      ".join(parts)


@dataclasses.dataclass
class DoorKinematicSceneConfig:
    """Same hinge geometry/limits as door_small.xml (kept identical so
    results here are comparable to the dynamics-side scripts), plus the
    visual additions described in the module docstring."""
    width: int = 480
    height: int = 360
    fovy_deg: float = 45.0
    hinge_range: tuple[float, float] = (-0.17, 2.09)
    panel_half_w: float = 0.225
    panel_half_h: float = 0.5
    panel_center_x: float = 0.225
    panel_center_z: float = 0.52
    panel_thickness: float = 0.02
    n_tiles: int = 8
    room_half_extent: float = 3.0
    room_height: float = 6.0
    znear: float = 0.05
    zfar: float = 10.0
    # Empirically chosen (see door_kinematic_estimation notes): keeps the
    # panel's textured face reasonably front-on across roughly [-10, 90] deg
    # of hinge rotation -- it foreshortens to a near-edge-on sliver above
    # ~95 deg, which is why the demo's scripted swing stops at 90 deg
    # rather than using door_small.xml's full ~120 deg range.
    cam_pos: tuple[float, float, float] = (1.9, -1.0, 1.05)
    cam_xyaxes: tuple[float, ...] = (0.5127, 0.8587, 0, -0.2331, 0.1392, 0.9627)


def build_door_kinematic_scene_xml(config: DoorKinematicSceneConfig = DoorKinematicSceneConfig()) -> str:
    """Returns an MJCF XML string: a single hinge-jointed door (joint name
    "hinge"), checkerboarded on both faces of its panel, inside an enclosing
    room so depth is finite everywhere in frame, viewed by a camera named
    "cam_est"."""
    tiles_front = _tile_grid_xml(
        "door_tile_f", config.panel_half_w, config.panel_half_h,
        config.panel_center_x, config.panel_center_z,
        y_offset=-(config.panel_thickness + 0.002), n=config.n_tiles)
    tiles_back = _tile_grid_xml(
        "door_tile_b", config.panel_half_w, config.panel_half_h,
        config.panel_center_x, config.panel_center_z,
        y_offset=(config.panel_thickness + 0.002), n=config.n_tiles)

    re = config.room_half_extent
    rh = config.room_height
    cam_pos_str = " ".join(f"{v:.4f}" for v in config.cam_pos)
    cam_xyaxes_str = " ".join(f"{v:.4f}" for v in config.cam_xyaxes)

    return f"""
<mujoco model="door_kinematic_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <map znear="{config.znear}" zfar="{config.zfar}"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1" diffuse="1 1 1"/>

    <!-- Floor: checker texture is fine on a plane geom (see module docstring). -->
    <geom name="floor" type="plane" size="{re} {re} 0.1" material="grid"/>

    <!-- Enclosing room (flat-colored box walls + ceiling): guarantees every
         camera pixel hits real, finite geometry instead of the empty-sky
         far-plane depth. -->
    <geom name="room_wall_north" type="box" pos="0 {re} {rh / 2}" size="{re} 0.05 {rh / 2}" rgba="0.55 0.6 0.68 1" contype="0" conaffinity="0"/>
    <geom name="room_wall_south" type="box" pos="0 -{re} {rh / 2}" size="{re} 0.05 {rh / 2}" rgba="0.55 0.6 0.68 1" contype="0" conaffinity="0"/>
    <geom name="room_wall_east" type="box" pos="{re} 0 {rh / 2}" size="0.05 {re} {rh / 2}" rgba="0.5 0.55 0.63 1" contype="0" conaffinity="0"/>
    <geom name="room_wall_west" type="box" pos="-{re} 0 {rh / 2}" size="0.05 {re} {rh / 2}" rgba="0.5 0.55 0.63 1" contype="0" conaffinity="0"/>
    <geom name="room_ceiling" type="box" pos="0 0 {rh}" size="{re} {re} 0.05" rgba="0.6 0.63 0.7 1" contype="0" conaffinity="0"/>

    <!-- Door frame (static wall the door is hinged to), same as door_small.xml. -->
    <geom name="wall" type="box" pos="-0.02 0 0.5" size="0.02 0.4 0.5"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

    <body name="door" pos="0 0 0">
      <joint name="hinge" type="hinge" pos="0 0 0" axis="0 0 1"
             limited="true" range="{config.hinge_range[0]} {config.hinge_range[1]}"
             damping="0.05" frictionloss="0.6"/>
      <geom name="panel" type="box" pos="{config.panel_center_x} 0 {config.panel_center_z}"
            size="{config.panel_half_w} {config.panel_thickness} {config.panel_half_h}"
            density="600" rgba="0.8 0.7 0.5 1"/>
      {tiles_front}
      {tiles_back}
      <site name="handle" pos="0.4 0 0.5" size="0.02" rgba="1 0 0 1"/>
    </body>

    <camera name="cam_est" pos="{cam_pos_str}" xyaxes="{cam_xyaxes_str}" fovy="{config.fovy_deg}"/>
  </worldbody>
</mujoco>
"""


def linear_ramp(frame_idx: int, start_frame: int, end_frame: int, start_val: float, end_val: float) -> float:
    """Scripted joint-angle trajectory: holds at start_val, ramps linearly
    to end_val, then holds -- matching the pattern already validated in
    omip_mujoco_wrapper/scene_utils.py's reference demo."""
    if frame_idx <= start_frame:
        return start_val
    if frame_idx >= end_frame:
        return end_val
    t = (frame_idx - start_frame) / (end_frame - start_frame)
    return start_val + t * (end_val - start_val)
