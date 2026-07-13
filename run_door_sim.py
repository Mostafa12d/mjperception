"""
Load the single-hinge door model and watch it swing under the two push
strategies from Phase 0 (quasi-static vs. excited) -- this time as an actual
rendered video instead of a line graph.

Install first (on your own machine -- this needs internet access):
    pip install mujoco imageio[ffmpeg]

Run:
    python3 run_door_sim.py

Produces two files: door_quasistatic.mp4 and door_excited.mp4

Troubleshooting: if you get an OpenGL/rendering context error on a headless
machine, try setting the environment variable before running:
    MUJOCO_GL=egl python3 run_door_sim.py
(or MUJOCO_GL=osmesa if egl isn't available)
"""

import numpy as np
import mujoco
import imageio

MODEL_PATH = "door.xml"
DT = 0.002
T_END = 6.0
N_STEPS = int(T_END / DT)
RENDER_EVERY = 10
FPS = 1.0 / (DT * RENDER_EVERY)

HANDLE_DIST = 0.85  # must match the handle site's local x-offset in door.xml


def tangential_direction(theta):
    """Unit vector, in the world XY plane, perpendicular to the line from the
    hinge to the handle at the current hinge angle theta. Pushing in this
    direction produces pure torque about the (vertical) hinge axis, with no
    force wasted radially -- the same idealization used in the Phase 0 model
    and in the papers we've been comparing against (M&B, Buchanan et al.)."""
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


def run(tau_fn, video_path):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    hinge_qpos_adr = model.joint("hinge").qposadr[0]
    handle_site_id = model.site("handle").id
    door_body_id = model.body("door").id

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    angle_log = []

    for step in range(N_STEPS):
        t = step * DT
        theta = data.qpos[hinge_qpos_adr]

        tau_desired = tau_fn(t)                  # desired torque about the hinge (N*m)
        force_mag = tau_desired / HANDLE_DIST     # force at the handle producing that torque
        direction = tangential_direction(theta)
        force = force_mag * direction
        torque = np.zeros(3)  # no additional applied torque beyond what the force produces

        handle_pos_world = data.site_xpos[handle_site_id].copy()

        # reset the applied-force accumulator each step -- mj_applyFT ADDS to
        # it, so forgetting this would compound the force every step
        data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(model, data, force, torque, handle_pos_world,
                           door_body_id, data.qfrc_applied)

        mujoco.mj_step(model, data)

        if step % RENDER_EVERY == 0:
            renderer.update_scene(data, camera="view")
            frames.append(renderer.render().copy())
            angle_log.append(np.degrees(data.qpos[hinge_qpos_adr]))

    # Explicit FFMPEG plugin — without imageio[ffmpeg] installed, imageio
    # silently falls back to TIFF, which rejects the fps= kwarg.
    imageio.mimsave(video_path, frames, fps=FPS, format="FFMPEG")
    angles = np.array(angle_log)
    print(f"Saved {len(frames)} frames to {video_path} at {FPS:.1f} fps")
    print(f"  angle: start={angles[0]:.1f}°  end={angles[-1]:.1f}°  "
          f"peak={angles.max():.1f}°  contacts_at_end={data.ncon}")


# ----- Condition A: quasi-static -----
# ramp up to just above the joint's frictionloss (3.0 N*m), then hold roughly steady
STICTION_TORQUE_GUESS = 3.5

def tau_quasistatic(t):
    if t < 0.3:
        return STICTION_TORQUE_GUESS * (t / 0.3)
    return STICTION_TORQUE_GUESS


# ----- Condition B: excited -----
# same baseline, plus an oscillating component -> real, visible acceleration swings
def tau_excited(t):
    base = STICTION_TORQUE_GUESS
    return base + 6.0 * np.sin(2 * np.pi * 0.5 * t)


if __name__ == "__main__":
    run(tau_quasistatic, "door_quasistatic.mp4")
    run(tau_excited, "door_excited.mp4")
