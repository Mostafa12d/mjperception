"""
Barebones MuJoCo simulation test for the KUKA iiwa 14.

Uses the mujoco_viewer package (same as the flappy sim), which works with
a regular Python interpreter — no mjpython required.

Run with:
    python3.10 test_kuka_sim.py
"""

import os
import mujoco as mj
import mujoco_viewer
from scenes import SCENES_DIR

MODEL_PATH = os.path.join(SCENES_DIR, "kuka_iiwa_14", "scene.xml")


def main() -> None:
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data = mj.MjData(model)

    print(f"Model loaded: {MODEL_PATH}")
    print(f"  Bodies   : {model.nbody}")
    print(f"  Joints   : {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  DOF      : {model.nv}")
    # Number of physics steps per rendered frame so the sim runs in real-time.
    # Target ~60 FPS display: steps_per_render = (1/60) / timestep
    target_fps = 60.0
    steps_per_render = max(1, int(round(1.0 / (target_fps * model.opt.timestep))))
    print(f"  Timestep : {model.opt.timestep*1000:.2f} ms  →  {steps_per_render} steps/frame")
    print("\nViewer open — close the window to exit.")

    viewer = mujoco_viewer.MujocoViewer(model, data)

    while viewer.is_alive:
        for _ in range(steps_per_render):
            mj.mj_step(model, data)
        viewer.render()

    viewer.close()


if __name__ == "__main__":
    main()
