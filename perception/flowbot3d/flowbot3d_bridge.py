"""Subprocess bridge to FlowBot3D.

This project runs on the system Python 3.10 (torch==2.8.0), while FlowBot3D
needs its own pinned stack (torch==1.13.1 + source-built torch-scatter/
sparse/cluster/spline-conv). Installing both into one environment risks the
exact pip-resolver drift observed when setting flowbot3d up: `pip install`
silently upgrading torch and breaking the compiled PyTorch Geometric
extensions. So instead of importing flowbot3d here, this module shells out
to flowbot3d's own venv interpreter as a subprocess, passing the point
cloud through a temp .npz file and reading the prediction back the same way.

Usage:
    from flowbot3d_bridge import query_flowbot3d
    contact_point, pull_dir, flow = query_flowbot3d(points_world)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FLOWBOT3D_REPO = Path(
    "/Users/mostafalotfy/Documents/University/Master/HutchinsonGroup/flowbot3d"
)
FLOWBOT3D_VENV_PYTHON = FLOWBOT3D_REPO / ".venv" / "bin" / "python"
FLOWBOT3D_CLI = FLOWBOT3D_REPO / "scripts" / "flowbot3d_query_cli.py"
DEFAULT_CKPT = FLOWBOT3D_REPO / "pretrained" / "model_nomask_vpa.ckpt"


def query_flowbot3d(
    xyz: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ckpt_path: Path | str = DEFAULT_CKPT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run FlowBot3D on a point cloud via subprocess.

    Args:
        xyz: (N, 3) point cloud, world frame (or any consistent frame).
        mask: (N,) binary mask of "moving part" points. Only meaningful with
            a mask_input_channel=True checkpoint (e.g. pretrained/model.ckpt);
            leave as None when using the default model_nomask_vpa.ckpt.
        ckpt_path: path to a flowbot3d checkpoint.

    Returns:
        (contact_point, pull_dir, flow) — contact_point (3,) and pull_dir
        (3,, unit norm) are the suggested grasp point and pull direction;
        flow (N,3) is the full per-point predicted flow field. All in the
        same frame as the input xyz.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.npz"
        out_path = Path(tmp) / "out.npz"

        if mask is not None:
            np.savez(in_path, xyz=xyz, mask=mask)
        else:
            np.savez(in_path, xyz=xyz)

        proc = subprocess.run(
            [
                str(FLOWBOT3D_VENV_PYTHON),
                str(FLOWBOT3D_CLI),
                "--input", str(in_path),
                "--output", str(out_path),
                "--ckpt", str(ckpt_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"flowbot3d query failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        result = np.load(out_path)
        return result["contact_point"], result["pull_dir"], result["flow"]
