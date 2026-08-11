"""MuJoCo scene assets, and the one place that resolves them to absolute paths.

Before the reorganisation the scene XMLs sat in the repo root and were
referenced by bare relative name (``MODEL_PATH = "door.xml"``), which quietly
required every script to be run with the repo root as the working directory --
enough of a trap that two of the viewers called ``os.chdir()`` on startup to
force it.

Paths here are anchored to ``__file__`` instead, so a script works from any
working directory and a wrong path fails immediately and by name rather than
as a confusing MuJoCo parse error.

    from scenes import scene_path
    model = mujoco.MjModel.from_xml_path(scene_path("door.xml"))

``scene_path`` also accepts a path that is already absolute, or one that
resolves against the current directory, and returns it unchanged. That is what
lets configs and cached datasets keep storing the bare name ``"door.xml"``
while still loading correctly.
"""
from __future__ import annotations

from pathlib import Path

SCENES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCENES_DIR.parent

__all__ = ["SCENES_DIR", "REPO_ROOT", "scene_path"]


def scene_path(name: str | Path) -> str:
    """Absolute path to a scene asset.

    Accepts a bare filename (``"door.xml"``), a path relative to the repo root
    (``"scenes/door.xml"``), or an already-resolved absolute path. Anything that
    already points at a real file is returned as-is, which keeps datasets and
    checkpoints written before the reorganisation loadable.
    """
    p = Path(name)
    if p.is_absolute():
        return str(p)
    candidate = SCENES_DIR / p.name
    if candidate.exists():
        return str(candidate)
    if p.exists():                       # relative to the caller's cwd
        return str(p.resolve())
    raise FileNotFoundError(
        f"scene asset {name!r} not found in {SCENES_DIR} or relative to {Path.cwd()}"
    )
