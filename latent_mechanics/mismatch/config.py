"""
Configuration for the Stage-3 robustness study.

Every source of mismatch is enabled independently through a ``Sweep``: one
mechanism, one parameter, a list of severity levels whose first entry is always
the unperturbed control. Nothing is hard-coded in the simulator or the study
driver -- adding a new mismatch means adding a ``PlantPerturbation`` subclass and
one ``Sweep`` entry.

Severity levels below were calibrated by measuring the RMS unmodelled torque
each setting produces against the 5.96 N*m RMS commanded torque of the held-out
population, so the plant sweeps are roughly comparable to one another at
approximately 0%, 3%, 10% and 30% unmodelled torque.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Sweep:
    """One mismatch mechanism swept over severity levels.

    Args:
        name: identifier used in filenames and tables.
        kind: ``"plant"`` (needs re-simulation) or ``"sensor"`` (post-processes
            an ideal rollout).
        target: for plant sweeps, the ``PERTURBATION_TYPES`` key; for sensor
            sweeps, unused.
        param: the field varied.
        levels: severity values. ``levels[0]`` must be the no-mismatch control.
        fixed: other constructor arguments held constant.
        label: axis label for figures.
        experiment: which numbered experiment this belongs to.
    """

    name: str
    kind: str
    param: str
    levels: list[Any]
    target: str = ""
    fixed: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    experiment: int = 0

    def axis_label(self) -> str:
        return self.label or self.param

    def numeric_levels(self) -> list[float]:
        """Levels as floats for plotting; ``None`` (disabled) becomes 0."""
        out = []
        for v in self.levels:
            if v is None:
                out.append(0.0)
            elif self.param == "quantize_bits":
                # More bits = finer = less mismatch. Plot the resolution itself.
                out.append(0.0 if v is None else 1.0 / (2**v))
            else:
                out.append(float(v))
        return out


def default_sweeps() -> list[Sweep]:
    """The four experiments, as eight single-mechanism sweeps."""
    return [
        # -- Experiment 1: measurement noise ------------------------------
        Sweep(
            name="encoder_noise", kind="sensor", param="theta_sigma",
            levels=[0.0, 1e-4, 5e-4, 2e-3],
            label="encoder noise $\\sigma_\\theta$ [rad]", experiment=1,
        ),
        Sweep(
            name="quantization", kind="sensor", param="quantize_bits",
            levels=[None, 14, 12, 10],
            label="encoder resolution [bits]", experiment=1,
        ),
        Sweep(
            name="dropout", kind="sensor", param="dropout_prob",
            levels=[0.0, 0.05, 0.15, 0.30],
            label="dropped-sample probability", experiment=1,
        ),
        Sweep(
            name="latency", kind="sensor", param="latency_steps",
            levels=[0, 1, 2, 4],
            label="sensor latency [model steps]", experiment=1,
        ),
        # -- Experiment 2: nonlinear friction -----------------------------
        Sweep(
            name="stribeck", kind="plant", target="stribeck", param="excess",
            levels=[0.0, 0.5, 1.5, 4.0], fixed={"v_stribeck": 0.05},
            label="Stribeck excess $\\mu_s-\\mu_c$ [N$\\cdot$m]", experiment=2,
        ),
        Sweep(
            name="position_friction", kind="plant", target="position_friction",
            param="amplitude", levels=[0.0, 0.3, 1.0, 3.0], fixed={"period": 0.8},
            label="position-dependent friction [N$\\cdot$m]", experiment=2,
        ),
        # -- Experiment 3: compliance -------------------------------------
        Sweep(
            name="compliance", kind="plant", target="compliance", param="k_cubic",
            levels=[0.0, 0.2, 0.6, 2.0],
            label="cubic stiffness $k_3$ [N$\\cdot$m/rad$^3$]", experiment=3,
        ),
        # -- Experiment 4: time-varying dynamics --------------------------
        Sweep(
            name="drift", kind="plant", target="drift", param="friction_rate",
            levels=[0.0, 0.05, 0.15, 0.40], fixed={"mode": "linear"},
            label="friction drift rate [1/s]", experiment=4,
        ),
    ]


EXPERIMENT_TITLES = {
    1: "Measurement noise",
    2: "Nonlinear friction",
    3: "Compliance",
    4: "Time-varying dynamics",
}


@dataclass
class StudyConfig:
    checkpoint: str = "runs/latent_mechanics/base/best.pt"
    stage1_config: str = "configs/latent_mechanics.yaml"
    online_config: str = "configs/online_adaptation.yaml"
    # Doors are re-simulated for plant sweeps, so they come from the sampler
    # rather than the cached dataset. These are the Stage-1 held-out doors.
    n_train_doors: int = 48
    n_doors: int = 8
    episodes_per_door: int = 5
    episode_seconds: float = 6.0
    frame_skip: int = 10
    seed: int = 0
    # Which sweeps to run; empty means all.
    only: tuple[str, ...] = ()
    out_dir: str = "runs/latent_mechanics/base/mismatch"
    device: str = "cpu"
    rolling_window: int = 200
    latent_init: str = "medoid"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def load_config(path: str | Path | None) -> StudyConfig:
    if path is None or not Path(path).exists():
        return StudyConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(StudyConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)}; valid: {sorted(known)}")
    if "only" in raw and isinstance(raw["only"], list):
        raw["only"] = tuple(raw["only"])
    return StudyConfig(**raw)


__all__ = ["Sweep", "StudyConfig", "default_sweeps", "load_config", "EXPERIMENT_TITLES"]
