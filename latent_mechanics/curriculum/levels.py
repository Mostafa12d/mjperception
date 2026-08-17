"""The curriculum, and the fixed evaluation suite.

Seven levels of increasing mechanical diversity: the first two vary parameter
spread within one mechanism, the rest add mechanisms one at a time.

The training budget is held FIXED -- same instance count, same episodes each --
so only the mixture changes and "more diversity helps" cannot be confused with
"more data helps". The intended consequence is that higher levels see fewer
instances of any given family.

The evaluation suite is drawn once from a dedicated seed and is byte-identical
for every level, packed as each level's held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Families that appear in the evaluation suite. ``door_narrow`` is deliberately
# excluded: it is a training-side device for level 1, not a mechanism to test.
EVAL_FAMILIES = ("door", "nonlinear_hinge", "soft_close", "drawer", "bifold", "laptop")


@dataclass(frozen=True)
class Level:
    index: int
    name: str
    families: tuple[str, ...]
    description: str

    @property
    def n_families(self) -> int:
        return len(self.families)

    def label(self) -> str:
        return f"L{self.index}"


CURRICULUM: tuple[Level, ...] = (
    Level(1, "narrow_doors", ("door_narrow",),
          "revolute doors, narrow parameter ranges"),
    Level(2, "wide_doors", ("door",),
          "revolute doors, full friction/inertia/stiffness spread"),
    Level(3, "plus_nonlinear", ("door", "nonlinear_hinge"),
          "+ nonlinear (Stribeck, position-dependent) hinges"),
    Level(4, "plus_drawer", ("door", "nonlinear_hinge", "drawer"),
          "+ prismatic drawers (different physical dimension)"),
    Level(5, "plus_softclose", ("door", "nonlinear_hinge", "drawer", "soft_close"),
          "+ soft-close dampers"),
    Level(6, "plus_bifold",
          ("door", "nonlinear_hinge", "drawer", "soft_close", "bifold"),
          "+ two-link bifold cabinets (partially observed)"),
    Level(7, "plus_laptop",
          ("door", "nonlinear_hinge", "drawer", "soft_close", "bifold", "laptop"),
          "+ laptop hinges (1500x smaller inertia)"),
)


@dataclass
class CurriculumConfig:
    """Budgets. Everything here is held constant across levels by design."""

    # Total training instances per level, split as evenly as possible across
    # that level's families. Fixed, so diversity is the only thing that varies.
    train_instances: int = 48
    episodes_per_train_instance: int = 5
    # The fixed evaluation suite.
    eval_instances_per_family: int = 10
    episodes_per_eval_instance: int = 4
    episode_seconds: float = 6.0
    frame_skip: int = 10
    epochs: int = 30
    train_seed: int = 100
    eval_seed: int = 999  # deliberately far from any training seed
    out_dir: str = "runs/latent_mechanics/curriculum"
    latent_init: str = "medoid"
    rolling_window: int = 200


def split_budget(level: Level, total: int) -> dict[str, int]:
    """Divide the fixed instance budget across a level's families. Remainders go to
    the earliest, so the door count does not jitter between levels."""
    n = len(level.families)
    base, rem = divmod(total, n)
    return {f: base + (1 if i < rem else 0) for i, f in enumerate(level.families)}


def curriculum_table() -> str:
    lines = [f"  {'level':>5} {'families':>2}  {'composition':<62} description"]
    for lv in CURRICULUM:
        comp = split_budget(lv, CurriculumConfig().train_instances)
        comp_s = ", ".join(f"{k}x{v}" for k, v in comp.items())
        lines.append(f"  {lv.label():>5} {lv.n_families:>2}  {comp_s:<62} {lv.description}")
    return "\n".join(lines)


__all__ = ["Level", "CURRICULUM", "CurriculumConfig", "EVAL_FAMILIES",
           "split_budget", "curriculum_table"]
