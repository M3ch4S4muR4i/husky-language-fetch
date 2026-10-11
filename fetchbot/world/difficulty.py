"""Evaluation conditions: knobs that make the randomized arena harder.

The standard arena is what the demo was developed on. The harder conditions were
added afterwards and are reported *as is* (they were not used for tuning), so the
batch numbers show where the simple pipeline breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: Near-miss colors for distractor blocks: each sits just outside one of the
#: ColorDetector hue ranges (red 345-12, yellow 45-70, green 90-160, blue 205-250).
DISTRACTOR_COLORS: dict[str, tuple[float, float, float, float]] = {
    "orange": (1.00, 0.45, 0.05, 1.0),  # hue ~25 deg  (next to red)
    "pink": (0.85, 0.10, 0.45, 1.0),  # hue ~332 deg (next to red)
    "lime": (0.60, 0.80, 0.05, 1.0),  # hue ~76 deg  (between yellow and green)
    "teal": (0.05, 0.65, 0.60, 1.0),  # hue ~175 deg (between green and blue)
    "purple": (0.45, 0.15, 0.85, 1.0),  # hue ~266 deg (next to blue)
}


@dataclass(frozen=True)
class Difficulty:
    """How hard a generated scene / episode is."""

    name: str = "standard"
    clutter: float = 1.0  # multiplier on the number of crates, bins and shelves
    door_width: float = 1.6  # doorway width (m); the Husky is 0.69 m wide
    hide_objects: bool = False  # tuck objects behind shelves/crates, away from the start
    distractors: int = 0  # number of near-miss colored blocks (max 5)
    # Step limit (control ticks at 20 Hz; None = time limit only). The limits below are
    # 1.5 x the original 2400 ticks, scaled with the arena (randomizer.S).
    max_ticks: int | None = None
    require_person: bool = False  # regenerate until a walking person fits (standard: ~half)
    object_set: str = "standard"  # "shapes": same-color objects of different shapes


STANDARD = Difficulty()

#: Named evaluation conditions. Each "hard" knob is isolated, then all are combined.
#: Narrow doors (1.10 m) leave the 0.69 m wide Husky 0.41 m of slack, the same as the
#: 0.80 m doors left the original 0.40 m robot.
CONDITIONS: dict[str, Difficulty] = {
    "standard": STANDARD,
    "clutter": replace(STANDARD, name="clutter", clutter=2.5),
    "narrow_doors": replace(STANDARD, name="narrow_doors", door_width=1.10),
    "hidden": replace(STANDARD, name="hidden", hide_objects=True),
    "distractors": replace(STANDARD, name="distractors", distractors=5),
    "person": replace(STANDARD, name="person", require_person=True),
    "step_limit": replace(STANDARD, name="step_limit", max_ticks=3600),
    "shapes": replace(STANDARD, name="shapes", object_set="shapes"),
    "hard_all": Difficulty(
        name="hard_all",
        clutter=2.5,
        door_width=1.10,
        hide_objects=True,
        distractors=5,
        max_ticks=3600,
        require_person=True,
    ),
}


def get_condition(name: str) -> Difficulty:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"unknown condition {name!r}; choose from {sorted(CONDITIONS)}") from None
