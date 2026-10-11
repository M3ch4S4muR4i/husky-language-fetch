"""Execution of a :class:`~fetchbot.language.schema.TaskProgram`: target selection
(color/shape filters, ``all``, spatial relations) and the step/job sequencing.

Policy (documented in the README):

* Steps run in order. A step that fails stops the program; the result reports how
  many steps succeeded (partial result).
* ``one`` without a superlative: the robot fetches the first matching object it has
  seen (best-observed first). ``in_room`` only filters the candidates.
* Superlatives (``nearest_to``, ``farthest_from``, ``leftmost``, ...) and ``all`` need
  to know every candidate, so the step starts with a **survey**: frontier exploration
  without stopping at the first match, until no reachable frontier is left (for
  ``survey_patience`` s) or ``survey_cap`` s of sim time have passed. Then the best
  candidate (or, for ``all``, every candidate, nearest first) is fetched. Objects seen
  later (e.g. while delivering) are added to an ``all`` step.
* Anchors: drop zones and shelves are known landmarks (the floor plan); ``robot`` is
  the robot's position when the step starts; object anchors ("the box nearest the blue
  cylinder") come from the semantic map.
* Objects already inside the step's destination zone are not candidates.
* A candidate's shape is the majority vote of its classified views (at least 2
  votes); an instance whose shape is still unknown is not a candidate for a step that
  names a shape (the robot keeps looking).

The same :func:`select_candidates` is used with the robot's belief (semantic map)
for acting and with simulator ground truth for scoring.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from fetchbot.language.schema import DISTANCE_RELATIONS, EXTREME_RELATIONS, FetchStep


@dataclass(frozen=True)
class ObjInfo:
    """One object as seen by the selector (belief or ground truth)."""

    oid: str
    color: str | None
    shape: str | None
    x: float
    y: float


@dataclass
class Geometry:
    """Known landmarks of the arena, in the map frame."""

    zones: dict[str, tuple[float, float, float]]  # name -> (x, y, radius)
    shelves: list[tuple[float, float, float, float]]  # (xmin, ymin, xmax, ymax)
    regions: dict[str, tuple[float, float, float, float]]  # room -> (xmin, ymin, xmax, ymax)
    robot_xy: tuple[float, float] = (0.0, 0.0)  # robot position when the step started


def needs_survey(step: FetchStep) -> bool:
    """True if the step must see every candidate before choosing (all / superlative)."""
    return step.quantifier == "all" or step.relation in DISTANCE_RELATIONS + EXTREME_RELATIONS


def _rect_dist(x: float, y: float, r: tuple[float, float, float, float]) -> float:
    dx = max(r[0] - x, 0.0, x - r[2])
    dy = max(r[1] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)


def _in_zone(o: ObjInfo, zone: tuple[float, float, float] | None, margin: float = 0.0) -> bool:
    return zone is not None and math.hypot(o.x - zone[0], o.y - zone[1]) <= zone[2] + margin


def matches(step: FetchStep, o: ObjInfo) -> bool:
    return (step.color is None or o.color == step.color) and (
        step.shape is None or o.shape == step.shape
    )


def anchor_distance(
    step: FetchStep, o: ObjInfo, objects: Sequence[ObjInfo], geo: Geometry
) -> float | None:
    """Distance from ``o`` to the step's anchor, or None if the anchor is unknown."""
    a = step.anchor
    if a in geo.zones:
        zx, zy, _ = geo.zones[a]
        return math.hypot(o.x - zx, o.y - zy)
    if a == "shelf":
        if not geo.shelves:
            return None
        return min(_rect_dist(o.x, o.y, r) for r in geo.shelves)
    if a == "robot":
        return math.hypot(o.x - geo.robot_xy[0], o.y - geo.robot_xy[1])
    ref = step.anchor_object
    if ref is None:
        return None
    color, shape = ref
    anchors = [
        q
        for q in objects
        if q.oid != o.oid
        and (color is None or q.color == color)
        and (shape is None or q.shape == shape)
    ]
    if not anchors:
        return None
    return min(math.hypot(o.x - q.x, o.y - q.y) for q in anchors)


def select_candidates(
    step: FetchStep,
    objects: Sequence[ObjInfo],
    geo: Geometry,
    exclude: Callable[[ObjInfo], bool] | None = None,
) -> list[ObjInfo]:
    """Objects the step refers to, in fetch order.

    ``one`` + superlative -> at most one object (the best); ``one`` otherwise -> every
    match (any of them satisfies the step); ``all`` -> every match, nearest to the robot
    first. Raises ``LookupError`` if the anchor or room is unknown.
    """
    dest = geo.zones.get(step.destination)
    cands = [
        o
        for o in objects
        if matches(step, o) and not _in_zone(o, dest) and not (exclude and exclude(o))
    ]
    if step.relation == "in_room":
        if step.anchor not in geo.regions:
            raise LookupError(f"this arena has no {step.anchor}")
        r = geo.regions[step.anchor]
        cands = [o for o in cands if r[0] <= o.x <= r[2] and r[1] <= o.y <= r[3]]
    rx, ry = geo.robot_xy
    cands.sort(key=lambda o: math.hypot(o.x - rx, o.y - ry))
    if step.relation in DISTANCE_RELATIONS:
        scored = []
        for o in cands:
            d = anchor_distance(step, o, objects, geo)
            if d is None:
                raise LookupError(f"anchor '{step.anchor}' not known")
            scored.append((d, o))
        if not scored:
            return []
        scored.sort(key=lambda t: t[0], reverse=step.relation == "farthest_from")
        return [scored[0][1]]
    if step.relation in EXTREME_RELATIONS:
        if not cands:
            return []
        key = {
            "leftmost": lambda o: o.x,
            "rightmost": lambda o: -o.x,
            "southmost": lambda o: o.y,
            "northmost": lambda o: -o.y,
        }[step.relation]
        return [min(cands, key=key)]
    return cands


@dataclass
class StepRecord:
    """What happened in one program step (robot view + ground-truth check)."""

    index: int
    step: str
    expected: list[str] = field(default_factory=list)  # ground-truth object names
    delivered: list[str] = field(default_factory=list)  # objects released at the zone
    robot_failure: str = ""
    success: bool = False
    cause: str = ""
    t_start: float = 0.0
    t_end: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)
