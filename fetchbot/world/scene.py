"""Backend-neutral scene description produced by :class:`SceneRandomizer`.

A :class:`SceneSpec` is plain data (boxes, cylinders, objects, drop zones, start
pose). Any simulator backend (PyBullet today, Isaac Sim later) can build it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# RGBA colors of the graspable objects. These are the only strongly saturated
# colors in the scene, so a simple HSV detector can find them.
OBJECT_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.08, 0.08, 1.0),
    "green": (0.10, 0.70, 0.15, 1.0),
    "blue": (0.10, 0.25, 0.90, 1.0),
    "yellow": (0.90, 0.80, 0.05, 1.0),
}


@dataclass(frozen=True)
class BoxSpec:
    """Static axis-aligned box obstacle (wall, shelf, clutter). Sizes are full extents."""

    name: str
    center: tuple[float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float, float] = (0.72, 0.72, 0.70, 1.0)


@dataclass(frozen=True)
class CylinderSpec:
    """Static upright cylinder obstacle (bin, plant pot, ...)."""

    name: str
    center: tuple[float, float]
    radius: float
    height: float
    color: tuple[float, float, float, float] = (0.55, 0.55, 0.58, 1.0)


@dataclass(frozen=True)
class ObjectSpec:
    """A small graspable object, e.g. a red box or a blue mug."""

    name: str
    color: str  # key into OBJECT_COLORS
    shape: str  # "box" | "cylinder" | "ball"
    position: tuple[float, float]

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(frozen=True)
class ZoneSpec:
    """A named drop zone painted on the floor (destinations such as 'dock')."""

    name: str
    center: tuple[float, float]
    radius: float = 0.55
    color: tuple[float, float, float, float] = (0.62, 0.58, 0.66, 1.0)


@dataclass(frozen=True)
class RegionSpec:
    """A named axis-aligned region (room) of the arena, e.g. ``'west room'``."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass(frozen=True)
class ActorSpec:
    """A dynamic actor (a 'person') walking back and forth between waypoints."""

    name: str
    waypoints: tuple[tuple[float, float], ...]
    speed: float = 0.35
    radius: float = 0.22


@dataclass
class SceneSpec:
    """Complete description of one randomized arena."""

    seed: int
    size: tuple[float, float]
    boxes: list[BoxSpec] = field(default_factory=list)
    cylinders: list[CylinderSpec] = field(default_factory=list)
    objects: list[ObjectSpec] = field(default_factory=list)
    zones: list[ZoneSpec] = field(default_factory=list)
    actors: list[ActorSpec] = field(default_factory=list)
    regions: list[RegionSpec] = field(default_factory=list)
    robot_start: tuple[float, float, float] = (1.0, 1.0, 0.0)

    def zone(self, name: str) -> ZoneSpec:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(f"no drop zone named {name!r}; known: {[z.name for z in self.zones]}")

    def region(self, name: str) -> RegionSpec | None:
        return next((r for r in self.regions if r.name == name), None)

    def find_objects(self, color: str | None = None, shape: str | None = None) -> list[ObjectSpec]:
        return [
            o
            for o in self.objects
            if (color is None or o.color == color) and (shape is None or o.shape == shape)
        ]
