"""Small, dependency-free data types shared across subpackages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pose2D:
    """Planar pose in a named frame (meters, radians)."""

    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"


@dataclass(frozen=True)
class Detection:
    """A single object detection in image space, optionally lifted to the map."""

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    map_pose: Pose2D | None = None
    shape: str | None = None  # "box" | "cylinder" | "ball" from the silhouette, if known


@dataclass(frozen=True)
class FetchCommand:
    """Structured result of parsing a command such as
    'bring the red mug to the kitchen table'."""

    target_object: str
    destination: str
    attributes: tuple[str, ...] = field(default_factory=tuple)
    raw_text: str = ""
