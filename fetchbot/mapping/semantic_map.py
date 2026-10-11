"""Semantic layer over the occupancy grid: labeled object instances in the map frame."""

from __future__ import annotations

import math
from dataclasses import replace

from fetchbot.types import Detection, Pose2D


class SemanticMap:
    """Stores labeled object instances in the map frame.

    Detections arrive already projected into the map (pixel + depth + camera
    pose, see :mod:`fetchbot.perception.projection`). A detection within
    ``fuse_radius`` of an existing instance with the same label refines that
    instance (running mean); otherwise a new instance is created.
    """

    def __init__(self, resolution_m: float = 0.05, fuse_radius: float = 0.6) -> None:
        self.resolution_m = resolution_m
        self.fuse_radius = fuse_radius
        self._objects: dict[str, Detection] = {}
        self._counts: dict[str, int] = {}
        self._shape_votes: dict[str, dict[str, int]] = {}
        self._next_id = 0

    def add_detection(self, detection: Detection, camera_pose: Pose2D | None = None) -> str:
        """Fuse a map-projected detection; return the object instance id."""
        if detection.map_pose is None:
            raise ValueError("detection has no map_pose; project it first")
        p = detection.map_pose
        for oid, obj in self._objects.items():
            q = obj.map_pose
            if (
                obj.label == detection.label
                and q is not None
                and math.hypot(p.x - q.x, p.y - q.y) < self.fuse_radius
            ):
                n = self._counts[oid]
                fused = Pose2D((q.x * n + p.x) / (n + 1), (q.y * n + p.y) / (n + 1))
                self._objects[oid] = replace(
                    detection, map_pose=fused, score=max(obj.score, detection.score)
                )
                self._counts[oid] = n + 1
                self._vote(oid, detection.shape)
                return oid
        oid = f"{detection.label.replace(' ', '_')}_{self._next_id}"
        self._next_id += 1
        self._objects[oid] = detection
        self._counts[oid] = 1
        self._shape_votes[oid] = {}
        self._vote(oid, detection.shape)
        return oid

    def _vote(self, oid: str, shape: str | None) -> None:
        if shape is not None:
            votes = self._shape_votes.setdefault(oid, {})
            votes[shape] = votes.get(shape, 0) + 1

    def shape_of(self, object_id: str, min_votes: int = 2) -> str | None:
        """Majority shape of an instance over all its classified views, or None if it
        has fewer than ``min_votes`` classified views."""
        votes = self._shape_votes.get(object_id, {})
        if sum(votes.values()) < min_votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def shape_votes(self, object_id: str) -> dict[str, int]:
        return dict(self._shape_votes.get(object_id, {}))

    def find(self, label: str, min_sightings: int = 1) -> list[Detection]:
        """Return known instances whose label matches ``label`` (best-observed first)."""
        ids = [
            oid
            for oid, d in self._objects.items()
            if d.label == label and self._counts[oid] >= min_sightings
        ]
        ids.sort(key=lambda oid: -self._counts[oid])
        return [self._objects[oid] for oid in ids]

    def find_ids(self, label: str, min_sightings: int = 1) -> list[str]:
        ids = [
            oid
            for oid, d in self._objects.items()
            if d.label == label and self._counts[oid] >= min_sightings
        ]
        return sorted(ids, key=lambda oid: -self._counts[oid])

    def get(self, object_id: str) -> Detection:
        return self._objects[object_id]

    def sightings(self, object_id: str) -> int:
        return self._counts.get(object_id, 0)

    def remove(self, object_id: str) -> None:
        """Forget an instance (e.g. it was relocated or picked up)."""
        self._objects.pop(object_id, None)
        self._counts.pop(object_id, None)
        self._shape_votes.pop(object_id, None)

    def items(self):
        return list(self._objects.items())

    def __len__(self) -> int:
        return len(self._objects)
