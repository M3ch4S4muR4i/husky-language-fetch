"""Lift image-space detections into the map frame using depth + camera pose."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fetchbot.types import Detection, Pose2D


def backproject(frame, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """World (x, y, z) of pixels (u, v) of a :class:`~fetchbot.data.CameraFrame`."""
    h, w = frame.depth.shape
    z = frame.depth[vs, us]
    cx, cy = w / 2.0, h / 2.0
    xr = (us + 0.5 - cx) / frame.fx * z
    yu = -(vs + 0.5 - cy) / frame.fy * z
    return (
        frame.position[None, :]
        + z[:, None] * frame.forward[None, :]
        + xr[:, None] * frame.right[None, :]
        + yu[:, None] * frame.up[None, :]
    )


def localize(
    detection: Detection, mask: np.ndarray, frame, max_range: float = 5.0
) -> Detection | None:
    """Return ``detection`` with ``map_pose`` set to the median 3D point of its mask.

    Points beyond ``max_range`` are discarded (far blobs are unreliable). Returns
    ``None`` if no valid depth remains.
    """
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return None
    pts = backproject(frame, us, vs)
    rng = np.linalg.norm(pts[:, :2] - frame.position[None, :2], axis=1)
    pts = pts[rng < max_range]
    if pts.shape[0] < 4:
        return None
    x, y = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    return replace(detection, map_pose=Pose2D(x, y))
