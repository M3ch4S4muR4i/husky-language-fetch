"""Camera-coverage layer: which free cells has the robot's camera actually looked at?

Lidar sees 360 degrees and 6 m, the camera only ~75 degrees and (for reliable color
detection of a 15 cm object) ~3 m. So "no lidar frontiers left" does not mean "every
object has been seen". This layer marks grid cells inside the camera frustum that are
in line of sight (using the current lidar scan as a visibility proxy), and the
:class:`~fetchbot.mapping.FrontierExplorer` sends the robot to unseen free areas once
the ordinary frontiers are gone.
"""

from __future__ import annotations

import math

import numpy as np

from fetchbot.mapping.occupancy import OccupancyGrid


class CameraCoverage:
    """Boolean grid of cells observed by the camera."""

    def __init__(
        self,
        grid: OccupancyGrid,
        hfov_deg: float = 75.0,
        max_range: float = 3.0,
        min_range: float = 0.5,
    ) -> None:
        self.grid = grid
        self.half_fov = math.radians(hfov_deg) / 2
        self.max_range = max_range
        self.min_range = min_range
        self.seen = np.zeros((grid.ny, grid.nx), dtype=bool)

    def update(self, scan) -> None:
        """Mark cells visible from ``scan.origin`` (robot pose) within the camera frustum."""
        g, pose = self.grid, scan.origin
        r0, c0 = g.world_to_cell(pose.x, pose.y)
        k = int(math.ceil(self.max_range / g.resolution))
        rows = np.arange(max(r0 - k, 0), min(r0 + k + 1, g.ny))
        cols = np.arange(max(c0 - k, 0), min(c0 + k + 1, g.nx))
        if rows.size == 0 or cols.size == 0:
            return
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        x = g.origin[0] + (cc + 0.5) * g.resolution
        y = g.origin[1] + (rr + 0.5) * g.resolution
        dx, dy = x - pose.x, y - pose.y
        d = np.hypot(dx, dy)
        bearing = (np.arctan2(dy, dx) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
        in_fov = (np.abs(bearing) <= self.half_fov) & (d <= self.max_range) & (d >= self.min_range)
        # Line of sight: closer than the lidar return along the same bearing.
        n = scan.angles.size
        idx = np.round((bearing + math.pi) / (2 * math.pi) * n).astype(int) % n
        visible = d <= np.where(np.isfinite(scan.ranges), scan.ranges, np.inf)[idx] + 0.05
        self.seen[rr[in_fov & visible], cc[in_fov & visible]] = True

    def fraction_seen(self) -> float:
        free = self.grid.free
        return float((self.seen & free).sum() / max(1, free.sum()))
