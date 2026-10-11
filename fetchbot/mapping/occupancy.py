"""Log-odds occupancy grid built from lidar scans.

Odometry in the simulator is ground truth, so this is *mapping with known
poses* (not full SLAM): each scan is ray-traced into the grid at the true pose.
A real robot would get its pose from slam_toolbox instead.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


@lru_cache(maxsize=16)
def disk_offsets(radius_cells: float) -> tuple[tuple[int, int, float], ...]:
    """All integer (dy, dx, dist) offsets with dist <= radius_cells."""
    r = int(math.ceil(radius_cells))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = math.hypot(dx, dy)
            if d <= radius_cells:
                out.append((dy, dx, d))
    return tuple(out)


def shifted(a: np.ndarray, dy: int, dx: int, fill=False) -> np.ndarray:
    """Return ``a`` shifted by (dy, dx), padding with ``fill`` (out[y, x] = a[y-dy, x-dx])."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


class OccupancyGrid:
    """2D grid in the map frame. Row index = y, column index = x."""

    def __init__(
        self,
        width_m: float,
        height_m: float,
        resolution: float = 0.1,
        origin: tuple[float, float] = (0.0, 0.0),
        l_occ: float = 0.9,
        l_free: float = -0.45,
        l_clamp: float = 2.5,
    ) -> None:
        self.resolution = resolution
        self.origin = origin
        self.nx = int(math.ceil(width_m / resolution))
        self.ny = int(math.ceil(height_m / resolution))
        self.logodds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.l_occ, self.l_free, self.l_clamp = l_occ, l_free, l_clamp
        self.occ_thresh, self.free_thresh = 0.7, -0.5
        #: Cells observed free at some point; see :attr:`transient`.
        self.seen_free = np.zeros((self.ny, self.nx), dtype=bool)
        self.version = 0  # bumped on every update; used to cache derived layers
        self._dist_cache: dict[tuple[float, bool], tuple[int, np.ndarray]] = {}

    # ------------------------------------------------------------ geometry
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return (row, col) for world (x, y)."""
        return (
            int(math.floor((y - self.origin[1]) / self.resolution)),
            int(math.floor((x - self.origin[0]) / self.resolution)),
        )

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.ny and 0 <= col < self.nx

    # ------------------------------------------------------------- updates
    def update_from_scan(self, scan) -> None:
        """Ray-trace a :class:`~fetchbot.data.LaserScan` into the grid."""
        ox, oy = scan.origin.x, scan.origin.y
        ends = scan.endpoints()
        hit = np.isfinite(scan.ranges)
        step = self.resolution * 0.5
        lengths = np.hypot(ends[:, 0] - ox, ends[:, 1] - oy)
        n = int(np.max(lengths) / step) + 1
        t = np.arange(n) * step  # distances along each beam
        # Free cells: samples strictly before the endpoint (minus half a cell for hits).
        stop = np.where(hit, lengths - self.resolution * 0.75, lengths)
        mask = t[None, :] < stop[:, None]
        dirs = (ends - [ox, oy]) / np.maximum(lengths, 1e-6)[:, None]
        px = ox + dirs[:, 0:1] * t[None, :]
        py = oy + dirs[:, 1:2] * t[None, :]
        free_cells = self._cells(px[mask], py[mask])
        occ_cells = self._cells(ends[hit, 0], ends[hit, 1])
        if free_cells.size:
            self.logodds.ravel()[free_cells] += self.l_free
        if occ_cells.size:
            self.logodds.ravel()[occ_cells] += self.l_occ  # hits outweigh the free update
        np.clip(self.logodds, -self.l_clamp, self.l_clamp, out=self.logodds)
        self.seen_free |= self.logodds < self.free_thresh
        self.version += 1

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = np.floor((xs - self.origin[0]) / self.resolution).astype(int)
        rows = np.floor((ys - self.origin[1]) / self.resolution).astype(int)
        ok = (rows >= 0) & (rows < self.ny) & (cols >= 0) & (cols < self.nx)
        return np.unique(rows[ok] * self.nx + cols[ok])

    # -------------------------------------------------------------- queries
    @property
    def occupied(self) -> np.ndarray:
        return self.logodds > self.occ_thresh

    @property
    def free(self) -> np.ndarray:
        return self.logodds < self.free_thresh

    @property
    def unknown(self) -> np.ndarray:
        return ~(self.occupied | self.free)

    def states(self) -> np.ndarray:
        """Grid of UNKNOWN / FREE / OCCUPIED codes."""
        s = np.full(self.logodds.shape, UNKNOWN, dtype=np.int8)
        s[self.free] = FREE
        s[self.occupied] = OCCUPIED
        return s

    @property
    def transient(self) -> np.ndarray:
        """Temporary obstacles: occupied cells that were seen free before (something moved
        in, e.g. the walking person). Walls and furniture are never seen free."""
        return self.occupied & self.seen_free

    def obstacle_distance(
        self, max_cells: float = 8.0, ignore_transient: bool = False
    ) -> np.ndarray:
        """Euclidean distance (in cells) to the nearest occupied cell, capped at ``max_cells``;
        ``ignore_transient`` leaves out the :attr:`transient` cells."""
        key = (max_cells, ignore_transient)
        cached = self._dist_cache.get(key)
        if cached is not None and cached[0] == self.version:
            return cached[1]
        occ = self.occupied & ~self.seen_free if ignore_transient else self.occupied
        dist = np.full(occ.shape, max_cells, dtype=np.float32)
        if occ.any():
            for dy, dx, d in disk_offsets(max_cells):
                m = shifted(occ, dy, dx)
                np.minimum(dist, np.where(m, d, max_cells), out=dist)
        self._dist_cache[key] = (self.version, dist)
        return dist

    def known_fraction(self) -> float:
        return float(1.0 - self.unknown.mean())
