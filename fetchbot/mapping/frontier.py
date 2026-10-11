"""Frontier exploration for unknown environments."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from fetchbot.mapping.coverage import CameraCoverage
from fetchbot.mapping.occupancy import OccupancyGrid, shifted
from fetchbot.types import Pose2D


@dataclass(frozen=True)
class Frontier:
    """A connected cluster of frontier cells."""

    cells: tuple[tuple[int, int], ...]
    centroid: tuple[float, float]  # world (x, y)
    goal: tuple[float, float]  # reachable world (x, y) inside the cluster
    path_distance: float  # BFS distance from the robot (m)

    @property
    def size(self) -> int:
        return len(self.cells)


class FrontierExplorer:
    """Finds boundaries between known-free and unknown cells and picks the next goal.

    Frontier cells are free, not inside the inflated obstacle zone, and touch at
    least one unknown cell. They are clustered (8-connectivity); each cluster is
    scored by BFS travel distance from the robot minus a bonus for its size.
    Goals that were reached, went stale or could not be reached are blacklisted.

    With a :class:`CameraCoverage` attached, free cells the camera has not looked
    at yet also count as exploration targets (see ``strategy``), because lidar
    "seeing" an area does not mean the camera has seen the objects in it.
    """

    def __init__(
        self,
        min_frontier_cells: int = 4,
        safe_distance_cells: float = 2.5,
        size_weight: float = 0.04,
        coverage: CameraCoverage | None = None,
        min_coverage_cells: int = 8,
        strategy: str = "union",
    ) -> None:
        self.min_frontier_cells = min_frontier_cells
        self.safe_distance_cells = safe_distance_cells
        self.size_weight = size_weight
        self.coverage = coverage
        self.min_coverage_cells = min_coverage_cells
        # "union": lidar frontiers and camera-unseen free areas are explored together,
        # nearest first (default; 40/40 vs 39/40 successes on seeds 0-19 and 100-119).
        # "frontier_first": lidar frontiers first, camera-unseen areas only afterwards.
        self.strategy = strategy
        self.grid: OccupancyGrid | None = None
        self.frontiers: list[Frontier] = []
        self.frontier_mask: np.ndarray | None = None
        self.coverage_mask: np.ndarray | None = None
        self.mode = "frontier"  # "frontier", "coverage" or "frontier+coverage" (last goal)
        self.blacklist: list[tuple[float, float]] = []

    def update(self, occupancy_grid: OccupancyGrid) -> None:
        """Ingest the latest occupancy grid."""
        self.grid = occupancy_grid

    def blacklist_goal(self, xy: tuple[float, float]) -> None:
        self.blacklist.append(xy)

    def compute_frontier_mask(self) -> np.ndarray:
        g = self.grid
        assert g is not None
        free = g.free
        unknown = g.unknown
        safe = g.obstacle_distance(self.safe_distance_cells + 1) > self.safe_distance_cells
        touch = np.zeros_like(free)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            touch |= shifted(unknown, dy, dx)
        self.frontier_mask = free & safe & touch
        if self.coverage is not None:
            self.coverage_mask = free & safe & ~self.coverage.seen
            if self.strategy == "union":
                self.coverage_mask |= self.frontier_mask
        return self.frontier_mask

    def is_goal_stale(self, goal: Pose2D) -> bool:
        """True once the area around ``goal`` no longer needs exploring."""
        self.compute_frontier_mask()
        mask = self.frontier_mask if self.mode == "frontier" else self.coverage_mask
        if mask is None or self.grid is None:
            return False
        r, c = self.grid.world_to_cell(goal.x, goal.y)
        return not mask[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3].any()

    def next_goal(self, robot_pose: Pose2D) -> Pose2D | None:
        """Return the best exploration goal, or ``None`` when exploration is complete.

        ``"union"``: lidar frontiers and camera-unseen free clusters compete together.
        ``"frontier_first"``: lidar frontiers first, camera-unseen clusters afterwards.
        """
        if self.grid is None:
            return None
        if self.strategy == "union" and self.coverage is not None:
            self.mode = "frontier+coverage"
            return self._best(self.find_frontiers(robot_pose, use_coverage=True))
        self.mode = "frontier"
        goal = self._best(self.find_frontiers(robot_pose))
        if goal is None and self.coverage is not None:
            self.mode = "coverage"
            goal = self._best(self.find_frontiers(robot_pose, use_coverage=True))
        return goal

    def _best(self, frontiers: list[Frontier]) -> Pose2D | None:
        self.frontiers = frontiers
        best, best_score = None, math.inf
        for f in self.frontiers:
            if any(math.dist(f.goal, b) < 0.6 for b in self.blacklist):
                continue
            score = f.path_distance - self.size_weight * min(f.size, 60)
            if score < best_score:
                best, best_score = f, score
        if best is None:
            return None
        return Pose2D(best.goal[0], best.goal[1])

    def find_frontiers(self, robot_pose: Pose2D, use_coverage: bool = False) -> list[Frontier]:
        g = self.grid
        assert g is not None
        mask = self.compute_frontier_mask()
        min_cells = self.min_frontier_cells
        if use_coverage:
            mask, min_cells = self.coverage_mask, self.min_coverage_cells
        dist = self._bfs_distance(robot_pose)
        seen = np.zeros_like(mask)
        out: list[Frontier] = []
        ys, xs = np.nonzero(mask)
        for r0, c0 in zip(ys.tolist(), xs.tolist(), strict=True):
            if seen[r0, c0]:
                continue
            cluster = []
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                cluster.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if g.in_bounds(rr, cc) and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
            if len(cluster) < min_cells:
                continue
            reachable = [(r, c) for r, c in cluster if np.isfinite(dist[r, c])]
            if not reachable:
                continue
            cr = sum(r for r, _ in cluster) / len(cluster)
            cc = sum(c for _, c in cluster) / len(cluster)
            gr, gc = min(reachable, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)
            out.append(
                Frontier(
                    tuple(cluster),
                    g.cell_to_world(int(round(cr)), int(round(cc))),
                    g.cell_to_world(gr, gc),
                    float(dist[gr, gc]) * g.resolution,
                )
            )
        return out

    def _bfs_distance(self, robot_pose: Pose2D) -> np.ndarray:
        """8-connected BFS distance (cells) through safe known-free space."""
        g = self.grid
        assert g is not None
        passable = g.free & (g.obstacle_distance(3.0) > 2.0)
        dist = np.full(passable.shape, np.inf, dtype=np.float32)
        r0, c0 = g.world_to_cell(robot_pose.x, robot_pose.y)
        if not g.in_bounds(r0, c0):
            return dist
        dist[r0, c0] = 0.0
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc, w in (
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, 1.414),
                (1, -1, 1.414),
                (-1, 1, 1.414),
                (-1, -1, 1.414),
            ):
                rr, cc = r + dr, c + dc
                if g.in_bounds(rr, cc) and passable[rr, cc] and dist[rr, cc] > d + w + 1e-6:
                    dist[rr, cc] = d + w
                    q.append((rr, cc))
        return dist
