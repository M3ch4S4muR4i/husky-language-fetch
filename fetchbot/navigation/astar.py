"""A* path planning on a 2D cost grid (8-connected)."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

SQRT2 = math.sqrt(2.0)
_MOVES = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
)


def astar(
    cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], goal_radius: float = 0.0
) -> list[tuple[int, int]] | None:
    """Shortest path from ``start`` to ``goal`` (row, col) over ``cost``.

    ``cost[r, c]`` is the per-meter traversal cost of a cell (>= 1); ``inf``
    marks a lethal cell. The start cell is always allowed so the robot can
    leave an inflated zone. With ``goal_radius > 0`` (in cells) any free cell
    within that distance of ``goal`` is accepted, so the goal itself may sit in
    an obstacle (e.g. "get within 0.6 m of the object"). Returns the list of
    cells, or ``None`` if unreachable. The octile-distance heuristic minus
    ``goal_radius`` stays admissible because cost >= 1.
    """
    ny, nx = cost.shape
    if goal_radius <= 0 and (
        not (0 <= goal[0] < ny and 0 <= goal[1] < nx) or not np.isfinite(cost[goal])
    ):
        return None
    r2 = goal_radius * goal_radius

    def at_goal(r: int, c: int) -> bool:
        return (r - goal[0]) ** 2 + (c - goal[1]) ** 2 <= r2 if goal_radius > 0 else (r, c) == goal

    if at_goal(*start):
        return [start]

    def h(r: int, c: int) -> float:
        dr, dc = abs(r - goal[0]), abs(c - goal[1])
        return max(0.0, (dr + dc) + (SQRT2 - 2.0) * min(dr, dc) - goal_radius)

    g = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = [(h(*start), 0.0, start)]
    closed: set[tuple[int, int]] = set()
    cost_l = cost.tolist()  # python lists are much faster to index in the inner loop
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if at_goal(*cur):
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        for dr, dc, step in _MOVES:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < ny and 0 <= cc < nx):
                continue
            cell_cost = cost_l[rr][cc]
            if cell_cost == math.inf:
                continue
            if dr and dc and (cost_l[r][cc] == math.inf or cost_l[rr][c] == math.inf):
                continue  # no corner cutting
            ng = gc + step * cell_cost
            nxt = (rr, cc)
            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + h(rr, cc), ng, nxt))
    return None


def nearest_free(
    cost: np.ndarray, cell: tuple[int, int], max_radius: int = 12
) -> tuple[int, int] | None:
    """Closest non-lethal cell to ``cell`` (BFS), or ``None`` within ``max_radius``."""
    ny, nx = cost.shape
    if 0 <= cell[0] < ny and 0 <= cell[1] < nx and np.isfinite(cost[cell]):
        return cell
    seen = {cell}
    q = deque([cell])
    while q:
        r, c = q.popleft()
        if abs(r - cell[0]) > max_radius or abs(c - cell[1]) > max_radius:
            continue
        for dr, dc, _ in _MOVES:
            nxt = (r + dr, c + dc)
            if nxt in seen or not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx):
                continue
            if np.isfinite(cost[nxt]):
                return nxt
            seen.add(nxt)
            q.append(nxt)
    return None


def build_costmap(
    obstacle_dist_cells: np.ndarray,
    unknown: np.ndarray,
    lethal_cells: float,
    inflation_cells: float,
    unknown_cost: float = 2.0,
    inflation_weight: float = 25.0,
) -> np.ndarray:
    """Turn an obstacle-distance field into an A* cost grid.

    * distance <= ``lethal_cells`` -> ``inf`` (robot footprint would hit an obstacle)
    * up to ``inflation_cells`` -> exponentially decaying extra cost (keeps clearance)
    * unknown cells -> ``unknown_cost`` (optimistic: plan through, replan when seen)
    """
    d = obstacle_dist_cells
    cost = np.ones(d.shape, dtype=np.float64)
    band = (d > lethal_cells) & (d < inflation_cells)
    cost[band] += inflation_weight * np.exp(-1.0 * (d[band] - lethal_cells))
    cost[unknown] *= unknown_cost
    cost[d <= lethal_cells] = np.inf
    return cost
