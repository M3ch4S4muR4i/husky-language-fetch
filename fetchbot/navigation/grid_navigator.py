"""In-process navigator: A* on the live occupancy grid + pure-pursuit tracking."""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from fetchbot.navigation.astar import astar, build_costmap, nearest_free
from fetchbot.navigation.base import Navigator
from fetchbot.navigation.controller import PurePursuitController, wrap_angle
from fetchbot.types import Pose2D

if TYPE_CHECKING:
    from fetchbot.data.pybullet_streams import RaycastLidarStream
    from fetchbot.mapping.occupancy import OccupancyGrid
    from fetchbot.robots.base import Robot


class NavStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class GridNavigator(Navigator):
    """Plans with A* on an inflated costmap derived from the occupancy grid and
    tracks the path with pure pursuit.

    Replanning happens periodically (the map keeps growing), immediately when
    the current path crosses a newly lethal cell (``blocked_replans``), and after
    a stuck recovery. A lidar "collision monitor" stops the robot if something
    (e.g. a walking person) is right in front of it.
    """

    def __init__(
        self,
        robot: Robot,
        grid: OccupancyGrid,
        lidar: RaycastLidarStream | None = None,
        robot_radius: float = 0.2,
        inflation_radius: float = 0.6,
        goal_tolerance_m: float = 0.25,
        replan_period_s: float = 1.0,
        plan_fail_timeout_s: float = 8.0,
        monitor_distance: float = 0.42,
        controller: PurePursuitController | None = None,
    ) -> None:
        self.robot = robot
        self.grid = grid
        self.lidar = lidar
        self.lethal_cells = robot_radius / grid.resolution + 0.05
        self.inflation_cells = inflation_radius / grid.resolution
        self.goal_tolerance_m = goal_tolerance_m
        self.replan_period_s = replan_period_s
        self.plan_fail_timeout_s = plan_fail_timeout_s
        self.monitor_distance = monitor_distance  # lidar range (from the center) that stops it
        self.controller = controller or PurePursuitController()
        self.status = NavStatus.IDLE
        self.goal: Pose2D | None = None
        self.tolerance = goal_tolerance_m
        self.path: list[tuple[float, float]] = []
        self._path_cells: list[tuple[int, int]] = []
        self._path_idx = 0
        self._since_plan = math.inf
        self._plan_failures = 0
        self._progress_ref: tuple[float, float, float] | None = None  # x, y, time
        self._time = 0.0
        self._recovery_left = 0.0
        self.stuck_recoveries = 0
        self._costs: dict[bool, tuple[int, np.ndarray]] = {}  # through_transient -> cost
        self.through_transient = False  # current path crosses temporary obstacles
        # Statistics (reported by the evaluation).
        self.plans = 0
        self.blocked_replans = 0
        self.monitor_stops = 0
        self._monitor_hold = 0.0

    # ------------------------------------------------------------- Navigator
    def go_to(self, goal: Pose2D, tolerance: float | None = None) -> None:
        self.goal = goal
        self.tolerance = self.goal_tolerance_m if tolerance is None else tolerance
        self.status = NavStatus.ACTIVE
        self.path, self._path_cells, self._path_idx = [], [], 0
        self._since_plan = math.inf
        self._plan_failures = 0
        self.stuck_recoveries = 0
        self._progress_ref = None
        self._recovery_left = 0.0

    def is_done(self) -> bool:
        return self.status in (NavStatus.SUCCEEDED, NavStatus.FAILED, NavStatus.CANCELED)

    def succeeded(self) -> bool:
        return self.status == NavStatus.SUCCEEDED

    def cancel(self) -> None:
        if self.status == NavStatus.ACTIVE:
            self.status = NavStatus.CANCELED
        self.robot.stop()

    @property
    def active(self) -> bool:
        return self.status == NavStatus.ACTIVE

    # ------------------------------------------------------------- planning
    def costmap(self, through_transient: bool = False) -> np.ndarray:
        """Inflated costmap. ``through_transient`` leaves out temporary obstacles
        (:attr:`OccupancyGrid.transient`, e.g. where the person was last seen)."""
        cached = self._costs.get(through_transient)
        if cached is None or cached[0] != self.grid.version:
            dist = self.grid.obstacle_distance(
                math.ceil(self.inflation_cells) + 1.0, ignore_transient=through_transient
            )
            cost = build_costmap(dist, self.grid.unknown, self.lethal_cells, self.inflation_cells)
            cached = self._costs[through_transient] = (self.grid.version, cost)
        return cached[1]

    def plan(self, start: Pose2D, goal: Pose2D) -> list[tuple[int, int]] | None:
        """A* from ``start`` to ``goal``. If only temporary obstacles block the way (the
        person mapped in a doorway while out of view), plan through them: they expire as
        soon as the lidar sees those cells free again, and the collision monitor stops
        the robot if the person is still there."""
        self.through_transient = False
        s = self.grid.world_to_cell(start.x, start.y)
        if not self.grid.in_bounds(*s):
            return None
        g = self.grid.world_to_cell(goal.x, goal.y)
        radius = 0.9 * self.tolerance / self.grid.resolution
        self.plans += 1
        cost = self.costmap()
        if not np.isfinite(cost[s]):
            cost = self._escape_costmap(cost, s)
        path = self._search(cost, s, g, radius)
        if path is None:
            cost = self.costmap(through_transient=True)
            if np.isfinite(cost[s]):
                path = self._search(cost, s, g, radius)
                self.through_transient = path is not None
        return path

    def _search(self, cost: np.ndarray, s, g, radius: float) -> list[tuple[int, int]] | None:
        if radius >= 1.0:  # accept any free cell inside the goal tolerance
            path = astar(cost, s, g, goal_radius=radius)
            if path is not None:
                return path
        g = nearest_free(cost, g, max_radius=int(0.8 / self.grid.resolution))
        return None if g is None else astar(cost, s, g)

    def _escape_costmap(self, cost: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        """The robot is inside the inflated zone (e.g. a person stepped right next to it):
        make nearby inflated-but-not-occupied cells expensive instead of lethal so the
        planner can lead it out."""
        cost = cost.copy()
        r0, c0 = start
        k = int(round(self.lethal_cells)) + 2
        dist = self.grid.obstacle_distance(math.ceil(self.inflation_cells) + 1.0)
        win = (slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1))
        sub, dsub = cost[win], dist[win]
        sub[~np.isfinite(sub) & (dsub >= 1.0)] = 80.0
        return cost

    def _replan(self, pose: Pose2D) -> bool:
        cells = self.plan(pose, self.goal)
        self._since_plan = 0.0
        if cells is None:
            self._plan_failures += 1
            self.path, self._path_cells, self._path_idx = [], [], 0
            return False
        self._plan_failures = 0
        self._path_cells = cells
        self.path = [self.grid.cell_to_world(r, c) for r, c in cells]
        self._path_idx = 0
        return True

    def _path_blocked(self) -> bool:
        cost = self.costmap(self.through_transient)
        ahead = self._path_cells[self._path_idx + 2 : self._path_idx + 40]
        return any(not np.isfinite(cost[c]) for c in ahead)

    def _front_clearance(self) -> float:
        scan = self.lidar.latest() if self.lidar is not None else None
        if scan is None:
            return math.inf
        front = np.abs(scan.angles) < math.radians(30)
        r = scan.ranges[front]
        return float(np.min(r)) if r.size else math.inf

    # ------------------------------------------------------------- control
    def update(self, dt: float) -> None:
        self._time += dt
        if self.status != NavStatus.ACTIVE or self.goal is None:
            return
        pose = self.robot.get_pose()
        dist_goal = math.hypot(self.goal.x - pose.x, self.goal.y - pose.y)
        if dist_goal <= self.tolerance:
            self.robot.stop()
            self.status = NavStatus.SUCCEEDED
            return

        # Stuck recovery: back up and turn for a moment, then replan.
        if self._recovery_left > 0:
            self._recovery_left -= dt
            self.robot.send_velocity(-0.15, 0.8)
            if self._recovery_left <= 0:
                self._since_plan = math.inf
            return

        self._since_plan += dt
        need = self._since_plan >= (self.replan_period_s if self.path else 0.5)
        if self.path and self._path_blocked():
            self.blocked_replans += 1
            need = True
        if need and not self._replan(pose):
            # No path right now. Dynamic obstacles (a person in a doorway) can cause this
            # temporarily, so keep retrying every 0.5 s and only fail after a timeout.
            self.robot.stop()
            if self._plan_failures * 0.5 >= self.plan_fail_timeout_s:
                self.status = NavStatus.FAILED
            return

        if not self.path:  # waiting for the next replanning attempt
            self.robot.stop()
            return
        target, self._path_idx = self.controller.target_point(pose, self.path, self._path_idx)
        # Speed scaling from obstacle clearance at the robot's cell.
        dist = self.grid.obstacle_distance(math.ceil(self.inflation_cells) + 1.0)
        r, c = self.grid.world_to_cell(pose.x, pose.y)
        clear = float(dist[r, c]) if self.grid.in_bounds(r, c) else 10.0
        scale = float(np.clip((clear - self.lethal_cells) / 3.0, 0.35, 1.0))
        v, w = self.controller.compute(pose, target, dist_goal, scale)

        # Collision monitor: stop for anything right in front (e.g. a person).
        if v > 0 and self._front_clearance() < self.monitor_distance:
            if self._monitor_hold <= 0:
                self.monitor_stops += 1
            self._monitor_hold += dt
            v = 0.0
            if self._monitor_hold > 2.0:  # still blocked: rotate toward the path, replan
                w = math.copysign(
                    0.6, wrap_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)
                )
                self._since_plan = math.inf
        else:
            self._monitor_hold = 0.0
        self.robot.send_velocity(v, w)
        self._check_progress(pose)

    def _check_progress(self, pose: Pose2D) -> None:
        if self._progress_ref is None:
            self._progress_ref = (pose.x, pose.y, self._time)
            return
        x, y, t = self._progress_ref
        if math.hypot(pose.x - x, pose.y - y) > 0.25:
            self._progress_ref = (pose.x, pose.y, self._time)
        elif self._time - t > 6.0:
            self.stuck_recoveries += 1
            self._progress_ref = (pose.x, pose.y, self._time)
            if self.stuck_recoveries > 3:
                self.robot.stop()
                self.status = NavStatus.FAILED
            else:
                self._recovery_left = 1.2
