"""States of the fetch task. Each state does one job and names its successor.

Pipeline: ``explore -> approach -> grab -> deliver -> release -> done``.
A state that gives up records ``ctx.failure`` and jumps to ``done``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fetchbot.navigation.controller import wrap_angle
from fetchbot.types import FetchCommand, Pose2D

if TYPE_CHECKING:  # imported only for type hints to keep the package import light
    from fetchbot.mapping import FrontierExplorer, SemanticMap
    from fetchbot.navigation import Navigator
    from fetchbot.robots import Robot


@dataclass
class TaskContext:
    """Everything a state may read or act on during one fetch episode."""

    robot: Robot
    navigator: Navigator
    semantic_map: SemanticMap
    explorer: FrontierExplorer
    command: FetchCommand
    #: Named destinations in the map frame (e.g. ``{"dock": Pose2D(...)}``).
    destinations: dict[str, Pose2D] = field(default_factory=dict)
    #: Contact sensor: returns the id of a graspable object touching the robot's front.
    #: Contact sensor: ``grasp_sensor(near=(x, y))`` -> object at the bumper near (x, y).
    grasp_sensor: Callable[..., str | None] | None = None
    dt: float = 0.05
    time: float = 0.0
    failure: str | None = None
    events: list[tuple[float, str]] = field(default_factory=list)
    target_id: str | None = None  # semantic-map instance currently being fetched
    grabbed: str | None = None  # simulator object id that was attached
    retry_target_after: float = 0.0  # explore ignores the known target until this time
    approach_rounds: int = 0  # times approach gave up and went back to exploring
    #: Task-program hooks (see fetchbot.tasks.program). ``select_target`` returns the
    #: semantic-map id to fetch next (None if no candidate is known yet); without it the
    #: target is looked up by ``target_key``.
    select_target: Callable[[], str | None] | None = None
    label: str = ""  # human-readable target of the current step
    surveying: bool = False  # explore the whole arena before choosing (all / superlatives)
    survey_deadline: float = float("inf")
    look_around: bool = True  # spin 360 degrees when exploration starts
    #: Where this robot has set objects down, per zone (so a second object delivered to
    #: the same zone is placed beside the first instead of pushing it out).
    placed: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    @property
    def target_label(self) -> str:
        """Human-readable target, e.g. ``'red box'``."""
        if self.label:
            return self.label
        return " ".join((*self.command.attributes, self.command.target_object)).strip()

    def find_target(self, min_sightings: int = 2) -> str | None:
        """Semantic-map id of the object to fetch next, or None."""
        if self.select_target is not None:
            return self.select_target()
        found = self.semantic_map.find_ids(self.target_key, min_sightings)
        return found[0] if found else None

    @property
    def target_key(self) -> str:
        """Semantic-map label to look for. The color detector labels by color, so the
        color word is the key (``'red'``); shape-only commands fall back to the shape."""
        return self.command.attributes[0] if self.command.attributes else self.command.target_object

    def log(self, message: str) -> None:
        self.events.append((round(self.time, 2), message))

    def fail(self, reason: str) -> str:
        self.failure = reason
        self.log(f"FAILED: {reason}")
        self.robot.stop()
        return DoneState.name


class State(ABC):
    """One state of the fetch task.

    ``step`` is called every tick and returns the name of the next state
    (its own name to stay put).
    """

    #: Unique key used by :class:`~fetchbot.tasks.TaskManager` for transitions.
    name: str = "state"

    def on_enter(self, ctx: TaskContext) -> None:
        """Hook run once when the state becomes active."""

    @abstractmethod
    def step(self, ctx: TaskContext) -> str:
        """Advance one tick and return the next state's name."""

    def on_exit(self, ctx: TaskContext) -> None:
        """Hook run once when leaving the state."""


class ExploreState(State):
    """Explore frontiers until the target object appears in the semantic map.

    Starts with a 360-degree look-around, then repeatedly drives to the best
    frontier, looking around again at each one (the camera sees only ~80 degrees).
    """

    name = "explore"
    min_sightings = 2
    spin_rate = 1.2  # rad/s
    goal_check_period = 2.0  # s
    no_frontier_patience = 40.0  # s without any reachable frontier before giving up
    survey_patience = 8.0  # s without a reachable frontier that ends a survey

    def __init__(self) -> None:
        self.no_frontier_time = 0.0
        self.goal: Pose2D | None = None
        self.spin_left = 0.0
        self.since_check = 0.0
        self.first_entry = True
        self.surveyed = False  # a survey of this job finished (no frontier left / cap)

    def on_enter(self, ctx: TaskContext) -> None:
        self.goal = None
        self.since_check = 0.0
        if self.first_entry:
            if ctx.look_around:
                self.spin_left = 2 * math.pi / self.spin_rate
            self.first_entry = False
        mode = "surveying the arena for" if ctx.surveying else "looking for"
        ctx.log(f"explore: {mode} '{ctx.target_label}'")

    def _end_survey(self, ctx: TaskContext, why: str) -> None:
        ctx.surveying = False
        self.no_frontier_time = 0.0
        ctx.log(f"explore: survey done ({why})")

    def step(self, ctx: TaskContext) -> str:
        if ctx.surveying and ctx.time >= ctx.survey_deadline:
            self._end_survey(ctx, "time cap")
        try:
            found = None if ctx.surveying else ctx.find_target(self.min_sightings)
        except LookupError as exc:
            if ctx.surveying:
                found = None
            else:
                return ctx.fail(str(exc))
        if not ctx.surveying and found is None and self.surveyed:
            return ctx.fail(f"explored the arena but found no '{ctx.target_label}'")
        if found and ctx.time >= ctx.retry_target_after:
            ctx.target_id = found
            ctx.navigator.cancel()
            p = ctx.semantic_map.get(found).map_pose
            ctx.log(f"explore: found '{ctx.target_label}' at ({p.x:.2f}, {p.y:.2f})")
            return ApproachState.name
        if self.spin_left > 0:
            self.spin_left -= ctx.dt
            ctx.robot.send_velocity(0.0, self.spin_rate)
            return self.name

        nav = ctx.navigator
        self.since_check += ctx.dt
        stale = False
        if self.goal is not None and self.since_check >= self.goal_check_period:
            self.since_check = 0.0
            stale = ctx.explorer.is_goal_stale(self.goal)  # area already explored
            if stale:
                ctx.explorer.blacklist_goal((self.goal.x, self.goal.y))
        if self.goal is None or nav.is_done() or stale:
            if self.goal is not None and nav.is_done():
                if nav.succeeded():
                    # Arrived: look around, and never pick this exact spot again (some
                    # frontiers, e.g. unknown shelf interiors, can never be resolved).
                    ctx.explorer.blacklist_goal((self.goal.x, self.goal.y))
                    self.spin_left = 2 * math.pi / self.spin_rate
                    self.goal = None
                    return self.name
                ctx.explorer.blacklist_goal((self.goal.x, self.goal.y))
                ctx.log("explore: frontier unreachable, blacklisted")
            self.goal = ctx.explorer.next_goal(ctx.robot.get_pose())
            if self.goal is None:
                # No reachable frontier. This can be temporary (e.g. a person standing in
                # a doorway is mapped as an obstacle), so look around, forget blacklisted
                # goals and retry for a while before giving up.
                self.no_frontier_time += ctx.dt
                if ctx.surveying and self.no_frontier_time > self.survey_patience:
                    self._end_survey(ctx, "no reachable frontier left")
                    self.surveyed = True
                    return self.name
                if self.no_frontier_time > self.no_frontier_patience:
                    return ctx.fail(f"map fully explored but '{ctx.target_label}' not found")
                if self.no_frontier_time % 5.0 < ctx.dt:
                    ctx.explorer.blacklist.clear()
                ctx.robot.send_velocity(0.0, 0.6)
                return self.name
            self.no_frontier_time = 0.0
            ctx.log(f"explore: {ctx.explorer.mode} goal ({self.goal.x:.1f}, {self.goal.y:.1f})")
            nav.go_to(self.goal, tolerance=0.35)
        return self.name


class ApproachState(State):
    """Navigate to a pre-grasp pose a standoff in front of the target object.

    The goal is a free pose on the robot's side of the object (not the object itself),
    so the planner never drives the bumper into it; the grab covers the last gap.
    """

    name = "approach"
    object_radius = 0.1  # m, largest object half-size
    gap = 0.3  # m bumper-to-object clearance where navigation hands over to the grab
    goal_tolerance = 0.15
    max_attempts = 3  # navigation failures before going back to exploring
    max_rounds = 3  # explore/approach rounds before giving up

    def __init__(self) -> None:
        self.attempts = 0
        self.goal_xy: tuple[float, float] | None = None

    @classmethod
    def standoff(cls, ctx: TaskContext) -> float:
        """Base center to object center at the pre-grasp pose (m)."""
        return ctx.robot.front_extent + cls.object_radius + cls.gap

    def on_enter(self, ctx: TaskContext) -> None:
        self.goal_xy = None

    def _target(self, ctx: TaskContext) -> Pose2D | None:
        if ctx.target_id is None or ctx.semantic_map.sightings(ctx.target_id) == 0:
            return None
        return ctx.semantic_map.get(ctx.target_id).map_pose

    def _go(self, ctx: TaskContext, target: Pose2D) -> None:
        pose, standoff = ctx.robot.get_pose(), self.standoff(ctx)
        d = max(math.hypot(pose.x - target.x, pose.y - target.y), 1e-6)
        ux, uy = (pose.x - target.x) / d, (pose.y - target.y) / d
        goal = Pose2D(target.x + standoff * ux, target.y + standoff * uy, math.atan2(-uy, -ux))
        ctx.navigator.go_to(goal, tolerance=self.goal_tolerance + 0.1 * self.attempts)

    def step(self, ctx: TaskContext) -> str:
        target = self._target(ctx)
        if target is None:
            return ExploreState.name
        pose = ctx.robot.get_pose()
        nav = ctx.navigator
        if math.hypot(target.x - pose.x, target.y - pose.y) < self.standoff(ctx) + 0.1:
            nav.cancel()
            return GrabState.name
        if self.goal_xy is None or math.dist(self.goal_xy, (target.x, target.y)) > 0.25:
            self.goal_xy = (target.x, target.y)
            self._go(ctx, target)
        elif nav.is_done():
            if nav.succeeded():
                return GrabState.name
            self.attempts += 1
            ctx.log(f"approach: navigation failed (attempt {self.attempts})")
            if self.attempts >= self.max_attempts:
                # Keep mapping elsewhere for a while (the way may be blocked by a person
                # or not mapped yet), then try again; give up after a few rounds.
                self.attempts = 0
                ctx.approach_rounds += 1
                if ctx.approach_rounds > self.max_rounds:
                    return ctx.fail("could not reach the target object")
                ctx.retry_target_after = ctx.time + 20.0
                ctx.log("approach: exploring for 20 s before retrying")
                return ExploreState.name
            self._go(ctx, target)
        return self.name


class GrabState(State):
    """Face the object, creep up until the contact sensor fires, then pick it up.

    Heading control has hysteresis: turn in place above 0.25 rad until within
    0.06 rad, and while creeping ignore errors under the 0.03 rad deadband.
    """

    name = "grab"
    timeout = 12.0
    pick_timeout = 25.0  # s for the whole pick motion
    max_attempts = 3

    def __init__(self) -> None:
        self.elapsed = 0.0
        self.attempts = 0
        self.turning = False
        self.picking: str | None = None
        self.pick_time = 0.0

    def on_enter(self, ctx: TaskContext) -> None:
        self.elapsed = 0.0
        self.turning = False
        self.picking = None
        ctx.log("grab: aligning with object")

    def _retry(self, ctx: TaskContext, why: str) -> str:
        self.attempts += 1
        ctx.robot.stop()
        if self.attempts >= self.max_attempts:
            return ctx.fail(why)
        ctx.log(f"grab: {why}, re-approaching")
        return ApproachState.name

    def step(self, ctx: TaskContext) -> str:
        if self.picking is not None:
            ctx.robot.stop()
            self.pick_time += ctx.dt
            status = ctx.robot.update_manipulation(ctx.dt)
            if status == "done":
                ctx.grabbed = self.picking
                if ctx.target_id is not None:
                    ctx.semantic_map.remove(ctx.target_id)
                ctx.log(f"grab: picked up {self.picking}")
                return DeliverState.name
            if status == "failed" or self.pick_time > self.pick_timeout:
                self.picking = None
                return self._retry(ctx, "pick failed")
            return self.name
        self.elapsed += ctx.dt
        if ctx.target_id is None or ctx.semantic_map.sightings(ctx.target_id) == 0:
            return ExploreState.name
        target = ctx.semantic_map.get(ctx.target_id).map_pose
        hit = None
        if ctx.grasp_sensor is not None:
            hit = ctx.grasp_sensor(near=(target.x, target.y))
        if hit is not None:
            ctx.robot.stop()
            ctx.robot.attach(hit)
            self.picking, self.pick_time = hit, 0.0
            ctx.log(f"grab: picking up {hit}")
            return self.name
        if self.elapsed > self.timeout:
            return self._retry(ctx, "grasp timed out")
        pose = ctx.robot.get_pose()
        err = wrap_angle(math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw)
        dist = math.hypot(target.x - pose.x, target.y - pose.y)
        if dist > ApproachState.standoff(ctx) + 0.3:  # it moved (pushed, re-estimated)
            ctx.robot.stop()
            ctx.log("grab: object out of reach, re-approaching")
            return ApproachState.name
        if self.turning and abs(err) < 0.06:
            self.turning = False
        elif not self.turning and abs(err) > 0.25:
            self.turning = True
        if self.turning:
            w = math.copysign(min(0.6, max(0.25, 1.2 * abs(err))), err)
            ctx.robot.send_velocity(0.0, w)
        else:
            v = 0.25 if dist > ctx.robot.front_extent + 0.35 else 0.1
            ctx.robot.send_velocity(v, 0.0 if abs(err) < 0.03 else 0.8 * err)
        return self.name


class DeliverState(State):
    """Navigate to the destination while carrying the object."""

    name = "deliver"
    tolerance = 0.2
    max_attempts = 4
    drop_offset = 0.3  # m from the zone center for the 2nd, 3rd, ... object in a zone

    def __init__(self) -> None:
        self.attempts = 0
        self.goal: Pose2D | None = None
        self.goal_tol = self.tolerance

    def on_enter(self, ctx: TaskContext) -> None:
        self.goal, self.goal_tol = self._drop_goal(ctx)
        dest = self.goal
        ctx.log(f"deliver: heading to '{ctx.command.destination}' ({dest.x:.2f}, {dest.y:.2f})")
        ctx.navigator.go_to(dest, tolerance=self.goal_tol)

    def _drop_goal(self, ctx: TaskContext) -> tuple[Pose2D, float]:
        """Zone center for the first object; otherwise the point on a ring around the
        center that is farthest from objects already placed there (the robot stops
        about ``place_ahead`` short of it, so the object lands on it)."""
        center = ctx.destinations[ctx.command.destination]
        placed = ctx.placed.get(ctx.command.destination, [])
        if not placed:
            return center, self.tolerance
        best, best_d = center, -1.0
        for k in range(8):
            a = k * math.pi / 4
            x = center.x + self.drop_offset * math.cos(a)
            y = center.y + self.drop_offset * math.sin(a)
            d = min(math.hypot(x - px, y - py) for px, py in placed)
            if d > best_d:
                best, best_d = Pose2D(x, y, 0.0), d
        return best, 0.3

    def step(self, ctx: TaskContext) -> str:
        nav = ctx.navigator
        if not nav.is_done():
            return self.name
        if nav.succeeded():
            return ReleaseState.name
        self.attempts += 1
        ctx.log(f"deliver: navigation failed (attempt {self.attempts})")
        if self.attempts >= self.max_attempts:
            return ctx.fail("could not reach the destination")
        nav.go_to(self.goal, tolerance=self.goal_tol)
        return self.name


class ReleaseState(State):
    """Set the carried object down at the destination."""

    name = "release"
    timeout = 25.0

    def __init__(self) -> None:
        self.elapsed = 0.0

    def on_enter(self, ctx: TaskContext) -> None:
        self.elapsed = 0.0
        ctx.robot.stop()
        ctx.robot.detach()

    def step(self, ctx: TaskContext) -> str:
        self.elapsed += ctx.dt
        ctx.robot.stop()
        status = ctx.robot.update_manipulation(ctx.dt)
        if status == "done":
            pose, ahead = ctx.robot.get_pose(), ctx.robot.place_ahead
            ctx.placed.setdefault(ctx.command.destination, []).append(
                (pose.x + ahead * math.cos(pose.yaw), pose.y + ahead * math.sin(pose.yaw))
            )
            ctx.log(f"release: placed object at '{ctx.command.destination}'")
            return DoneState.name
        if status == "failed" or self.elapsed > self.timeout:
            return ctx.fail("place failed")
        return self.name


class DoneState(State):
    """Terminal state."""

    name = "done"

    def on_enter(self, ctx: TaskContext) -> None:
        ctx.robot.stop()

    def step(self, ctx: TaskContext) -> str:
        return self.name
