"""One complete fetch episode in PyBullet: world + robot + sensors + mapping + tasks.

This is the "app wiring": it instantiates the concrete classes behind each
abstract interface and runs the fixed-rate control loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fetchbot.data import GroundTruthOdomStream, PyBulletCameraStream, RaycastLidarStream
from fetchbot.evaluation.metrics import EpisodeResult
from fetchbot.language import TaskProgram, make_parser, program_to_command
from fetchbot.language.schema import FetchStep
from fetchbot.mapping import CameraCoverage, FrontierExplorer, OccupancyGrid, SemanticMap
from fetchbot.navigation import GridNavigator, PurePursuitController
from fetchbot.perception import ColorDetector, localize
from fetchbot.robots import HuskyRobot
from fetchbot.tasks import TaskContext, TaskManager
from fetchbot.tasks.program import (
    Geometry,
    ObjInfo,
    StepRecord,
    needs_survey,
    select_candidates,
)
from fetchbot.types import FetchCommand, Pose2D
from fetchbot.world import DynamicActors, PyBulletWorld, SceneRandomizer, get_condition
from fetchbot.world.scene import SceneSpec


@dataclass
class EpisodeConfig:
    """Knobs for one episode."""

    command: str = "bring the red box to the dock"
    parser: str = "llm"  # "llm" (run-time parser) or "rule" (model-free, for tests)
    seed: int = 0
    max_sim_time: float = 450.0  # s of simulated time before giving up (1.5 x 300 s, arena S)
    condition: str = "standard"  # evaluation condition, see fetchbot.world.difficulty
    max_ticks: int | None = None  # step limit; None = use the condition's (if any)
    control_hz: float = 20.0  # fixed tick rate of the control loop (sim time)
    lidar_every: int = 2  # ticks between lidar scans / map updates (10 Hz)
    camera_every: int = 4  # ticks between camera frames / detections (5 Hz)
    with_person: bool = True
    explore_strategy: str = "union"  # or "frontier_first" (see FrontierExplorer)
    realtime: bool = False  # sleep so that sim time tracks wall-clock time
    video_path: str | None = None
    video_every: int = 2  # ticks between video frames (10 fps of sim time)
    video_fps: int = 20  # playback fps (20 fps / every 2 ticks = 2x speed)
    map_png: str | None = None
    verbose: bool = False
    status_callback: object = field(default=None, repr=False)  # callable(state, message)
    survey_cap: float = 180.0  # s of sim time a survey (all / superlative step) may take


def object_in_reach(
    pose: Pose2D,
    objects: dict[str, tuple[float, float]],
    front_extent: float,
    reach: float,
    half_width: float,
    near: tuple[float, float] | None = None,
    near_tol: float = 0.4,
) -> str | None:
    """The object (true position) right in front of the bumper, if any.

    In the robot frame its center must lie ``front_extent - 0.05 < x <= reach`` and
    ``|y| <= half_width``. With ``near`` (the target's map estimate) only an object
    within ``near_tol`` of it counts, so the grab never takes a different object.
    """
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    best, best_x = None, math.inf
    for name, (x, y) in objects.items():
        dx, dy = x - pose.x, y - pose.y
        xr, yr = c * dx + s * dy, -s * dx + c * dy
        if not (front_extent - 0.05 < xr <= reach and abs(yr) <= half_width):
            continue
        if near is not None and math.dist((x, y), near) > near_tol:
            continue
        if xr < best_x:
            best, best_x = name, xr
    return best


class FetchEpisode:
    """Builds every component for one seed/command and runs it to completion."""

    def __init__(
        self, config: EpisodeConfig, command: FetchCommand | TaskProgram | None = None
    ) -> None:
        self.config = config
        self.seed = config.seed
        self.difficulty = get_condition(config.condition)
        self.max_ticks = config.max_ticks or self.difficulty.max_ticks
        self.spec: SceneSpec = SceneRandomizer(
            seed=config.seed, with_person=config.with_person, difficulty=self.difficulty
        ).randomize()
        self.program = self._as_program(command, config.command)
        self.command = program_to_command(self.program)

        dt = 1.0 / config.control_hz
        self.world = PyBulletWorld(control_dt=dt)
        self.world.build(self.spec)
        self.robot = HuskyRobot(self.world, Pose2D(*self.spec.robot_start))
        self.robot.spawn()
        self.actors = DynamicActors(num_people=1 if config.with_person else 0)
        self.actors.attach_world(self.world, self.spec.actors)
        self.actors.spawn()
        self.actors.robot_xy = lambda: (self.robot.get_pose().x, self.robot.get_pose().y)

        rb = self.robot
        self.camera = PyBulletCameraStream(
            self.world, rb, mount=rb.camera_mount, pitch_deg=rb.camera_pitch_deg
        )
        self.lidar = RaycastLidarStream(
            self.world, rb, height=rb.lidar_height, footprint=(rb.half_length, rb.half_width)
        )
        self.odom = GroundTruthOdomStream(self.robot)
        for s in (self.camera, self.lidar, self.odom):
            s.start()

        w, h = self.spec.size
        self.grid = OccupancyGrid(w + 1.0, h + 1.0, resolution=0.1, origin=(-0.5, -0.5))
        self.coverage = CameraCoverage(self.grid, hfov_deg=self.camera.hfov_deg)
        self.explorer = FrontierExplorer(coverage=self.coverage, strategy=config.explore_strategy)
        self.explorer.update(self.grid)
        self.semantic_map = SemanticMap()
        # Lethal zone: half the Husky's width (+5 cm); the inflation cost band extends to
        # its footprint radius (+0.3 m) so it keeps room to turn its 1 m long body.
        self.navigator = GridNavigator(
            rb,
            self.grid,
            self.lidar,
            robot_radius=rb.half_width + 0.05,
            inflation_radius=rb.radius + 0.3,
            monitor_distance=rb.half_length + 0.25,
            controller=PurePursuitController(lookahead=0.7, max_linear=0.8, max_angular=1.5),
        )
        self.detector = ColorDetector()
        self.ctx = TaskContext(
            robot=self.robot,
            navigator=self.navigator,
            semantic_map=self.semantic_map,
            explorer=self.explorer,
            command=self.command,
            destinations={z.name: Pose2D(*z.center) for z in self.spec.zones},
            grasp_sensor=self._contact_sensor,
            dt=dt,
        )
        self.manager = TaskManager()
        self.last_camera_rgb: np.ndarray | None = None
        self.last_detections: list = []
        self.ticks = 0

        self._set_queries()
        self._reset_program()

    def _as_program(self, command, text: str) -> TaskProgram:
        if isinstance(command, TaskProgram):
            return command
        zones = [z.name for z in self.spec.zones]
        if isinstance(command, FetchCommand):
            color = command.attributes[0] if command.attributes else None
            shape = None if command.target_object == "object" else command.target_object
            step = FetchStep(color=color, shape=shape, destination=command.destination)
            return TaskProgram(steps=[step], raw_text=command.raw_text)
        return make_parser(self.config.parser, destinations=zones).parse_program(text)

    def _set_queries(self) -> None:
        # Detect every known color (not just the target) so the semantic map also
        # remembers other objects for later commands.
        self.queries = list(ColorDetector.HUE_RANGES)

    def set_command(self, command: FetchCommand | TaskProgram) -> None:
        """Start a new task in the *same* world, keeping the map built so far."""
        self.program = self._as_program(command, "")
        self.command = program_to_command(self.program)
        self.ctx.events.clear()
        self.navigator.blocked_replans = 0
        self._set_queries()
        self._reset_program()

    # ------------------------------------------------------------ program
    def _reset_program(self) -> None:
        self.step_index = -1
        self.records: list[StepRecord] = []
        self.finished = False
        self._failed_xy: list[tuple[float, float]] = []
        self._next_step()

    def geometry(self) -> Geometry:
        pose = self.robot.get_pose()
        return Geometry(
            zones={z.name: (z.center[0], z.center[1], z.radius) for z in self.spec.zones},
            shelves=[
                (
                    b.center[0] - b.size[0] / 2,
                    b.center[1] - b.size[1] / 2,
                    b.center[0] + b.size[0] / 2,
                    b.center[1] + b.size[1] / 2,
                )
                for b in self.spec.boxes
                if b.name.startswith("shelf")
            ],
            regions={r.name: (r.xmin, r.ymin, r.xmax, r.ymax) for r in self.spec.regions},
            robot_xy=(pose.x, pose.y),
        )

    def belief_objects(self, min_sightings: int = 2) -> list[ObjInfo]:
        """Objects in the semantic map (color = label, shape = majority vote)."""
        out = []
        for oid, det in self.semantic_map.items():
            if det.map_pose is None or self.semantic_map.sightings(oid) < min_sightings:
                continue
            out.append(
                ObjInfo(
                    oid, det.label, self.semantic_map.shape_of(oid), det.map_pose.x, det.map_pose.y
                )
            )
        return out

    def truth_objects(self) -> list[ObjInfo]:
        out = []
        for o in self.spec.objects:
            if o.name == self.robot.attached_object:
                continue
            x, y = self.world.object_xy(o.name)
            out.append(ObjInfo(o.name, o.color, o.shape, x, y))
        return out

    max_jobs_per_step = 8  # cap on fetches for one 'all' step

    def _select(self) -> str | None:
        """Next object to fetch for the current step (robot belief), or None."""
        step = self.program.steps[self.step_index]
        skip = list(self._failed_xy) + self.ctx.placed.get(step.destination, [])
        cands = select_candidates(
            step,
            self.belief_objects(),
            self._step_geo,
            exclude=lambda o: any(math.hypot(o.x - x, o.y - y) < 0.4 for x, y in skip),
        )
        return cands[0].oid if cands else None

    def _next_step(self) -> None:
        """Advance to the next program step (or finish)."""
        self.step_index += 1
        if self.step_index >= len(self.program.steps):
            self.finished = True
            return
        step = self.program.steps[self.step_index]
        self._step_geo = self.geometry()
        rec = StepRecord(self.step_index, step.describe(), t_start=self.world.sim_time)
        try:
            rec.expected = [
                o.oid for o in select_candidates(step, self.truth_objects(), self._step_geo)
            ]
        except LookupError as exc:
            rec.cause = f"unsatisfiable in this arena: {exc}"
        self.records.append(rec)
        self._jobs = 0
        self.ctx.log(
            f"program: step {self.step_index + 1}/{len(self.program.steps)} {step.describe()}"
        )
        self._start_job(first=True)

    def _start_job(self, first: bool) -> None:
        step = self.program.steps[self.step_index]
        c = self.ctx
        c.command = FetchCommand(
            target_object=step.shape or "object",
            destination=step.destination,
            attributes=(step.color,) if step.color else (),
            raw_text=self.program.raw_text,
        )
        self.command = c.command
        c.label = step.describe()[len("fetch(") :].split(" -> ")[0]
        c.select_target = self._select
        c.failure = None
        c.target_id = None
        c.grabbed = None
        c.retry_target_after = 0.0
        c.approach_rounds = 0
        c.surveying = first and needs_survey(step)
        c.survey_deadline = self.world.sim_time + self.config.survey_cap
        c.look_around = first and self.step_index == 0
        self.manager = TaskManager()
        self.navigator.cancel()

    def _job_done(self) -> None:
        """Called when the state machine reaches DONE: record, then next job/step.

        For an 'all' step a failed fetch (object unreachable, grasp timeout) is noted,
        its position skipped, and the robot moves on to the next candidate."""
        step = self.program.steps[self.step_index]
        rec = self.records[-1]
        c = self.ctx
        self._jobs += 1
        if c.failure is None and c.grabbed:
            rec.delivered.append(c.grabbed)
        nothing_left = c.failure is not None and (
            "found no" in c.failure or "not found" in c.failure
        )
        if step.quantifier == "all" and c.failure is not None and not nothing_left:
            rec.robot_failure = c.failure
            if c.target_id is not None and c.target_id in dict(self.semantic_map.items()):
                p = self.semantic_map.get(c.target_id).map_pose
                self._failed_xy.append((p.x, p.y))
            if c.grabbed and self.robot.is_holding:
                self.robot.drop()  # drop it here; it is not counted as delivered
        more = False
        if step.quantifier == "all" and not nothing_left and self._jobs < self.max_jobs_per_step:
            try:
                more = self._select() is not None
            except LookupError:
                more = False
        if more:
            self._start_job(first=False)
            return
        if step.quantifier == "all" and rec.delivered and (nothing_left or c.failure is None):
            c.failure = None  # nothing more to fetch: the 'all' step is complete
        rec.robot_failure = c.failure or rec.robot_failure
        rec.t_end = self.world.sim_time
        self._score_step(rec, step)
        if not rec.success:
            self.finished = True  # a failed step stops the program (partial result)
            return
        self._next_step()

    def _score_step(self, rec: StepRecord, step: FetchStep) -> None:
        zone = self.spec.zone(step.destination)

        def in_zone(name: str) -> bool:
            if name == self.robot.attached_object:
                return False
            x, y = self.world.object_xy(name)
            return math.hypot(x - zone.center[0], y - zone.center[1]) <= zone.radius

        names = {o.name: o.label for o in self.spec.objects}
        if not rec.expected:
            rec.success = False
            rec.cause = rec.cause or "unsatisfiable: no matching object in this arena"
            return
        if step.quantifier == "all":
            ok = all(in_zone(n) for n in rec.expected)
        else:
            ok = any(in_zone(n) and n in rec.delivered for n in rec.expected)
        rec.success = ok
        if ok:
            return
        wrong = [names.get(d, d) for d in rec.delivered if d not in rec.expected]
        if wrong:
            rec.cause = f"wrong object delivered: {', '.join(wrong)}"
        elif rec.robot_failure:
            rec.cause = rec.robot_failure
        elif step.quantifier == "all":
            missing = [names[n] for n in rec.expected if not in_zone(n)]
            rec.cause = f"missed {len(missing)}/{len(rec.expected)}: {', '.join(missing)}"
        else:
            rec.cause = "object not inside destination zone"

    # ------------------------------------------------------------- sensors
    def _contact_sensor(self, near: tuple[float, float] | None = None) -> str | None:
        """Stand-in for a bumper/gripper contact sensor: an object right at the front."""
        objects = {
            name: self.world.object_xy(name)
            for name in self.world.objects
            if name != self.robot.attached_object
        }
        rb = self.robot
        return object_in_reach(
            rb.get_pose(), objects, rb.front_extent, rb.grab_reach, rb.half_width, near=near
        )

    def _perceive(self) -> None:
        frame = self.camera.poll()
        self.last_camera_rgb = frame.rgb
        dets = self.detector.detect(frame.rgb, self.queries, frame)
        self.last_detections = dets
        for det, mask in zip(dets, self.detector.last_component_masks, strict=True):
            located = localize(det, mask, frame)
            if located is not None:
                self.semantic_map.add_detection(located)

    # --------------------------------------------------------------- loop
    def tick(self) -> None:
        cfg = self.config
        self.world.step()
        self.ctx.time = self.world.sim_time
        self.odom.poll()
        if self.ticks % cfg.lidar_every == 0:
            scan = self.lidar.poll()
            self.grid.update_from_scan(scan)
            self.coverage.update(scan)
        if self.ticks % cfg.camera_every == 0:
            self._perceive()
        before = self.manager.current
        self.manager.tick(self.ctx)
        if self.manager.is_done:
            self._job_done()
        self.navigator.update(self.ctx.dt)
        if self.manager.current != before:
            self.ctx.log(f"state {before} -> {self.manager.current}")
            if callable(cfg.status_callback):
                cfg.status_callback(self.manager.current, self.ctx.events[-1][1])
            if cfg.verbose:
                print(f"[t={self.world.sim_time:6.1f}s] {before} -> {self.manager.current}")
        self.ticks += 1

    def run(self) -> EpisodeResult:
        from fetchbot.export import MapExporter, VideoExporter
        from fetchbot.export.visualization import CompositeFrameRenderer

        cfg = self.config
        renderer = CompositeFrameRenderer() if (cfg.video_path or cfg.map_png) else None
        writer = None
        if cfg.video_path:
            writer = VideoExporter(Path(cfg.video_path).parent).open(
                Path(cfg.video_path).name, fps=cfg.video_fps
            )
        wall0 = time.perf_counter()
        loop_s = 0.0
        self._t0, self._d0, self._k0 = self.world.sim_time, self.odom.distance, self.ticks
        deadline = self.world.sim_time + cfg.max_sim_time
        tick_limit = self.ticks + self.max_ticks if self.max_ticks else None
        try:
            while not self.finished and self.world.sim_time < deadline:
                if tick_limit is not None and self.ticks >= tick_limit:
                    self.ctx.failure = (
                        f"step limit: {self.max_ticks} ticks "
                        f"({self.max_ticks * self.ctx.dt:.0f} s) in state "
                        f"{self.manager.current}"
                    )
                    self.ctx.log("FAILED: " + self.ctx.failure)
                    break
                t0 = time.perf_counter()
                self.tick()
                loop_s += time.perf_counter() - t0
                if writer is not None and self.ticks % cfg.video_every == 0:
                    writer.append_data(renderer.render(self))
                if cfg.realtime:
                    lag = self.ctx.dt - (time.perf_counter() - t0)
                    if lag > 0:
                        time.sleep(lag)
            if writer is not None:  # hold the final frame for 2 s
                last = renderer.render(self)
                for _ in range(2 * cfg.video_fps):
                    writer.append_data(last)
        finally:
            if writer is not None:
                writer.close()
        wall = time.perf_counter() - wall0
        if not self.finished and self.ctx.failure is None:
            self.ctx.failure = (
                f"timeout after {cfg.max_sim_time:.0f} s sim time in state {self.manager.current}"
            )
            self.ctx.log("FAILED: " + self.ctx.failure)
        if cfg.map_png:
            MapExporter(Path(cfg.map_png).parent).export(
                renderer.panel.render(self), Path(cfg.map_png).name
            )
        return self._result(wall, loop_s, renderer)

    # ------------------------------------------------------------- scoring
    def target_objects(self):
        """Ground-truth objects the program's steps refer to (for warnings/reports)."""
        out = []
        for step in self.program.steps:
            try:
                names = [
                    o.oid for o in select_candidates(step, self.truth_objects(), self.geometry())
                ]
            except LookupError:
                names = []
            out += [o for o in self.spec.objects if o.name in names and o not in out]
        return out

    def _result(self, wall: float, loop_s: float, renderer) -> EpisodeResult:
        if self.records and not self.finished:  # time / step limit hit mid-step
            rec = self.records[-1]
            rec.robot_failure = self.ctx.failure or "timeout"
            rec.t_end = self.world.sim_time
            if (
                self.ctx.grabbed
                and self.ctx.grabbed not in rec.delivered
                and self.robot.attached_object != self.ctx.grabbed
            ):
                rec.delivered.append(self.ctx.grabbed)
            self._score_step(rec, self.program.steps[self.step_index])
            if not rec.success and not rec.cause.startswith("wrong"):
                rec.cause = rec.robot_failure
        steps_ok = sum(r.success for r in self.records)
        success = steps_ok == len(self.program.steps)
        failed = next((r for r in self.records if not r.success), None)
        failure = (
            ""
            if success
            else (f"step {failed.index + 1}: {failed.cause}" if failed else self.ctx.failure or "")
        )
        render_fps = (
            renderer.frames_rendered / renderer.render_seconds
            if renderer is not None and renderer.render_seconds > 0
            else 0.0
        )
        return EpisodeResult(
            episode=self.seed,
            success=success,
            time_s=round(self.world.sim_time - self._t0, 2),
            distance_m=round(self.odom.distance - self._d0, 2),
            replans=self.navigator.blocked_replans,
            ticks=self.ticks - self._k0,
            wall_s=round(wall, 2),
            ticks_per_s=round((self.ticks - self._k0) / loop_s, 1) if loop_s > 0 else 0.0,
            command=self.program.raw_text,
            failure=failure,
            grabbed=",".join(d for r in self.records for d in r.delivered),
            map_known=round(self.grid.known_fraction(), 3),
            plans=self.navigator.plans,
            monitor_stops=self.navigator.monitor_stops,
            render_fps=round(render_fps, 2),
            condition=self.difficulty.name,
            person=bool(self.spec.actors),
            false_detections=self.false_detections(),
            program=self.program.describe(),
            steps=len(self.program.steps),
            steps_ok=steps_ok,
            step_log=[r.as_dict() for r in self.records],
            events=list(self.ctx.events),
        )

    def false_detections(self) -> int:
        """Semantic-map instances that lie on a distractor block (color false positives)."""
        decoys = [b.center for b in self.spec.boxes if b.name.startswith("distractor")]
        decoys += [c.center for c in self.spec.cylinders if c.name.startswith("distractor")]
        n = 0
        for _oid, det in self.semantic_map.items():
            p = det.map_pose
            if p is not None and any(math.hypot(p.x - x, p.y - y) < 0.35 for x, y in decoys):
                n += 1
        return n

    def close(self) -> None:
        self.world.close()


def run_episode(config: EpisodeConfig) -> EpisodeResult:
    """Convenience wrapper: build, run and tear down one episode."""
    ep = FetchEpisode(config)
    try:
        return ep.run()
    finally:
        ep.close()
