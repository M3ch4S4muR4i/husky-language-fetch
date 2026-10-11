"""PyBullet implementations of the sensor streams.

They mirror the ROS 2 topics of the Isaac Sim plan (``/camera/rgb``, ``/scan``,
``/odom``) but read straight from the simulator each time :meth:`poll` is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fetchbot.data.streams import CameraStream, LidarStream, OdomStream
from fetchbot.types import Pose2D

if TYPE_CHECKING:
    from fetchbot.robots.base import Robot
    from fetchbot.world.backend import SimBackend


@dataclass(frozen=True)
class CameraFrame:
    """One RGB-D image plus the intrinsics/extrinsics needed to back-project pixels."""

    rgb: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW float32, meters along the optical axis
    position: np.ndarray  # camera center in the world (3,)
    forward: np.ndarray  # unit optical axis (3,)
    right: np.ndarray  # unit image +u direction (3,)
    up: np.ndarray  # unit image -v direction (3,)
    fx: float
    fy: float
    stamp: float


@dataclass(frozen=True)
class LaserScan:
    """Planar 360-degree scan; ``ranges`` are from the lidar center (inf = no return)."""

    origin: Pose2D
    angles: np.ndarray  # robot-relative beam angles (rad)
    ranges: np.ndarray
    max_range: float
    stamp: float

    def endpoints(self) -> np.ndarray:
        """World (x, y) of each beam end (hit point, or max range for misses)."""
        r = np.where(np.isfinite(self.ranges), self.ranges, self.max_range)
        a = self.angles + self.origin.yaw
        return np.stack([self.origin.x + r * np.cos(a), self.origin.y + r * np.sin(a)], axis=1)


class PyBulletCameraStream(CameraStream):
    """Forward-facing RGB-D camera on the robot's mast (CPU tiny renderer)."""

    def __init__(
        self,
        world: SimBackend,
        robot: Robot,
        width: int = 160,
        height: int = 120,
        fov_deg: float = 60.0,
        mount=(0.30, 0.0, 0.70),
        pitch_deg: float = -25.0,
        max_range: float = 6.0,
        topic: str = "/camera/rgbd",
    ) -> None:
        super().__init__(topic)
        self.world, self.robot = world, robot
        self.width, self.height, self.fov_deg = width, height, fov_deg
        self.mount, self.pitch = mount, math.radians(pitch_deg)
        self.max_range = max_range
        self._latest: CameraFrame | None = None

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view (``fov_deg`` is vertical, as in PyBullet)."""
        v = math.radians(self.fov_deg)
        return math.degrees(2 * math.atan(math.tan(v / 2) * self.width / self.height))

    def start(self) -> None:
        self._latest = None

    def poll(self) -> CameraFrame:
        """Render a new frame from the robot's current pose."""
        pose = self.robot.get_pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        mx, my, mz = self.mount
        pos = np.array([pose.x + c * mx - s * my, pose.y + s * mx + c * my, mz])
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fwd = np.array([c * cp, s * cp, sp])
        right = np.array([s, -c, 0.0])
        up = np.cross(right, fwd)
        rgb, depth = self.world.render(
            pos, pos + fwd, self.width, self.height, self.fov_deg, near=0.05, far=self.max_range + 4
        )
        fy = (self.height / 2) / math.tan(math.radians(self.fov_deg) / 2)
        self._latest = CameraFrame(rgb, depth, pos, fwd, right, up, fy, fy, self.world.sim_time)
        return self._latest

    def latest(self) -> CameraFrame | None:
        return self._latest


class RaycastLidarStream(LidarStream):
    """2D lidar emulated with ``rayTestBatch`` at a fixed height above the floor.

    Rays start just outside the robot's rectangular ``footprint`` (half-length,
    half-width) so they never hit the robot's own body; ranges are still measured
    from the robot center.
    """

    def __init__(
        self,
        world: SimBackend,
        robot: Robot,
        num_beams: int = 360,
        max_range: float = 6.0,
        height: float = 0.22,
        footprint: tuple[float, float] = (0.505, 0.346),
        margin: float = 0.03,
        topic: str = "/scan",
    ) -> None:
        super().__init__(topic)
        self.world, self.robot = world, robot
        self.angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)
        self.max_range, self.height = max_range, height
        hl, hw = footprint
        with np.errstate(divide="ignore"):
            edge = np.minimum(hl / np.abs(np.cos(self.angles)), hw / np.abs(np.sin(self.angles)))
        self.start_radius = (edge + margin)[:, None]
        self._latest: LaserScan | None = None

    def start(self) -> None:
        self._latest = None

    def poll(self) -> LaserScan:
        pose = self.robot.get_pose()
        a = self.angles + pose.yaw
        d = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
        o = np.array([pose.x, pose.y, self.height])
        starts = o + d * self.start_radius
        ends = o + d * self.max_range
        frac = self.world.raycast(starts, ends)
        sr = self.start_radius[:, 0]
        dist = sr + frac * (self.max_range - sr)
        ranges = np.where(frac < 1.0, dist, np.inf)
        self._latest = LaserScan(
            pose, self.angles.copy(), ranges, self.max_range, self.world.sim_time
        )
        return self._latest

    def latest(self) -> LaserScan | None:
        return self._latest


class GroundTruthOdomStream(OdomStream):
    """Odometry read from the simulator (perfect, no drift) and integrated path length."""

    def __init__(self, robot: Robot, topic: str = "/odom") -> None:
        super().__init__(topic)
        self.robot = robot
        self.distance = 0.0
        self._latest: Pose2D | None = None

    def start(self) -> None:
        self._latest = self.robot.get_pose()
        self.distance = 0.0

    def poll(self) -> Pose2D:
        pose = self.robot.get_pose()
        if self._latest is not None:
            self.distance += math.hypot(pose.x - self._latest.x, pose.y - self._latest.y)
        self._latest = pose
        return pose

    def latest(self) -> Pose2D | None:
        return self._latest
