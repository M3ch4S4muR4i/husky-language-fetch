"""PyBullet robot: the Clearpath Husky A200 loaded from ``pybullet_data/husky/husky.urdf``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fetchbot.robots.base import Robot
from fetchbot.types import Pose2D

if TYPE_CHECKING:
    from fetchbot.world.backend import PyBulletWorld


class HuskyRobot(Robot):
    """Clearpath Husky A200 skid-steer base driven through its four wheel joints.

    Each physics substep the commanded unicycle twist (v, w) is ramped with the
    acceleration limits and converted to wheel speeds with the URDF wheel radius
    and track, then applied to the four wheel joints with velocity control. The
    wheels really drive the chassis over the floor (friction, skidding, contacts).

    Speed and acceleration limits are Clearpath's published values:

    * max speed 1.0 m/s, footprint 990 x 670 x 390 mm, 50 kg: Husky A200 data
      sheet / spec comparison, https://clearpathrobotics.com/husky-spec-comparison/
    * angular 2.0 rad/s, linear accel 3.0 m/s^2, angular accel 6.0 rad/s^2 and the
      skid-steer ``wheel_separation_multiplier: 1.875``: husky_control/config/control.yaml,
      https://github.com/husky/husky/blob/noetic-devel/husky_control/config/control.yaml

    Grasping is *attach-on-contact* (no gripper physics): once the object touches the
    front bumper it is lifted kinematically onto the deck in 0.9 s (up, across, down)
    and held there with a fixed constraint; releasing is the reverse.
    """

    urdf = "husky/husky.urdf"
    # Geometry from the URDF (collision box 1.0074 x 0.5709 m plus the wheels).
    wheel_radius: float = 0.17775  # wheel collision cylinder radius
    track: float = 0.5708  # left/right wheel joint separation (2 x 0.2854)
    half_length: float = 0.505  # chassis collision box, x
    front_extent: float = 0.505  # base center to the front bumper
    half_width: float = 0.346  # outer face of the wheels, y
    radius: float = math.hypot(0.505, 0.346)  # footprint (circumscribed) radius, 0.61 m
    deck_height: float = 0.404  # deck surface (chassis collision box top) above the floor
    wheel_separation_multiplier: float = 1.875  # Clearpath's skid-steer correction
    wheel_force: float = 60.0  # max wheel torque for the velocity motors (N m)
    max_linear: float = 1.0
    max_angular: float = 2.0
    max_lin_acc: float = 3.0
    max_ang_acc: float = 6.0
    # Sensors (robot frame, m): camera on a short mast at the front of the top plate,
    # the emulated 2D lidar ring low at 0.10 m so it also sees the 0.14-0.16 m objects
    # (higher up, the 45 kg Husky would plow through objects it cannot see).
    camera_mount: tuple[float, float, float] = (0.30, 0.0, 0.70)
    camera_pitch_deg: float = -25.0
    lidar_height: float = 0.10
    #: Contact sensor reach: object center at most this far ahead of the base center
    #: (front bumper + object half-size ~0.08 m + 0.12 m) and within the width.
    grab_reach: float = 0.505 + 0.2
    place_ahead: float = 0.505 + 0.12  # where a released object is set down (m ahead)
    deck_spot: tuple[float, float] = (-0.10, 0.0)  # carried object on the deck (plate frame)
    lift_time: float = 0.9  # s for the visible lift onto / off the deck
    lift_clearance: float = 0.06  # m the object clears the deck while carried across

    WHEELS = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")

    def __init__(self, world: PyBulletWorld, start: Pose2D, name: str = "husky") -> None:
        super().__init__(name=name, base_frame="base_footprint")
        self.world = world
        self.start = start
        self.body_id: int | None = None
        self._constraint: int | None = None
        self._cmd = (0.0, 0.0)
        self._vel = (0.0, 0.0)
        self._wheels: list[int] = []
        self._left: list[bool] = []
        self.top_plate: int = -1
        self.chassis: int = -1
        self._move: dict | None = None  # running lift / set-down motion

    def spawn(self) -> None:
        p, cid = self.world.p, self.world.client
        self.body_id = self.world.load_urdf(
            self.urdf,
            (self.start.x, self.start.y, 0.0),
            p.getQuaternionFromEuler((0, 0, self.start.yaw)),
        )
        joints = {
            p.getJointInfo(self.body_id, j, physicsClientId=cid)[1].decode(): j
            for j in range(p.getNumJoints(self.body_id, physicsClientId=cid))
        }
        links = {
            p.getJointInfo(self.body_id, j, physicsClientId=cid)[12].decode(): j
            for j in joints.values()
        }
        self._wheels = [joints[n] for n in self.WHEELS]
        self._left = ["left" in n for n in self.WHEELS]
        self.top_plate = links["top_plate_link"]
        self.chassis = links["base_link"]  # its collision box top is the deck surface
        for j in self._wheels:
            p.changeDynamics(self.body_id, j, lateralFriction=0.8, physicsClientId=cid)
        self.world.pre_substep.append(self._apply_velocity)

    def get_pose(self) -> Pose2D:
        p, cid = self.world.p, self.world.client
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=cid)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return Pose2D(float(pos[0]), float(pos[1]), float(yaw))

    def send_velocity(self, linear: float, angular: float) -> None:
        self._cmd = (
            max(-self.max_linear, min(self.max_linear, linear)),
            max(-self.max_angular, min(self.max_angular, angular)),
        )

    def stop(self) -> None:
        self._cmd = (0.0, 0.0)

    @property
    def velocity(self) -> tuple[float, float]:
        """Commanded (v, w) after acceleration limiting."""
        return self._vel

    def _apply_velocity(self, dt: float) -> None:
        """Pre-substep hook: ramp toward the command and set the wheel speeds."""
        p, cid = self.world.p, self.world.client
        v, w = self._vel
        cv, cw = self._cmd
        dv, dw = self.max_lin_acc * dt, self.max_ang_acc * dt
        v += max(-dv, min(dv, cv - v))
        w += max(-dw, min(dw, cw - w))
        self._vel = (v, w)
        half = 0.5 * self.track * self.wheel_separation_multiplier
        left, right = (v - w * half) / self.wheel_radius, (v + w * half) / self.wheel_radius
        p.setJointMotorControlArray(
            self.body_id,
            self._wheels,
            p.VELOCITY_CONTROL,
            targetVelocities=[left if lw else right for lw in self._left],
            forces=[self.wheel_force] * len(self._wheels),
            physicsClientId=cid,
        )

    # -- grasping: attach-on-contact with a visible lift ----------------------
    def _collide(self, obj: int, enable: bool) -> None:
        p, cid = self.world.p, self.world.client
        for link in range(-1, p.getNumJoints(self.body_id, physicsClientId=cid)):
            p.setCollisionFilterPair(self.body_id, obj, link, -1, int(enable), physicsClientId=cid)

    def _plate(self):
        p, cid = self.world.p, self.world.client
        st = p.getLinkState(self.body_id, self.top_plate, physicsClientId=cid)
        return st[4], st[5]

    def _start_move(self, object_id: str, end: tuple[float, float, float], kind: str) -> None:
        """Move ``object_id`` kinematically to ``end`` (top-plate frame): up, across, down."""
        p, cid = self.world.p, self.world.client
        obj = self.world.objects[object_id]
        if self._constraint is not None:
            p.removeConstraint(self._constraint, physicsClientId=cid)
            self._constraint = None
        self._collide(obj, False)
        plate_pos, plate_orn = self._plate()
        inv = p.invertTransform(plate_pos, plate_orn)
        pos, _ = p.getBasePositionAndOrientation(obj, physicsClientId=cid)
        start = p.multiplyTransforms(inv[0], inv[1], pos, (0, 0, 0, 1))[0]
        mass = p.getDynamicsInfo(obj, -1, physicsClientId=cid)[0]
        p.changeDynamics(obj, -1, mass=0.0, physicsClientId=cid)  # kinematic while moved
        self._move = {
            "id": object_id,
            "obj": obj,
            "start": start,
            "end": end,
            "t": 0.0,
            "kind": kind,
            "mass": mass,
        }

    def _local_heights(self, obj: int) -> tuple[float, float, float]:
        """(deck top, floor, object half-height) in the top-plate frame (z)."""
        p, cid = self.world.p, self.world.client
        plate_z = self._plate()[0][2]
        deck_top = p.getAABB(self.body_id, self.chassis, physicsClientId=cid)[1][2]
        lo, hi = p.getAABB(obj, physicsClientId=cid)
        return deck_top - plate_z, -plate_z, (hi[2] - lo[2]) / 2

    def attach(self, object_id: str) -> None:
        """Start lifting ``object_id`` onto the deck; poll :meth:`update_manipulation`."""
        if self.body_id is None:
            raise RuntimeError("robot not spawned")
        deck, _, half_h = self._local_heights(self.world.objects[object_id])
        self._start_move(object_id, (*self.deck_spot, deck + half_h + 0.005), "pick")

    def detach(self) -> None:
        """Start setting the carried object down :attr:`place_ahead` in front."""
        if self.attached_object is None:
            return
        _, floor, half_h = self._local_heights(self.world.objects[self.attached_object])
        self._start_move(
            self.attached_object, (self.place_ahead, 0.0, floor + half_h + 0.005), "place"
        )

    def update_manipulation(self, dt: float) -> str:
        """Advance the lift / set-down motion: ``"busy"`` until it is ``"done"``."""
        m = self._move
        if m is None:
            return "done"
        p, cid = self.world.p, self.world.client
        m["t"] += dt
        s = min(1.0, m["t"] / self.lift_time)
        # Up, across above the deck / bumper, down: three equal-time eased segments.
        a, b = m["start"], m["end"]
        top = max(a[2], b[2]) + self.lift_clearance
        keys = [a, (a[0], a[1], top), (b[0], b[1], top), b]
        k = min(2, int(s * 3))
        u = s * 3 - k
        u = u * u * (3 - 2 * u)
        local = [p0 + (p1 - p0) * u for p0, p1 in zip(keys[k], keys[k + 1], strict=True)]
        plate_pos, plate_orn = self._plate()
        pos = p.multiplyTransforms(plate_pos, plate_orn, local, (0, 0, 0, 1))[0]
        yaw = self.get_pose().yaw
        orn = p.getQuaternionFromEuler((0, 0, yaw))
        p.resetBasePositionAndOrientation(m["obj"], pos, orn, physicsClientId=cid)
        if s < 1.0:
            return "busy"
        self._move = None
        p.changeDynamics(m["obj"], -1, mass=m["mass"], physicsClientId=cid)
        p.resetBaseVelocity(m["obj"], (0, 0, 0), (0, 0, 0), physicsClientId=cid)
        if m["kind"] == "pick":
            self._constraint = p.createConstraint(
                self.body_id,
                self.top_plate,
                m["obj"],
                -1,
                p.JOINT_FIXED,
                (0, 0, 0),
                parentFramePosition=m["end"],
                childFramePosition=(0, 0, 0),
                physicsClientId=cid,
            )
            p.changeConstraint(self._constraint, maxForce=500, physicsClientId=cid)
            self.attached_object = m["id"]
        else:
            self._collide(m["obj"], True)
            self.attached_object = None
        return "done"

    def drop(self) -> None:
        """Let go of the carried object at once (aborted job): set it down ahead."""
        if self.attached_object is None:
            return
        self.detach()
        self._move["t"] = self.lift_time
        self.update_manipulation(0.0)
