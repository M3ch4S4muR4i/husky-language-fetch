"""Moving people (and optionally relocated objects) to force replanning."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fetchbot.world.scene import ActorSpec

if TYPE_CHECKING:
    from fetchbot.world.backend import PyBulletWorld


class DynamicActors:
    """Spawns and drives dynamic obstacles (people).

    Each person is a kinematic cylinder that walks back and forth between its
    waypoints. It politely stops while the robot is close and in front of it
    (so a kinematic body never pushes into the robot), and if the robot is still
    in the way after ``give_way_after`` seconds the person turns around, like a
    human stepping back out of a doorway.
    """

    def __init__(
        self,
        num_people: int = 1,
        relocate_objects: bool = False,
        yield_distance: float = 0.9,
        give_way_after: float = 3.0,
    ) -> None:
        self.num_people = num_people
        self.relocate_objects = relocate_objects
        self.yield_distance = yield_distance
        self.give_way_after = give_way_after
        self.world: PyBulletWorld | None = None
        self.specs: list[ActorSpec] = []
        self._state: list[dict] = []
        self.robot_xy = None  # callable returning the robot (x, y), set by the episode

    def attach_world(self, world: PyBulletWorld, specs: list[ActorSpec]) -> None:
        self.world = world
        self.specs = list(specs)[: self.num_people]

    def spawn(self) -> None:
        """Add the actors to the world."""
        if self.world is None:
            raise RuntimeError("call attach_world() first")
        p, cid = self.world.p, self.world.client
        for s in self.specs:
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=s.radius,
                length=1.7,
                rgbaColor=(0.28, 0.27, 0.32, 1.0),
                physicsClientId=cid,
            )
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=s.radius, height=1.7, physicsClientId=cid
            )
            head = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.13, rgbaColor=(0.80, 0.68, 0.58, 1.0), physicsClientId=cid
            )
            x, y = s.waypoints[0]
            bid = p.createMultiBody(
                0,
                col,
                vis,
                basePosition=(x, y, 0.85),
                linkMasses=[0],
                linkCollisionShapeIndices=[-1],
                linkVisualShapeIndices=[head],
                linkPositions=[(0, 0, 0.98)],
                linkOrientations=[(0, 0, 0, 1)],
                linkInertialFramePositions=[(0, 0, 0)],
                linkInertialFrameOrientations=[(0, 0, 0, 1)],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[(0, 0, 1)],
                physicsClientId=cid,
            )
            self.world.actor_ids[s.name] = bid
            self._state.append({"id": bid, "pos": (x, y), "target": 1, "wait": 0.0})
        self.world.actors = self

    def update(self, dt: float) -> None:
        """Advance actor motion by ``dt`` seconds."""
        if self.world is None:
            return
        p, cid = self.world.p, self.world.client
        for s, st in zip(self.specs, self._state, strict=False):
            x, y = st["pos"]
            tx, ty = s.waypoints[st["target"]]
            if self.robot_xy is not None and self._robot_ahead((x, y), (tx, ty)):
                st["wait"] += dt
                if st["wait"] > self.give_way_after:  # give way: walk back the other way
                    st["target"] = (st["target"] - 1) % len(s.waypoints)
                    st["wait"] = 0.0
                continue
            st["wait"] = 0.0
            d = math.dist((x, y), (tx, ty))
            if d < 0.05:
                st["target"] = (st["target"] + 1) % len(s.waypoints)
                continue
            k = min(1.0, s.speed * dt / d)
            st["pos"] = (x + (tx - x) * k, y + (ty - y) * k)
            p.resetBasePositionAndOrientation(
                st["id"], (*st["pos"], 0.85), (0, 0, 0, 1), physicsClientId=cid
            )

    def _robot_ahead(self, pos: tuple[float, float], target: tuple[float, float]) -> bool:
        """True if the robot is within ``yield_distance`` and roughly in the walking direction."""
        rx, ry = self.robot_xy()
        d = math.dist(pos, (rx, ry))
        if d >= self.yield_distance:
            return False
        if d < 0.55:
            return True  # too close in any direction: freeze
        heading = math.atan2(target[1] - pos[1], target[0] - pos[0])
        bearing = math.atan2(ry - pos[1], rx - pos[0])
        return abs((bearing - heading + math.pi) % (2 * math.pi) - math.pi) < math.radians(75)

    def positions(self) -> list[tuple[float, float]]:
        return [st["pos"] for st in self._state]
