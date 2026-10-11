"""Simulator backends. The task stack only talks to :class:`SimBackend`.

:class:`PyBulletWorld` is the working CPU backend. :class:`IsaacSimWorld` is kept
as the alternate (GPU) backend from the original plan and is still a stub.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np

from fetchbot.world.scene import OBJECT_COLORS, SceneSpec


class SimBackend(ABC):
    """Owns the physics world: builds a :class:`SceneSpec`, steps time, renders, raycasts."""

    def __init__(self, control_dt: float = 0.05) -> None:
        self.control_dt = control_dt
        self.sim_time = 0.0
        self.spec: SceneSpec | None = None

    @abstractmethod
    def build(self, spec: SceneSpec) -> None:
        """Create every static obstacle, object, zone and actor of ``spec``."""

    @abstractmethod
    def step(self) -> None:
        """Advance the world by one control period (``control_dt`` seconds)."""

    @abstractmethod
    def render(
        self,
        eye,
        target,
        width: int,
        height: int,
        fov_deg: float,
        near: float = 0.05,
        far: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render RGB (HxWx3 uint8) and metric depth (HxW float, meters along the view axis)."""

    @abstractmethod
    def raycast(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Return the hit fraction (0..1, 1 = no hit) for each ray."""

    @abstractmethod
    def close(self) -> None:
        """Release the simulator."""


class IsaacSimWorld(SimBackend):
    """Alternate backend: NVIDIA Isaac Sim + ROS 2 bridge (needs an RTX GPU). Stub."""

    def build(self, spec: SceneSpec) -> None:
        raise NotImplementedError("Isaac Sim backend: author the USD stage from SceneSpec")

    def step(self) -> None:
        raise NotImplementedError

    def render(self, eye, target, width, height, fov_deg, near=0.05, far=20.0):
        raise NotImplementedError

    def raycast(self, starts, ends):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PyBulletWorld(SimBackend):
    """Headless PyBullet world (``p.DIRECT`` + CPU ``ER_TINY_RENDERER``).

    Physics runs at ``physics_hz``; each :meth:`step` advances one control period
    and calls the registered pre-substep hooks (e.g. the robot's velocity
    controller) before every physics substep.
    """

    def __init__(self, control_dt: float = 0.05, physics_hz: int = 240) -> None:
        super().__init__(control_dt)
        import pybullet as p  # imported lazily so the package imports without PyBullet

        self.p = p
        self.physics_hz = physics_hz
        self.substeps = max(1, int(round(control_dt * physics_hz)))
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / physics_hz, physicsClientId=self.client)
        self.objects: dict[str, int] = {}  # graspable object name -> body id
        self.static_ids: list[int] = []
        self.actor_ids: dict[str, int] = {}
        self.pre_substep: list[Callable[[float], None]] = []
        self.actors = None  # set by DynamicActors.spawn

    # ----------------------------------------------------------------- build
    def build(self, spec: SceneSpec) -> None:
        p, cid = self.p, self.client
        self.spec = spec
        w, h = spec.size
        # Floor: a large thin box with a procedurally generated checker texture.
        floor = self._box((w / 2, h / 2, -0.05), (w + 4, h + 4, 0.1), (1, 1, 1, 1), mass=0)
        tex = self._checker_texture()
        if tex is not None:
            p.changeVisualShape(floor, -1, textureUniqueId=tex, physicsClientId=cid)
        p.changeDynamics(floor, -1, lateralFriction=0.8, physicsClientId=cid)
        self.static_ids.append(floor)

        for b in spec.boxes:
            self.static_ids.append(
                self._box((b.center[0], b.center[1], b.size[2] / 2), b.size, b.color, mass=0)
            )
        for c in spec.cylinders:
            self.static_ids.append(
                self._cylinder(
                    (c.center[0], c.center[1], c.height / 2), c.radius, c.height, c.color, mass=0
                )
            )
        # Drop zones: very thin discs, visual only (no collision).
        for z in spec.zones:
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=z.radius,
                length=0.004,
                rgbaColor=z.color,
                physicsClientId=cid,
            )
            p.createMultiBody(
                0, -1, vis, basePosition=(z.center[0], z.center[1], 0.002), physicsClientId=cid
            )
        for o in spec.objects:
            rgba = OBJECT_COLORS[o.color]
            if o.shape == "box":
                bid = self._box(
                    (o.position[0], o.position[1], 0.07), (0.14, 0.14, 0.14), rgba, mass=0.3
                )
            elif o.shape == "cylinder":
                bid = self._cylinder(
                    (o.position[0], o.position[1], 0.08), 0.06, 0.16, rgba, mass=0.3
                )
            else:
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.075, physicsClientId=cid)
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.075, rgbaColor=rgba, physicsClientId=cid
                )
                bid = p.createMultiBody(
                    0.3,
                    col,
                    vis,
                    basePosition=(o.position[0], o.position[1], 0.075),
                    physicsClientId=cid,
                )
                p.changeDynamics(
                    bid, -1, rollingFriction=0.01, spinningFriction=0.01, physicsClientId=cid
                )
            p.changeDynamics(bid, -1, lateralFriction=0.9, physicsClientId=cid)
            self.objects[o.name] = bid

    def load_urdf(self, path: str, pos, orn=(0, 0, 0, 1)) -> int:
        """Load a URDF from ``pybullet_data`` (e.g. ``husky/husky.urdf``)."""
        import pybullet_data

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        with _quiet_stdout():  # Bullet prints a warning per inertia-less URDF link
            return self.p.loadURDF(path, pos, orn, physicsClientId=self.client)

    def _box(self, pos, size, rgba, mass: float) -> int:
        p, cid = self.p, self.client
        half = [s / 2 for s in size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=cid)
        return p.createMultiBody(mass, col, vis, basePosition=pos, physicsClientId=cid)

    def _cylinder(self, pos, radius, height, rgba, mass: float) -> int:
        p, cid = self.p, self.client
        col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=cid
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba, physicsClientId=cid
        )
        return p.createMultiBody(mass, col, vis, basePosition=pos, physicsClientId=cid)

    def _checker_texture(self) -> int | None:
        """Write a light-gray checker PNG to a temp dir and load it as a texture."""
        import os
        import tempfile

        try:
            from PIL import Image
        except ImportError:
            return None
        n, cell = 512, 32
        ys, xs = np.mgrid[0:n, 0:n]
        chk = ((xs // cell + ys // cell) % 2).astype(np.uint8)
        img = np.where(chk[..., None] == 1, [214, 210, 202], [190, 186, 178]).astype(np.uint8)
        # Per-process file, written atomically: parallel batch workers must not
        # read each other's half-written PNG ("Unsupported texture image format").
        path = os.path.join(tempfile.gettempdir(), f"fetchbot_floor_{os.getpid()}.png")
        tmp = path + ".tmp.png"
        Image.fromarray(img).save(tmp)
        os.replace(tmp, path)
        return self.p.loadTexture(path, physicsClientId=self.client)

    # ------------------------------------------------------------------ time
    def step(self) -> None:
        dt = 1.0 / self.physics_hz
        if self.actors is not None:
            self.actors.update(self.control_dt)
        for _ in range(self.substeps):
            for hook in self.pre_substep:
                hook(dt)
            self.p.stepSimulation(physicsClientId=self.client)
        self.sim_time += self.control_dt

    # ------------------------------------------------------------- rendering
    def render(self, eye, target, width, height, fov_deg, near=0.05, far=20.0):
        p, cid = self.p, self.client
        view = p.computeViewMatrix(eye, target, (0, 0, 1), physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(fov_deg, width / height, near, far, physicsClientId=cid)
        _, _, rgba, depth_buf, _ = p.getCameraImage(
            width,
            height,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=(0.4, -0.6, 1.0),
            shadow=0,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=cid,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        zb = np.asarray(depth_buf, dtype=np.float32).reshape(height, width)
        depth = far * near / (far - (far - near) * zb)  # OpenGL z-buffer -> meters
        return rgb, depth

    def raycast(self, starts, ends):
        res = self.p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=self.client)
        return np.array([r[2] if r[0] >= 0 else 1.0 for r in res], dtype=np.float32)

    # ------------------------------------------------------------- utilities
    def body_xy(self, body_id: int) -> tuple[float, float]:
        pos, _ = self.p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
        return float(pos[0]), float(pos[1])

    def object_xy(self, name: str) -> tuple[float, float]:
        return self.body_xy(self.objects[name])

    def close(self) -> None:
        try:
            self.p.disconnect(physicsClientId=self.client)
        except Exception:  # noqa: BLE001 - already disconnected
            pass


@contextmanager
def _quiet_stdout():
    """Silence C-level stdout (Bullet's URDF importer warnings) for the ``with`` block."""
    import os
    import sys

    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a rotation of ``yaw`` about +z."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
