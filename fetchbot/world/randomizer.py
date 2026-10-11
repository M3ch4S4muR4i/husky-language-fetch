"""Domain randomization: generate a different (but always solvable) arena per seed."""

from __future__ import annotations

import math

import numpy as np

from fetchbot.world.difficulty import DISTRACTOR_COLORS, STANDARD, Difficulty
from fetchbot.world.scene import (
    OBJECT_COLORS,
    ActorSpec,
    BoxSpec,
    CylinderSpec,
    ObjectSpec,
    RegionSpec,
    SceneSpec,
    ZoneSpec,
)

#: The object catalog. Colors are unique so the color detector can tell them
#: apart; shapes are parsed from the command and checked against the catalog.
OBJECT_CATALOG: tuple[tuple[str, str], ...] = (
    ("red", "box"),
    ("blue", "cylinder"),
    ("green", "ball"),
    ("yellow", "box"),
)

#: Catalog with same-color, different-shape pairs (condition ``shapes``): the robot
#: must tell a red box from a red ball by shape, not just by color.
SHAPES_CATALOG: tuple[tuple[str, str], ...] = (
    ("red", "box"),
    ("red", "ball"),
    ("green", "ball"),
    ("green", "box"),
    ("blue", "cylinder"),
    ("yellow", "box"),
)
CATALOGS: dict[str, tuple[tuple[str, str], ...]] = {
    "standard": OBJECT_CATALOG,
    "shapes": SHAPES_CATALOG,
}

ZONE_STYLES: dict[str, tuple[float, float, float, float]] = {
    "dock": (0.60, 0.52, 0.68, 1.0),
    "table": (0.72, 0.66, 0.56, 1.0),
    "kitchen": (0.52, 0.64, 0.62, 1.0),
}

WALL_T = 0.12  # wall thickness (m)
WALL_H = 0.8  # wall height (m): low enough for the chase camera to see over
DOOR_W = 1.6  # default doorway width (m); see Difficulty.door_width
#: Arena scale for the Clearpath Husky (1.01 x 0.69 m) relative to the original
#: layout made for a 0.40 m robot: arena size, door spacing, clearances, zone radius
#: and the number of crates/bins/shelves are all scaled by this factor.
S = 1.5
ROBOT_HALF_WIDTH = 0.35  # Husky half-width (m), for the solvability check


class SceneRandomizer:
    """Randomizes room layout, door positions, shelves, clutter, objects, zones and start.

    Every generated scene is checked for solvability with a coarse ground-truth
    occupancy grid (start, objects and zones must be mutually reachable).
    """

    def __init__(
        self,
        seed: int = 0,
        size: tuple[float, float] = (15.0, 12.0),
        with_person: bool = True,
        difficulty: Difficulty = STANDARD,
    ) -> None:
        self.seed = seed
        self.size = size
        self.with_person = with_person
        self.difficulty = difficulty

    # ------------------------------------------------------------------ public
    def randomize(self, episode: int = 0) -> SceneSpec:
        """Return a solvable :class:`SceneSpec` for ``seed + episode``."""
        base = self.seed + episode
        for attempt in range(400):
            rng = np.random.default_rng(base * 1000 + attempt)
            spec = self._generate(rng, base)
            if spec is not None:
                return spec
        raise RuntimeError(f"could not generate a solvable scene for seed {base}")

    # ----------------------------------------------------------------- helpers
    def _generate(self, rng: np.random.Generator, seed: int) -> SceneSpec | None:
        w, h = self.size
        diff = self.difficulty
        spec = SceneSpec(seed=seed, size=self.size)
        wall = (0.70, 0.70, 0.68, 1.0)

        # Outer walls.
        spec.boxes += [
            BoxSpec("wall_s", (w / 2, 0.0), (w + WALL_T, WALL_T, WALL_H), wall),
            BoxSpec("wall_n", (w / 2, h), (w + WALL_T, WALL_T, WALL_H), wall),
            BoxSpec("wall_w", (0.0, h / 2), (WALL_T, h + WALL_T, WALL_H), wall),
            BoxSpec("wall_e", (w, h / 2), (WALL_T, h + WALL_T, WALL_H), wall),
        ]

        # A vertical dividing wall with one or two doorways -> west / east halves.
        wx = float(rng.uniform(0.4 * w, 0.6 * w))
        n_doors = int(rng.integers(1, 3))
        door_ys = sorted(rng.uniform(1.0 * S, h - 1.0 * S, size=n_doors).tolist())
        if n_doors == 2 and door_ys[1] - door_ys[0] < 2.2 * S:
            door_ys = [door_ys[0]]
        spec.boxes += self._wall_with_doors(
            "div_v", wx, 0.0, h, door_ys, vertical=True, color=wall, door_w=diff.door_width
        )

        # A horizontal wall splitting one half into two rooms, with one doorway.
        east = bool(rng.integers(0, 2))
        x0, x1 = (wx, w) if east else (0.0, wx)
        wy = float(rng.uniform(0.4 * h, 0.6 * h))
        door_x = float(rng.uniform(x0 + 0.9 * S, x1 - 0.9 * S))
        spec.boxes += self._wall_with_doors(
            "div_h", wy, x0, x1, [door_x], vertical=False, color=wall, door_w=diff.door_width
        )
        doors = [(wx, y) for y in door_ys] + [(door_x, wy)]
        # Named rooms (map frame: x east, y north). Consumes no randomness.
        spec.regions = [
            RegionSpec("west room", 0.0, 0.0, wx, h),
            RegionSpec("east room", wx, 0.0, w, h),
        ]
        side = "east" if east else "west"
        spec.regions += [
            RegionSpec(f"north{side} room", x0, wy, x1, h),
            RegionSpec(f"south{side} room", x0, 0.0, x1, wy),
        ]

        # Shelves against walls and free-standing clutter.
        shelf_c = (0.36, 0.38, 0.44, 1.0)
        k = diff.clutter
        for i in range(int(round(rng.integers(2, 4) * min(k, 1.6) * S))):
            horiz = bool(rng.integers(0, 2))
            size = (1.3, 0.4, 0.9) if horiz else (0.4, 1.3, 0.9)
            spec.boxes.append(BoxSpec(f"shelf_{i}", self._rand_xy(rng, 0.9 * S), size, shelf_c))
        for i in range(int(round(rng.integers(3, 6) * k * S))):
            s = float(rng.uniform(0.3, 0.5))
            tone = float(rng.uniform(0.45, 0.65))
            height = float(rng.uniform(0.35, 0.6))
            spec.boxes.append(
                BoxSpec(
                    f"crate_{i}",
                    self._rand_xy(rng, 0.7 * S),
                    (s, s, height),
                    (tone, tone, tone * 1.02, 1.0),
                )
            )
        for i in range(int(round(rng.integers(1, 3) * k * S))):
            spec.cylinders.append(
                CylinderSpec(
                    f"bin_{i}", self._rand_xy(rng, 0.7 * S), float(rng.uniform(0.15, 0.25)), 0.5
                )
            )

        # Drop any obstacle that blocks a doorway.
        def clear_of_doors(c: tuple[float, float], r: float) -> bool:
            return all(math.dist(c, d) > r + 1.0 * S for d in doors)

        spec.boxes = [
            b
            for b in spec.boxes
            if b.name.startswith(("wall", "div")) or clear_of_doors(b.center, max(b.size[:2]) / 2)
        ]
        spec.cylinders = [c for c in spec.cylinders if clear_of_doors(c.center, c.radius)]

        grid, res = self._gt_grid(spec, inflate=ROBOT_HALF_WIDTH + 0.1)

        def free(p: tuple[float, float], clearance: float = 0.0) -> bool:
            gx, gy = int(p[0] / res), int(p[1] / res)
            k = int(math.ceil(clearance / res))
            sub = grid[max(gy - k, 0) : gy + k + 1, max(gx - k, 0) : gx + k + 1]
            return sub.size > 0 and not sub.any()

        def sample_free(clearance: float, avoid: list[tuple[float, float]], min_d: float):
            for _ in range(300):
                p = self._rand_xy(rng, 0.6 * S)
                if free(p, clearance) and all(math.dist(p, a) > min_d for a in avoid):
                    return p
            return None

        used: list[tuple[float, float]] = []
        start = sample_free(0.65, used, 0.0)  # room to turn in place
        if start is None:
            return None
        used.append(start)
        spec.robot_start = (start[0], start[1], float(rng.uniform(-math.pi, math.pi)))

        for name, color in ZONE_STYLES.items():
            p = sample_free(0.45 * S, used, 2.0 * S)
            if p is None:
                return None
            used.append(p)
            spec.zones.append(ZoneSpec(name, p, 0.55 * S, color))

        for i, (color, shape) in enumerate(CATALOGS[diff.object_set]):
            assert color in OBJECT_COLORS
            p = None
            if diff.hide_objects:
                p = self._hidden_spot(rng, spec, start, used, free)
            if p is None:
                p = sample_free(0.25 * S, used, 1.5 * S)
            if p is None:
                return None
            used.append(p)
            spec.objects.append(ObjectSpec(f"obj_{i}_{color}_{shape}", color, shape, p))

        # Static near-miss colored blocks (tall enough for the lidar to see them).
        goals = list(used[1:])  # zones + objects: must stay reachable
        for name in list(DISTRACTOR_COLORS)[: diff.distractors]:
            p = sample_free(0.25 * S, used, 1.0 * S)
            if p is None:
                return None
            used.append(p)  # spacing only; a decoy is an obstacle, not a goal
            rgba = DISTRACTOR_COLORS[name]
            if rng.integers(0, 2):
                spec.boxes.append(BoxSpec(f"distractor_{name}", p, (0.18, 0.18, 0.26), rgba))
            else:
                spec.cylinders.append(CylinderSpec(f"distractor_{name}", p, 0.09, 0.26, rgba))
        if diff.distractors:
            grid, res = self._gt_grid(spec, inflate=ROBOT_HALF_WIDTH + 0.1)

        # Everything must be reachable from the start pose.
        if not self._all_reachable(grid, res, start, goals):
            return None

        if self.with_person:
            # The person walks back and forth through a random doorway (a busy spot),
            # so the robot regularly has to wait for / replan around them.
            order = rng.permutation(len(doors)).tolist()
            for i in order:
                dx_, dy_ = doors[i]
                vertical = i < len(door_ys)
                off = 1.6 * S
                a = (dx_ - off, dy_) if vertical else (dx_, dy_ - off)
                b = (dx_ + off, dy_) if vertical else (dx_, dy_ + off)
                # Keep the walking line clear of the objects (the person must not kick them).
                clear = all(self._dist_to_segment(o.position, a, b) > 0.7 * S for o in spec.objects)
                if (
                    clear
                    and free(a, 0.1)
                    and free(b, 0.1)
                    and math.dist(a, start) > 1.5 * S
                    and math.dist(b, start) > 1.5 * S
                    and self._all_reachable(grid, res, a, [b])
                ):
                    spec.actors.append(ActorSpec("person_0", (a, b), speed=0.3))
                    break
            if not spec.actors and diff.require_person:
                # No doorway fits: walk between two random free points instead.
                for _ in range(40):
                    a, b = sample_free(0.3, [start], 1.5 * S), sample_free(0.3, [start], 1.5 * S)
                    if a is None or b is None or math.dist(a, b) < 2.5 * S:
                        continue
                    if all(
                        self._dist_to_segment(o.position, a, b) > 0.7 * S for o in spec.objects
                    ) and self._all_reachable(grid, res, a, [b]):
                        spec.actors.append(ActorSpec("person_0", (a, b), speed=0.3))
                        break
            if not spec.actors and diff.require_person:
                return None
        return spec

    @staticmethod
    def _hidden_spot(rng, spec: SceneSpec, start, used, free) -> tuple[float, float] | None:
        """A spot tucked against a shelf/crate on the side facing away from the start,
        so the object is occluded from the robot's initial view."""
        props = [b for b in spec.boxes if b.name.startswith(("shelf", "crate"))]
        for idx in rng.permutation(len(props)).tolist():
            b = props[idx]
            hx, hy = b.size[0] / 2, b.size[1] / 2
            vx, vy = b.center[0] - start[0], b.center[1] - start[1]
            # Pick the face whose outward normal points most away from the start.
            if abs(vx) / max(hx, 1e-3) > abs(vy) / max(hy, 1e-3):
                n = (math.copysign(1.0, vx), 0.0)
            else:
                n = (0.0, math.copysign(1.0, vy))
            gap = 0.55  # object center to box face (> the solvability inflation)
            along = float(rng.uniform(-0.6, 0.6)) * (hy if n[0] else hx)
            p = (
                b.center[0] + n[0] * (hx + gap) + (0.0 if n[0] else along),
                b.center[1] + n[1] * (hy + gap) + (along if n[0] else 0.0),
            )
            if free(p, 0.0) and all(math.dist(p, a) > 1.2 * S for a in used):
                return p
        return None

    @staticmethod
    def _dist_to_segment(p, a, b) -> float:
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / (vx * vx + vy * vy)))
        return math.dist(p, (ax + t * vx, ay + t * vy))

    def _rand_xy(self, rng: np.random.Generator, margin: float) -> tuple[float, float]:
        w, h = self.size
        return (float(rng.uniform(margin, w - margin)), float(rng.uniform(margin, h - margin)))

    @staticmethod
    def _wall_with_doors(
        name: str,
        fixed: float,
        lo: float,
        hi: float,
        doors: list[float],
        vertical: bool,
        color,
        door_w: float = DOOR_W,
    ) -> list[BoxSpec]:
        """Split a wall segment [lo, hi] at the given door centers."""
        out: list[BoxSpec] = []
        edges = [lo]
        for d in doors:
            edges += [d - door_w / 2, d + door_w / 2]
        edges.append(hi)
        for i in range(0, len(edges), 2):
            a, b = edges[i], edges[i + 1]
            if b - a < 0.05:
                continue
            mid = (a + b) / 2
            if vertical:
                out.append(BoxSpec(f"{name}_{i}", (fixed, mid), (WALL_T, b - a, WALL_H), color))
            else:
                out.append(BoxSpec(f"{name}_{i}", (mid, fixed), (b - a, WALL_T, WALL_H), color))
        return out

    def _gt_grid(self, spec: SceneSpec, inflate: float, res: float = 0.1):
        """Rasterize the static obstacles (inflated) into a boolean grid."""
        w, h = self.size
        nx, ny = int(round(w / res)) + 1, int(round(h / res)) + 1
        ys, xs = np.mgrid[0:ny, 0:nx]
        cx, cy = (xs + 0.5) * res, (ys + 0.5) * res
        grid = np.zeros((ny, nx), dtype=bool)
        for b in spec.boxes:
            hx, hy = b.size[0] / 2 + inflate, b.size[1] / 2 + inflate
            grid |= (np.abs(cx - b.center[0]) <= hx) & (np.abs(cy - b.center[1]) <= hy)
        for c in spec.cylinders:
            grid |= np.hypot(cx - c.center[0], cy - c.center[1]) <= c.radius + inflate
        return grid, res

    @staticmethod
    def _all_reachable(grid: np.ndarray, res: float, start, goals) -> bool:
        from collections import deque

        ny, nx = grid.shape
        s = (int(start[1] / res), int(start[0] / res))
        if grid[s]:
            return False
        seen = np.zeros_like(grid)
        seen[s] = True
        q = deque([s])
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < ny and 0 <= u < nx and not seen[v, u] and not grid[v, u]:
                    seen[v, u] = True
                    q.append((v, u))
        return all(seen[int(g[1] / res), int(g[0] / res)] for g in goals)
