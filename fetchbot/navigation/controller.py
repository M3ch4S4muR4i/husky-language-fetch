"""Path-tracking controller for a differential-drive base."""

from __future__ import annotations

import math

from fetchbot.types import Pose2D


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class PurePursuitController:
    """Pure pursuit with rotate-in-place for large heading errors.

    Returns (v, w). Speed is reduced near the goal and when the heading error is large.
    """

    def __init__(
        self,
        lookahead: float = 0.45,
        max_linear: float = 0.5,
        max_angular: float = 1.5,
        rotate_threshold: float = 0.9,
    ) -> None:
        self.lookahead = lookahead
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.rotate_threshold = rotate_threshold

    def target_point(
        self, pose: Pose2D, path: list[tuple[float, float]], start_idx: int
    ) -> tuple[tuple[float, float], int]:
        """Closest path index at/after ``start_idx`` and the lookahead point beyond it."""
        best_i, best_d = start_idx, math.inf
        for i in range(start_idx, min(len(path), start_idx + 30)):
            d = math.hypot(path[i][0] - pose.x, path[i][1] - pose.y)
            if d < best_d:
                best_i, best_d = i, d
        for j in range(best_i, len(path)):
            if math.hypot(path[j][0] - pose.x, path[j][1] - pose.y) >= self.lookahead:
                return path[j], best_i
        return path[-1], best_i

    def compute(
        self,
        pose: Pose2D,
        target: tuple[float, float],
        dist_to_goal: float,
        speed_scale: float = 1.0,
    ) -> tuple[float, float]:
        dx, dy = target[0] - pose.x, target[1] - pose.y
        alpha = wrap_angle(math.atan2(dy, dx) - pose.yaw)
        if abs(alpha) > self.rotate_threshold:
            return 0.0, math.copysign(min(self.max_angular, 2.0 * abs(alpha)), alpha)
        v = self.max_linear * speed_scale * max(0.2, 1.0 - abs(alpha) / self.rotate_threshold)
        v = min(v, max(0.12, 0.8 * dist_to_goal))
        ld = max(math.hypot(dx, dy), 1e-3)
        w = 2.0 * v * math.sin(alpha) / ld + 0.8 * alpha
        return v, max(-self.max_angular, min(self.max_angular, w))
