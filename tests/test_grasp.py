"""Contact-sensor reach for the grab (pure geometry, no simulator)."""

import math

from fetchbot.evaluation.episode import object_in_reach
from fetchbot.robots import HuskyRobot
from fetchbot.types import Pose2D

FRONT, REACH, HALF_W = HuskyRobot.front_extent, HuskyRobot.grab_reach, HuskyRobot.half_width


ORIGIN = Pose2D(0.0, 0.0, 0.0)


def _reach(objects, pose=ORIGIN, near=None):
    return object_in_reach(pose, objects, FRONT, REACH, HALF_W, near=near)


def test_only_objects_at_the_bumper_are_grabbed():
    assert _reach({"box": (FRONT + 0.15, 0.0)}) == "box"
    assert _reach({"box": (FRONT + 0.4, 0.0)}) is None  # in front, but not touching
    assert _reach({"box": (3.0, 0.0)}) is None  # several meters away
    assert _reach({"box": (FRONT + 0.15, HALF_W + 0.1)}) is None  # beside the bumper
    assert _reach({"box": (-FRONT - 0.15, 0.0)}) is None  # behind


def test_reach_follows_the_heading():
    pose = Pose2D(2.0, 1.0, math.pi / 2)
    assert _reach({"box": (2.0, 1.0 + FRONT + 0.1)}, pose) == "box"
    assert _reach({"box": (2.0 + FRONT + 0.1, 1.0)}, pose) is None


def test_grab_takes_the_target_not_a_neighbor():
    objects = {"ball": (FRONT + 0.1, 0.25), "box": (FRONT + 0.15, -0.2)}
    assert _reach(objects, near=(FRONT + 0.15, -0.2)) == "box"
    assert _reach(objects, near=(3.0, 0.0)) is None  # target estimate elsewhere
