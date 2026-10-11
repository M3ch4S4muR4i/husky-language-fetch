"""Robot platforms. Add real hardware by subclassing :class:`Robot`."""

from fetchbot.robots.base import Robot
from fetchbot.robots.pybullet_robot import HuskyRobot

__all__ = ["HuskyRobot", "Robot"]
