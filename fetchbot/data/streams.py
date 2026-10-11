"""Typed sensor streams. Each wraps one ROS 2 topic and exposes the latest sample."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SensorStream(ABC):
    """A subscription to one sensor topic with a ``latest()`` accessor."""

    def __init__(self, topic: str) -> None:
        self.topic = topic

    @abstractmethod
    def start(self) -> None:
        """Create the underlying subscription."""

    @abstractmethod
    def latest(self) -> Any:
        """Return the most recent sample, or ``None`` if nothing has arrived."""

    def stop(self) -> None:
        """Tear down the subscription. Override if cleanup is needed."""


class CameraStream(SensorStream):
    """RGB (and optionally depth) images from the robot's camera."""

    def start(self) -> None:
        """Subscribe to sensor_msgs/Image (and depth, if configured)."""
        raise NotImplementedError

    def latest(self) -> Any:
        """Return the newest image as a numpy array."""
        raise NotImplementedError


class LidarStream(SensorStream):
    """Laser scan or point cloud from the robot's lidar."""

    def start(self) -> None:
        """Subscribe to LaserScan / PointCloud2."""
        raise NotImplementedError

    def latest(self) -> Any:
        """Return the newest scan."""
        raise NotImplementedError


class OdomStream(SensorStream):
    """Wheel / visual odometry."""

    def start(self) -> None:
        """Subscribe to nav_msgs/Odometry."""
        raise NotImplementedError

    def latest(self) -> Any:
        """Return the newest odometry message."""
        raise NotImplementedError
