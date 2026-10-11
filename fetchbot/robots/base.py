"""Abstract robot interface used by every higher-level component."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fetchbot.types import Pose2D


class Robot(ABC):
    """Hardware-neutral mobile manipulator-ish robot.

    Concrete subclasses map these calls onto a simulator or real hardware.
    Task logic must only depend on this interface.
    """

    #: Where :meth:`detach` sets the carried object down (m ahead of the base center).
    place_ahead: float = 0.32
    #: Base center to the front of the robot (m); the approach standoff builds on it.
    front_extent: float = 0.2
    #: Contact sensor: object centers up to this far ahead of the base center are grabbed.
    grab_reach: float = 0.37
    half_width: float = 0.2

    def __init__(self, name: str, base_frame: str = "base_link") -> None:
        self.name = name
        self.base_frame = base_frame
        self.attached_object: str | None = None

    @abstractmethod
    def spawn(self) -> None:
        """Load the robot into the world (e.g. from URDF) or connect to hardware."""

    @abstractmethod
    def get_pose(self) -> Pose2D:
        """Return the current base pose in the map frame."""

    @abstractmethod
    def send_velocity(self, linear: float, angular: float) -> None:
        """Command base velocity (m/s, rad/s)."""

    @abstractmethod
    def stop(self) -> None:
        """Bring the base to a halt."""

    @abstractmethod
    def attach(self, object_id: str) -> None:
        """Start grasping an object (finish with :meth:`update_manipulation`)."""

    @abstractmethod
    def detach(self) -> None:
        """Start releasing the carried object (finish with :meth:`update_manipulation`)."""

    def update_manipulation(self, dt: float) -> str:
        """Advance a multi-tick :meth:`attach` / :meth:`detach` motion.

        Returns ``"busy"``, ``"done"`` or ``"failed"``; instant grippers are always done.
        """
        return "done"

    def drop(self) -> None:
        """Let go of the carried object at once (an aborted job); no placing motion."""
        self.detach()

    @property
    def is_holding(self) -> bool:
        """True if an object is currently attached."""
        return self.attached_object is not None
