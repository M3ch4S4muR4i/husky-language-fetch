"""Abstract navigator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fetchbot.types import Pose2D


class Navigator(ABC):
    """Sends goals to a planner/controller and reports progress."""

    @abstractmethod
    def go_to(self, goal: Pose2D, tolerance: float | None = None) -> None:
        """Start navigating to ``goal`` (non-blocking); ``tolerance`` overrides the default."""

    @abstractmethod
    def is_done(self) -> bool:
        """True once the current goal has succeeded, failed or been canceled."""

    @abstractmethod
    def succeeded(self) -> bool:
        """True if the last goal finished successfully."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the active goal."""

    def update(self, dt: float) -> None:
        """Advance planning/control by one tick.

        In-process navigators (e.g. :class:`GridNavigator`) do their work here;
        external stacks such as Nav2 run in their own process and ignore it.
        """
