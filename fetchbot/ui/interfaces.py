"""Front ends that collect commands and display live status."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Interface(ABC):
    """Gets commands from the user and shows task progress."""

    @abstractmethod
    def read_command(self) -> str | None:
        """Return the next command text, or ``None`` to quit."""

    @abstractmethod
    def show_status(self, state: str, message: str = "") -> None:
        """Display the current task state and an optional message."""


class CliInterface(Interface):
    """Terminal interface: prompt for commands, print state transitions."""

    prompt = "fetchbot> "

    def read_command(self) -> str | None:
        try:
            text = input(self.prompt).strip()
        except EOFError:
            return None
        return text or None

    def show_status(self, state: str, message: str = "") -> None:
        print(f"[{state}] {message}".rstrip())


class GuiInterface(Interface):
    """Live GUI showing the semantic map, camera feed and task state."""

    def read_command(self) -> str | None:
        raise NotImplementedError

    def show_status(self, state: str, message: str = "") -> None:
        raise NotImplementedError
