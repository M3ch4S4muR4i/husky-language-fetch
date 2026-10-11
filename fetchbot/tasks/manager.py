"""Runs the fetch state machine."""

from __future__ import annotations

from collections.abc import Iterable

from fetchbot.tasks.states import (
    ApproachState,
    DeliverState,
    DoneState,
    ExploreState,
    GrabState,
    ReleaseState,
    State,
    TaskContext,
)


def default_states() -> list[State]:
    """The standard fetch pipeline, in order."""
    return [
        ExploreState(),
        ApproachState(),
        GrabState(),
        DeliverState(),
        ReleaseState(),
        DoneState(),
    ]


class TaskManager:
    """Owns the states, tracks the active one, and applies transitions."""

    def __init__(self, states: Iterable[State] | None = None, initial: str = "explore") -> None:
        self.states: dict[str, State] = {s.name: s for s in (states or default_states())}
        if initial not in self.states:
            raise ValueError(f"unknown initial state: {initial!r}")
        self.current: str = initial
        self._entered = False

    @property
    def is_done(self) -> bool:
        """True once the machine has reached the terminal ``done`` state."""
        return self.current == DoneState.name

    def tick(self, ctx: TaskContext) -> str:
        """Run one step of the active state and transition if it asks to."""
        state = self.states[self.current]
        if not self._entered:
            state.on_enter(ctx)
            self._entered = True
        next_name = state.step(ctx)
        if next_name != self.current:
            if next_name not in self.states:
                raise KeyError(f"state {self.current!r} requested unknown state {next_name!r}")
            state.on_exit(ctx)
            self.current = next_name
            self._entered = False
        return self.current

    def run(self, ctx: TaskContext, max_ticks: int = 10_000) -> bool:
        """Tick until done or ``max_ticks``; return True on success."""
        for _ in range(max_ticks):
            if self.is_done:
                return True
            self.tick(ctx)
        return self.is_done
