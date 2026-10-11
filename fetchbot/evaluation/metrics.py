"""Per-episode results and aggregate metrics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single fetch episode (times are simulated seconds)."""

    episode: int
    success: bool
    time_s: float
    distance_m: float
    replans: int = 0  # replans triggered because the current path became blocked
    ticks: int = 0
    wall_s: float = 0.0
    ticks_per_s: float = 0.0  # control-loop ticks per wall-clock second (excl. video)
    command: str = ""
    failure: str = ""
    grabbed: str = ""
    map_known: float = 0.0
    plans: int = 0
    monitor_stops: int = 0
    render_fps: float = 0.0
    condition: str = "standard"
    person: bool = False  # a walking person was present in this arena
    false_detections: int = 0  # semantic-map entries sitting on a distractor block
    program: str = ""  # parsed task program, e.g. "1. fetch(red box -> dock)"
    steps: int = 1
    steps_ok: int = 0  # program steps that succeeded (ground truth)
    step_log: list = field(default_factory=list, compare=False)  # per-step records
    events: list = field(default_factory=list, compare=False)

    def row(self) -> dict:
        """Flat dict for CSV export (without the event and step logs)."""
        return {k: v for k, v in self.__dict__.items() if k not in ("events", "step_log")}


@dataclass
class Metrics:
    """Collects episode results and summarizes them."""

    results: list[EpisodeResult] = field(default_factory=list)

    def add(self, result: EpisodeResult) -> None:
        """Record one episode."""
        self.results.append(result)

    @property
    def success_rate(self) -> float:
        """Fraction of successful episodes (0.0 if none recorded)."""
        if not self.results:
            return 0.0
        return sum(r.success for r in self.results) / len(self.results)

    def summary(self) -> dict[str, float]:
        """Aggregate statistics. Time/distance means are over successful episodes."""
        ok = [r for r in self.results if r.success]

        def mean(xs):
            return round(statistics.fmean(xs), 2) if xs else 0.0

        return {
            "episodes": len(self.results),
            "successes": len(ok),
            "success_rate": round(self.success_rate, 3),
            "mean_time_s": mean([r.time_s for r in ok]),
            "mean_distance_m": mean([r.distance_m for r in ok]),
            "mean_replans": mean([r.replans for r in self.results]),
            "mean_ticks_per_s": mean([r.ticks_per_s for r in self.results]),
        }
