"""Runs many randomized episodes and saves metrics, videos and reports."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fetchbot.evaluation.metrics import EpisodeResult, Metrics

SHAPE_WORDS = {"box": "box", "cylinder": "mug", "ball": "ball"}


def command_for_seed(seed: int) -> str:
    """Deterministic pseudo-random command for a seed (object x destination)."""
    from fetchbot.world.randomizer import OBJECT_CATALOG, ZONE_STYLES

    rng = random.Random(seed)
    color, shape = rng.choice(OBJECT_CATALOG)
    dest = rng.choice(list(ZONE_STYLES))
    verb = rng.choice(["bring", "fetch", "take"])
    return f"{verb} the {color} {SHAPE_WORDS[shape]} to the {dest}"


def _run_one(args: tuple) -> EpisodeResult:
    from fetchbot.evaluation.episode import EpisodeConfig, FetchEpisode

    seed, command, overrides, program = args
    ep = FetchEpisode(EpisodeConfig(command=command, seed=seed, **overrides), program)
    try:
        return ep.run()
    finally:
        ep.close()


class BatchRunner:
    """Loops over episodes: randomize world -> run task -> record -> export.

    Episodes are independent PyBullet clients, so they run in parallel worker
    processes. Seeds are ``seed, seed+1, ..., seed+episodes-1``.
    """

    def __init__(
        self,
        episodes: int,
        output_dir: str | Path = "outputs",
        seed: int = 0,
        command: str | None = None,
        workers: int = 4,
        episode_overrides: dict | None = None,
    ) -> None:
        self.episodes = episodes
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.command = command
        self.workers = workers
        self.overrides = episode_overrides or {}
        self.metrics = Metrics()

    def _jobs(self) -> list[tuple]:
        """(seed, command, overrides, program). Commands are parsed once here, in the
        main process, so parallel workers do not each load the language model."""
        from fetchbot.language import make_parser

        parser = make_parser(self.overrides.get("parser", "llm"))
        cache: dict = {}
        jobs = []
        for s in range(self.seed, self.seed + self.episodes):
            text = self.command or command_for_seed(s)
            if text not in cache:
                cache[text] = parser.parse_program(text)
            jobs.append((s, text, self.overrides, cache[text]))
        return jobs

    def run_episode(self, episode: int) -> EpisodeResult:
        """Execute one episode end to end (in this process)."""
        seed = self.seed + episode
        text = self.command or command_for_seed(seed)
        return _run_one((seed, text, self.overrides, None))

    def run(self, progress=None) -> Metrics:
        """Run all episodes (in parallel), export CSV/JSON, and return collected metrics."""
        from fetchbot.export import ReportExporter

        jobs = self._jobs()
        if self.workers <= 1:
            results = []
            for job in jobs:
                results.append(_run_one(job))
                if progress:
                    progress(results[-1])
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                results = []
                for r in pool.map(_run_one, jobs):
                    results.append(r)
                    if progress:
                        progress(r)
        for r in sorted(results, key=lambda r: r.episode):
            self.metrics.add(r)
        rep = ReportExporter(self.output_dir)
        rep.export([r.row() for r in self.metrics.results], "batch_episodes.csv")
        rep.export([self.metrics.summary()], "batch_summary.csv")
        rep.export(
            {
                "summary": self.metrics.summary(),
                "episodes": [r.row() for r in self.metrics.results],
            },
            "batch_summary.json",
        )
        return self.metrics
