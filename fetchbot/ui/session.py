"""Interactive session: one persistent arena, the user types commands one by one."""

from __future__ import annotations

from pathlib import Path

from fetchbot.ui.interfaces import Interface

HELP = """Commands (one or more fetch steps, parsed into a task program):
  bring|fetch|take the <color> <box|mug|ball> to the <dock|table|kitchen>
  ... then bring the <object> to the <place>          (multi-step)
  bring every red object to the dock                  (all)
  bring the box nearest the shelf to the table        (nearest/farthest, leftmost, ...)
  fetch the blue mug in the west room to the kitchen  (rooms)
  objects   - list what the robot has seen so far (semantic map)
  scene     - list the objects and zones in this arena (ground truth, for the user)
  help      - this text
  quit      - exit"""


class InteractiveSession:
    """Keeps one :class:`FetchEpisode` alive so the map persists between commands."""

    def __init__(
        self,
        ui: Interface,
        seed: int = 3,
        out_dir: str | Path = "outputs/interactive",
        video: bool = False,
        parser: str = "llm",
        condition: str = "standard",
    ) -> None:
        self.ui = ui
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.video = video
        self.parser_kind = parser
        self.condition = condition
        self._parser_obj = None
        self.episode = None
        self.count = 0

    def _ensure_episode(self, text: str):
        from fetchbot.evaluation.episode import EpisodeConfig, FetchEpisode

        program = self._parser().parse_program(text)
        if self.episode is None:
            cfg = EpisodeConfig(
                command=text,
                seed=self.seed,
                condition=self.condition,
                parser=self.parser_kind,
                status_callback=self.ui.show_status,
            )
            self.episode = FetchEpisode(cfg, program)
        else:
            self.episode.set_command(program)
        return self.episode

    def _parser(self):
        from fetchbot.language import make_parser
        from fetchbot.world.randomizer import ZONE_STYLES

        if self._parser_obj is None:
            self._parser_obj = make_parser(self.parser_kind, destinations=list(ZONE_STYLES))
        return self._parser_obj

    def loop(self) -> int:
        from fetchbot.world import SceneRandomizer

        self.ui.show_status("ready", f"arena seed {self.seed}. Type 'help' for commands.")
        while True:
            text = self.ui.read_command()
            if text is None or text.lower() in ("quit", "exit", "q"):
                break
            low = text.lower()
            if low == "help":
                self.ui.show_status("help", HELP)
                continue
            if low == "scene":
                from fetchbot.world import get_condition

                spec = SceneRandomizer(
                    seed=self.seed, difficulty=get_condition(self.condition)
                ).randomize()
                self.ui.show_status(
                    "scene",
                    f"objects: {[o.label for o in spec.objects]}; "
                    f"zones: {[z.name for z in spec.zones]}",
                )
                continue
            if low == "objects":
                sm = None if self.episode is None else self.episode.semantic_map
                seen = (
                    []
                    if sm is None
                    else [
                        f"{d.label} {sm.shape_of(oid) or '?'} @ ({d.map_pose.x:.1f}, "
                        f"{d.map_pose.y:.1f})"
                        for oid, d in sm.items()
                    ]
                )
                self.ui.show_status("objects", ", ".join(seen) or "nothing seen yet")
                continue
            try:
                ep = self._ensure_episode(text)
            except ValueError as exc:
                self.ui.show_status("error", f"not understood: {exc}")
                continue
            except RuntimeError as exc:
                self.ui.show_status("error", f"parser unavailable: {exc}")
                return 3
            self.count += 1
            ep.config.video_path = (
                str(self.out_dir / f"interactive_{self.count}.mp4") if self.video else None
            )
            ep.config.map_png = str(self.out_dir / f"interactive_{self.count}_map.png")
            self.ui.show_status("running", f"program: {ep.program.describe()}")
            result = ep.run()
            verdict = (
                f"SUCCESS ({result.steps_ok}/{result.steps} steps)"
                if result.success
                else f"FAILURE ({result.steps_ok}/{result.steps} steps; {result.failure})"
            )
            self.ui.show_status(
                "done",
                f"{verdict}: {result.time_s:.1f} s sim, "
                f"{result.distance_m:.1f} m, replans {result.replans}",
            )
        if self.episode is not None:
            self.episode.close()
        return 0
