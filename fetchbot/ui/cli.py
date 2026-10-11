"""``fetchbot`` console entry point.

Subcommands::

    fetchbot run --command "bring the red box to the dock" --seed 3 --video out/demo.mp4
    fetchbot run --command "bring every red object to the dock" --condition shapes --parser llm
    fetchbot batch --episodes 10 --out out/batch
    fetchbot interactive --seed 3          # type commands one after another
    fetchbot parse "bring the red box to the dock, then the green ball to the table"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fetchbot import __version__
from fetchbot.world.difficulty import CONDITIONS


def build_arg_parser() -> argparse.ArgumentParser:
    """Command-line options shared by the CLI and scripts."""
    parser = argparse.ArgumentParser(
        prog="fetchbot", description="Language-commanded fetch robot (PyBullet backend)."
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument("--gui", action="store_true", help="(reserved) use the live GUI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    parser_help = (
        "command parser: llm (local GGUF model via llama.cpp, the default) or "
        "rule (keyword baseline, no model needed)"
    )

    run = sub.add_parser("run", help="run one fetch episode")
    run.add_argument("--command", default="bring the red box to the dock")
    run.add_argument("--parser", default="llm", choices=("llm", "rule"), help=parser_help)
    run.add_argument("--seed", type=int, default=3)
    run.add_argument("--video", default=None, help="MP4 output path (slower: renders frames)")
    run.add_argument("--out", default="outputs", help="directory for report + map PNG")
    run.add_argument("--max-time", type=float, default=450.0, help="sim-time limit (s)")
    run.add_argument("--no-person", action="store_true", help="disable the walking person")
    run.add_argument("--realtime", action="store_true", help="pace the loop to wall-clock")
    run.add_argument("--explore", default="union", choices=("union", "frontier_first"))
    run.add_argument(
        "--condition",
        default="standard",
        choices=sorted(CONDITIONS),
        help="evaluation condition (harder arenas, step limit)",
    )

    batch = sub.add_parser("batch", help="evaluate many randomized seeds")
    batch.add_argument("--episodes", type=int, default=10)
    batch.add_argument("--seed", type=int, default=0, help="first seed")
    batch.add_argument(
        "--command",
        default=None,
        help="fixed command (default: a random object/destination per seed)",
    )
    batch.add_argument("--out", default="outputs/batch")
    batch.add_argument("--parser", default="llm", choices=("llm", "rule"), help=parser_help)
    batch.add_argument("--workers", type=int, default=4)
    batch.add_argument("--max-time", type=float, default=450.0)
    batch.add_argument("--no-person", action="store_true")
    batch.add_argument("--explore", default="union", choices=("union", "frontier_first"))
    batch.add_argument(
        "--condition",
        default="standard",
        choices=sorted(CONDITIONS),
        help="evaluation condition (harder arenas, step limit)",
    )

    inter = sub.add_parser("interactive", help="type commands for the robot in one arena")
    inter.add_argument("--seed", type=int, default=3)
    inter.add_argument("--out", default="outputs/interactive")
    inter.add_argument("--video", action="store_true", help="save an MP4 per command")
    inter.add_argument("--parser", default="llm", choices=("llm", "rule"), help=parser_help)
    inter.add_argument("--condition", default="standard", choices=sorted(CONDITIONS))

    parse = sub.add_parser("parse", help="just parse a command and print the result")
    parse.add_argument("text")
    parse.add_argument("--parser", default="llm", choices=("llm", "rule"), help=parser_help)
    return parser


def _cmd_run(args) -> int:
    from fetchbot.evaluation.episode import EpisodeConfig, FetchEpisode
    from fetchbot.export import ReportExporter

    out = Path(args.out)
    stem = f"run_seed{args.seed}" + ("" if args.condition == "standard" else f"_{args.condition}")
    cfg = EpisodeConfig(
        command=args.command,
        seed=args.seed,
        max_sim_time=args.max_time,
        with_person=not args.no_person,
        realtime=args.realtime,
        explore_strategy=args.explore,
        condition=args.condition,
        video_path=args.video,
        map_png=str(out / f"{stem}_map.png"),
        verbose=True,
        parser=args.parser,
    )
    try:
        ep = FetchEpisode(cfg)
    except ValueError as exc:
        print(f"could not understand command: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # no language model available
        print(f"parser unavailable: {exc}", file=sys.stderr)
        return 3
    print(f"parsed program ({args.parser}): {ep.program.describe()}")
    print(
        f"scene seed {args.seed}: objects {[o.label for o in ep.spec.objects]}, "
        f"zones {[z.name for z in ep.spec.zones]}"
    )
    if not ep.target_objects():
        print("warning: no object in this arena matches the command")
    try:
        result = ep.run()
    finally:
        ep.close()
    report = result.row() | {"seed": args.seed, "video": args.video or "", "events": result.events}
    rep = ReportExporter(out)
    rep.export(report, f"{stem}_report.json")
    rep.export(report, f"{stem}_report.md")
    for rec in result.step_log:
        print(
            f"  step {rec['index'] + 1}: {rec['step']}: {'ok' if rec['success'] else 'FAILED'}"
            f" delivered {rec['delivered']} {rec['cause']}"
        )
    print(
        f"{'SUCCESS' if result.success else 'FAILURE'} ({result.steps_ok}/{result.steps} "
        f"steps): sim time {result.time_s:.1f} s, "
        f"distance {result.distance_m:.1f} m, replans {result.replans} {result.failure}"
    )
    rate = cfg.control_hz
    print(
        f"control loop: fixed {rate:.0f} Hz sim tick; measured {result.ticks_per_s:.0f} "
        f"ticks/s wall-clock excluding video ({result.ticks_per_s / rate:.1f}x real time)"
        + (f"; video render {result.render_fps:.1f} fps" if args.video else "")
    )
    print(
        f"report: {out / (stem + '_report.json')}  map: {cfg.map_png}"
        + (f"  video: {args.video}" if args.video else "")
    )
    return 0 if result.success else 1


def _cmd_batch(args) -> int:
    from fetchbot.evaluation import BatchRunner

    runner = BatchRunner(
        args.episodes,
        args.out,
        seed=args.seed,
        command=args.command,
        workers=args.workers,
        episode_overrides={
            "max_sim_time": args.max_time,
            "with_person": not args.no_person,
            "explore_strategy": args.explore,
            "condition": args.condition,
            "parser": args.parser,
        },
    )

    def progress(r):
        print(
            f"seed {r.episode:3d} {'OK  ' if r.success else 'FAIL'} {r.time_s:6.1f}s "
            f"{r.distance_m:5.1f}m replans {r.replans:2d}  '{r.command}' {r.failure}"
        )

    metrics = runner.run(progress)
    print("summary:", metrics.summary())
    print(f"wrote {Path(args.out) / 'batch_episodes.csv'} and batch_summary.csv/json")
    return 0


def _cmd_interactive(args) -> int:
    from fetchbot.ui.interfaces import CliInterface
    from fetchbot.ui.session import InteractiveSession

    return InteractiveSession(
        CliInterface(),
        seed=args.seed,
        out_dir=args.out,
        video=args.video,
        parser=args.parser,
        condition=args.condition,
    ).loop()


def _cmd_parse(args) -> int:
    import json

    from fetchbot.language import make_parser

    parser = make_parser(args.parser)
    try:
        program = parser.parse_program(args.text)
    except ValueError as exc:
        print(f"not understood: {exc}")
        return 2
    except RuntimeError as exc:
        print(f"parser unavailable: {exc}")
        return 3
    print(program.describe())
    print(json.dumps({"steps": program.canonical()}, indent=1))
    print(f"({args.parser} parser, {parser.last_info.get('latency_s', 0.0):.2f} s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    args = build_arg_parser().parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "batch": _cmd_batch,
        "interactive": _cmd_interactive,
        "parse": _cmd_parse,
    }
    if args.cmd is None:
        build_arg_parser().print_help()
        return 0
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
