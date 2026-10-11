"""Evaluation of the command language: parsers, shape classifier, end-to-end programs.

    python scripts/eval_language.py parsers  --set commands.json --out outputs/lang/parsers
    python scripts/eval_language.py shapes   --train-seeds 1000-1029 --test-seeds 5000-5019
    python scripts/eval_language.py episodes --seeds 7000-7009 --parser llm --workers 4

parsers:  a command counts as correct if the parsed program equals the expected one
          exactly (every field of every step, in order), or, for commands labeled
          invalid, if the parser rejects it (raises ValueError).
shapes:   robot-camera views of the ``shapes`` catalog (same-color objects of different
          shapes) at random poses 0.6-3.0 m from each object; the object's color mask
          component is labeled with its true shape. Train and test seeds are disjoint.
episodes: held-out seeds per category (multi_step, all, spatial, shape); the command
          for a seed is fixed by a template, parsed by the chosen parser and executed;
          success is checked against simulator ground truth per step.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from fetchbot.language import LLMConfig, LLMParser, RuleBasedParser, validate_program

try:  # the simulator stack; `parsers` (e.g. against a remote llama-server) needs only pydantic
    import numpy as np

    from fetchbot.data.pybullet_streams import CameraFrame
    from fetchbot.perception import ColorDetector, rgb_to_hsv
    from fetchbot.perception.projection import backproject
    from fetchbot.perception.shape import (
        SHAPES,
        ShapeClassifier,
        connected_components,
        shape_features,
        touches_border,
    )
    from fetchbot.world import PyBulletWorld, SceneRandomizer, get_condition
except ImportError:  # pragma: no cover
    pass

DATA = Path(__file__).resolve().parents[1] / "fetchbot/evaluation/data"


def parse_range(r: str) -> list[int]:
    a, b = r.split("-")
    return list(range(int(a), int(b) + 1))


# ---------------------------------------------------------------- parsers
def _build(kind: str, config: LLMConfig):
    if kind == "rule":
        return RuleBasedParser()
    return LLMParser(config=config)


def load_commands(name: str) -> list[dict]:
    """Labeled commands; ``expected`` is expanded to the canonical (full-key) form."""
    cmds = json.loads((DATA / name).read_text())["commands"]
    for c in cmds:
        if c["expected"]:
            c["expected"] = validate_program({"steps": c["expected"]}).canonical()
    return cmds


def cmd_parsers(args) -> int:
    """Exact-match accuracy + latency of the parsers on a labeled command set."""
    cfg = LLMConfig()
    if args.model:
        cfg.model_path = args.model
    if args.threads:
        cfg.n_threads = args.threads
    if args.server:
        cfg.server_url = args.server
    cfg.thinking = args.thinking
    cfg.near_synonyms = args.near_synonyms
    cmds = load_commands(args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for kind in args.parsers.split(","):
        parser = _build(kind, cfg)
        if kind.startswith("llm") and not parser.available():
            print(f"{kind}: skipped ({parser.load_error})")
            continue
        # Per-command checkpoint: an interrupted run resumes where it stopped.
        ckpt = out / f"rows_{kind}.jsonl"
        done = {}
        if ckpt.exists():
            for line in ckpt.read_text().splitlines():
                r = json.loads(line)
                done[r["text"]] = r
        if kind.startswith("llm") and len(done) < len(cmds):  # warm-up, not timed
            parser.parse_program("bring the red box to the dock")
        rows, lat = [], []
        per_cat = defaultdict(lambda: [0, 0])
        used = defaultdict(int)
        for c in cmds:
            if c["text"] in done:
                r = done[c["text"]]
                rows.append(r)
                lat.append(r["latency_s"])
                per_cat[c["category"]][0] += r["ok"]
                per_cat[c["category"]][1] += 1
                used[r.get("outcome", "parsed" if r["got"] is not None else "rejected")] += 1
                continue
            t0 = time.perf_counter()
            try:
                prog = parser.parse_program(c["text"])
                got = prog.canonical()
                err = ""
            except ValueError as exc:
                got, err = None, str(exc)[:300]
            except RuntimeError as exc:  # e.g. the llama-server died: counted as a miss
                got, err = None, f"backend error: {exc}"[:300]
            dt = time.perf_counter() - t0
            lat.append(dt)
            ok = got == c["expected"]
            per_cat[c["category"]][0] += ok
            per_cat[c["category"]][1] += 1
            info = getattr(parser, "last_info", {})
            outcome = "parsed" if got is not None else f"rejected:{info.get('rejected', 'rule')}"
            used[outcome] += 1
            rows.append(
                {
                    "category": c["category"],
                    "text": c["text"],
                    "ok": ok,
                    "expected": c["expected"],
                    "got": got,
                    "error": err,
                    "latency_s": round(dt, 3),
                    "attempts": info.get("attempts", 1),
                    "raw_outputs": info.get("raw_outputs", []),
                    "reasoning": info.get("reasoning", []),
                    "blank": info.get("blank", 0),
                    "empty": info.get("empty", 0),
                    "outcome": outcome,
                }
            )
            with ckpt.open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            print(f"{kind} {len(rows)}/{len(cmds)} ok={ok} {dt:.1f}s {c['text']}", flush=True)
        acc = sum(r["ok"] for r in rows) / len(rows)
        summary = {
            "parser": kind,
            "n": len(rows),
            "exact_match": round(acc, 3),
            "mean_latency_s": round(statistics.fmean(lat), 3),
            "median_latency_s": round(statistics.median(lat), 3),
            "max_latency_s": round(max(lat), 3),
            "per_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_cat.items())},
            "paths": dict(used),
            "blank_outputs": sum(r.get("blank", 0) for r in rows),
            "commands_with_blank": sum(bool(r.get("blank")) for r in rows),
            # empty plan(s) first, then a correct parse after the retry feedback
            "empty_then_correct": sum(
                bool(r.get("empty")) and r["ok"] and r["got"] is not None for r in rows
            ),
            # a command that must be rejected, but a program came out
            "false_accepts": sum(r["expected"] is None and r["got"] is not None for r in rows),
            "retried": sum(r.get("attempts", 1) > 1 for r in rows),
        }
        if kind.startswith("llm"):
            summary["model"] = Path(cfg.model_path).name
            summary["thinking"] = cfg.thinking
            summary["near_synonyms"] = cfg.near_synonyms
            summary["backend"] = cfg.server_url or f"llama-cpp-python, {cfg.n_threads} CPU threads"
        table.append(summary)
        print(json.dumps(summary))
        (out / f"rows_{kind.replace('+', '_')}.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    lines = [
        "| parser | exact match | "
        + " | ".join(sorted(table[0]["per_category"]))
        + " | mean latency (s) |",
        "|---" * (3 + len(table[0]["per_category"])) + "|",
    ]
    for s in table:
        lines.append(
            f"| {s['parser']} | {100 * s['exact_match']:.1f}% | "
            + " | ".join(s["per_category"][k] for k in sorted(s["per_category"]))
            + f" | {s['mean_latency_s']:.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------- shapes
def render_frame(world, x, y, yaw, width=160, height=120, fov=60.0, pitch=-18.0):
    c, s = math.cos(yaw), math.sin(yaw)
    pos = np.array([x + 0.17 * c, y + 0.17 * s, 0.5])
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    fwd = np.array([c * cp, s * cp, sp])
    right = np.array([s, -c, 0.0])
    up = np.cross(right, fwd)
    rgb, depth = world.render(pos, pos + fwd, width, height, fov, near=0.05, far=10.0)
    fy = (height / 2) / math.tan(math.radians(fov) / 2)
    return CameraFrame(rgb, depth, pos, fwd, right, up, fy, fy, 0.0)


def collect(seeds, views_per_object=12, rng_seed=0):
    det = ColorDetector()
    rng = np.random.default_rng(rng_seed)
    out = []
    for seed in seeds:
        spec = SceneRandomizer(
            seed=seed, with_person=False, difficulty=get_condition("shapes")
        ).randomize()
        world = PyBulletWorld()
        world.build(spec)
        for o in spec.objects:
            ox, oy = world.object_xy(o.name)
            for _ in range(views_per_object):
                d = float(rng.uniform(0.6, 3.0))
                a = float(rng.uniform(-math.pi, math.pi))
                cx, cy = ox + d * math.cos(a), oy + d * math.sin(a)
                if not (0.3 < cx < spec.size[0] - 0.3 and 0.3 < cy < spec.size[1] - 0.3):
                    continue
                yaw = math.atan2(oy - cy, ox - cx) + float(rng.uniform(-0.35, 0.35))
                frame = render_frame(world, cx, cy, yaw)
                m = det.mask(rgb_to_hsv(frame.rgb), o.color)
                for comp in connected_components(m, min_pixels=20):
                    vs, us = np.nonzero(comp)
                    pts = backproject(frame, us, vs)
                    px, py = np.median(pts[:, 0]), np.median(pts[:, 1])
                    if math.hypot(px - ox, py - oy) > 0.2 or touches_border(comp):
                        continue
                    f = shape_features(comp, frame)
                    if f is not None:
                        out.append((f, o.shape, seed, d))
        world.close()
    return out


def cmd_shapes(args) -> int:
    """Fit the shape classifier on train seeds, report accuracy on disjoint test seeds."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = collect(parse_range(args.train_seeds), rng_seed=1)
    test = collect(parse_range(args.test_seeds), rng_seed=2)
    model = ShapeClassifier.fit([(f, s) for f, s, *_ in train])
    clf = ShapeClassifier(model)
    res = {"train_n": len(train), "test_n": len(test), "model": model}
    for name, data in (("train", train), ("test", test)):
        conf = {t: Counter() for t in SHAPES}
        for f, s, *_ in data:
            conf[s][clf.predict(f) or "none"] += 1
        acc = sum(conf[s][s] for s in SHAPES) / max(1, len(data))
        res[f"{name}_accuracy"] = round(acc, 3)
        res[f"{name}_confusion"] = {s: dict(c) for s, c in conf.items()}
        # Accuracy by distance band (test only is interesting).
        bands = {}
        for lo, hi in ((0.6, 1.2), (1.2, 2.0), (2.0, 3.0)):
            sel = [(f, s) for f, s, _, d in data if lo <= d < hi]
            if sel:
                bands[f"{lo}-{hi} m"] = round(
                    sum(clf.predict(f) == s for f, s in sel) / len(sel), 3
                )
        res[f"{name}_by_distance"] = bands
    (out / "shape_eval.json").write_text(json.dumps(res, indent=1))
    (out / "shape_samples.json").write_text(
        json.dumps(
            {
                "train": [(f, s, sd, d) for f, s, sd, d in train],
                "test": [(f, s, sd, d) for f, s, sd, d in test],
            }
        )
    )
    print(json.dumps({k: v for k, v in res.items() if k != "model"}, indent=1))
    print("model:", json.dumps(model))
    return 0


# ---------------------------------------------------------------- end-to-end episodes
#: Held-out seeds for the end-to-end program episodes (never used during development).
EPISODE_SEEDS = range(7000, 7010)


def _episode_commands(category: str, seed: int) -> tuple[str, str, dict]:
    """(command, condition, expected program) for one held-out episode.

    The template and objects are a deterministic function of the seed, so the set is
    fixed before any episode runs."""
    zones = ["dock", "table", "kitchen"]
    d1, d2 = zones[seed % 3], zones[(seed + 1) % 3]

    def step(color, shape, dest, quantifier="one", relation=None, anchor=None):
        return {
            "color": color,
            "shape": shape,
            "quantifier": quantifier,
            "relation": relation,
            "anchor": anchor,
            "destination": dest,
        }

    if category == "multi_step":
        objs = [("red", "box"), ("blue", "cylinder"), ("green", "ball"), ("yellow", "box")]
        a, b = objs[seed % 4], objs[(seed + 1 + seed // 4) % 4]
        if a == b:
            b = objs[(seed + 2) % 4]
        word = {"cylinder": "mug"}
        text = (
            f"grab the {a[0]} {word.get(a[1], a[1])} and take it to the {d1}, "
            f"then bring the {b[0]} {word.get(b[1], b[1])} to the {d2}"
        )
        return text, "standard", {"steps": [step(*a, d1), step(*b, d2)]}
    if category == "all":
        if seed % 2 == 0:
            return (
                f"bring every red object to the {d1}",
                "shapes",
                {"steps": [step("red", None, d1, "all")]},
            )
        return (
            f"put all the boxes in the {d1}",
            "standard",
            {"steps": [step(None, "box", d1, "all")]},
        )
    if category == "spatial":
        k = seed % 3
        if k == 0:
            return (
                f"bring the box nearest the shelf to the {d1}",
                "standard",
                {"steps": [step(None, "box", d1, relation="nearest_to", anchor="shelf")]},
            )
        if k == 1:
            far = zones[(seed + 2) % 3]
            return (
                f"bring the box farthest from the {far} to the {d1}",
                "standard",
                {"steps": [step(None, "box", d1, relation="farthest_from", anchor=far)]},
            )
        return (
            f"take the leftmost box to the {d1}",
            "standard",
            {"steps": [step(None, "box", d1, relation="leftmost")]},
        )
    if category == "shape":
        target = [("red", "ball"), ("red", "box"), ("green", "box"), ("green", "ball")][seed % 4]
        return (
            f"bring the {target[0]} {target[1]} to the {d1}",
            "shapes",
            {"steps": [step(*target, d1)]},
        )
    raise ValueError(category)


def _run_program_episode(job):
    from fetchbot.evaluation.episode import EpisodeConfig, FetchEpisode

    category, seed, text, condition, program, max_time = job
    cfg = EpisodeConfig(command=text, seed=seed, condition=condition, max_sim_time=max_time)
    ep = FetchEpisode(cfg, program)
    try:
        r = ep.run()
    finally:
        ep.close()
    return {
        "category": category,
        "seed": seed,
        "command": text,
        "condition": condition,
        "program": r.program,
        "success": r.success,
        "steps": r.steps,
        "steps_ok": r.steps_ok,
        "time_s": r.time_s,
        "failure": r.failure,
        "step_log": r.step_log,
    }


def cmd_episodes(args) -> int:
    """End-to-end episodes on held-out seeds, commands parsed by the chosen parser."""
    from concurrent.futures import ProcessPoolExecutor

    from fetchbot.language import TaskProgram, make_parser, validate_program

    # Optional parse cache (JSONL, one line per command text): parse once, e.g. against a
    # GPU llama-server elsewhere with --parse-only, then run the episodes from the cache.
    cache: dict[str, dict] = {}
    if args.cache and Path(args.cache).exists():
        for line in Path(args.cache).read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                cache[c["text"]] = c
    parser = None

    def parse(text: str) -> TaskProgram:
        nonlocal parser
        if text not in cache:
            if parser is None:
                config = LLMConfig()
                if args.model:
                    config.model_path = args.model
                if args.server:
                    config.server_url = args.server
                parser = make_parser(args.parser, config=config)
            entry = {"text": text, "parser": args.parser, "program": None, "error": None}
            try:
                entry["program"] = {"steps": parser.parse_program(text).canonical()}
            except ValueError as exc:
                entry["error"] = str(exc)
            entry["latency_s"] = round(parser.last_info.get("latency_s", 0.0), 2)
            if args.parser == "llm":
                entry["model"] = Path(parser.config.model_path).name
            cache[text] = entry
            if args.cache:
                with open(args.cache, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        entry = cache[text]
        if entry["program"] is None:
            raise ValueError(entry["error"])
        return validate_program(entry["program"], raw_text=text)

    seeds = parse_range(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Per-episode checkpoint: a rerun skips the episodes already done.
    ckpt = out / "episodes.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            r = json.loads(line)
            done[(r["category"], r["seed"])] = r
    jobs, rows = [], []
    # Seed-major order, so a partial run covers every category.
    for seed in seeds:
        for category in args.categories:
            if (category, seed) in done:
                rows.append(done[(category, seed)])
                continue
            text, condition, expected = _episode_commands(category, seed)
            exp = validate_program(expected, raw_text=text)
            row = {
                "category": category,
                "seed": seed,
                "command": text,
                "condition": condition,
                "expected": exp.describe(),
            }
            try:
                prog: TaskProgram = parse(text)
            except ValueError as exc:
                rows.append(
                    row
                    | {
                        "success": False,
                        "steps": len(exp.steps),
                        "steps_ok": 0,
                        "parse_ok": False,
                        "time_s": 0.0,
                        "failure": f"parse rejected: {exc}",
                    }
                )
                with ckpt.open("a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
                continue
            row["parse_ok"] = prog.canonical() == exp.canonical()
            row["parse_latency_s"] = cache.get(text, {}).get("latency_s", 0.0)
            jobs.append((row, (category, seed, text, condition, prog, args.max_time)))
    if args.parse_only:
        print(f"parsed {len(cache)} distinct commands into {args.cache}")
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, res in zip(
            [j[0] for j in jobs], pool.map(_run_program_episode, [j[1] for j in jobs]), strict=True
        ):
            rows.append(row | res)
            with ckpt.open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            print(
                f"{res['category']:10s} seed {res['seed']}: "
                f"{'ok ' if res['success'] else 'FAIL'} {res['steps_ok']}/{res['steps']} "
                f"{res['time_s']:6.1f}s {res['failure']}",
                flush=True,
            )
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| category | success | steps ok | parse exact | mean time (s) | failures |",
        "|---|---|---|---|---|---|",
    ]
    for category in args.categories:
        rs = [r for r in rows if r["category"] == category]
        ok = [r for r in rs if r["success"]]
        fails = Counter(_failure_kind(r["failure"]) for r in rs if not r["success"])
        t = sum(r["time_s"] for r in ok) / len(ok) if ok else 0.0
        lines.append(
            f"| {category} | {len(ok)}/{len(rs)} | {sum(r['steps_ok'] for r in rs)}/"
            f"{sum(r['steps'] for r in rs)} | {sum(bool(r.get('parse_ok')) for r in rs)}/"
            f"{len(rs)} | {t:.1f} | {', '.join(f'{k} x{v}' for k, v in fails.items()) or '-'} |"
        )
    (out / "episodes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _failure_kind(failure: str) -> str:
    f = failure.split(": ", 1)[-1] if failure.startswith("step") else failure
    for key, kind in (
        ("parse", "parse"),
        ("wrong object", "wrong object"),
        ("time limit", "time limit"),
        ("timeout", "time limit"),
        ("found no", "not found"),
        ("unsatisfiable", "unsatisfiable"),
        ("missed", "missed objects"),
        ("grab", "grab failed"),
    ):
        if key in f:
            return kind
    return f[:40] or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parsers", help="parser exact-match accuracy and latency")
    p.add_argument("--out", default="outputs/lang/parsers")
    p.add_argument("--parsers", default="rule,llm")
    p.add_argument("--model", default=None, help="GGUF path (default: LLMConfig)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--server", default=None, help="llama-server URL (default: in-process)")
    p.add_argument("--set", default="commands.json", help="file in fetchbot/evaluation/data")
    p.add_argument("--thinking", action="store_true", help="Gemma 4 thinking (llama-server)")
    p.add_argument("--near-synonyms", action="store_true", help="near-synonym prompt rule")
    s = sub.add_parser("shapes", help="shape classifier accuracy (refit on train seeds)")
    s.add_argument("--train-seeds", default="1000-1029")
    s.add_argument("--test-seeds", default="5000-5019")
    s.add_argument("--out", default="outputs/lang/shapes")
    e = sub.add_parser("episodes", help="end-to-end program episodes on held-out seeds")
    e.add_argument("--seeds", default="7000-7009")
    e.add_argument("--categories", nargs="*", default=["multi_step", "all", "spatial", "shape"])
    e.add_argument("--parser", default="llm", choices=("llm", "rule"))
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-time", type=float, default=600.0)
    e.add_argument("--out", default="outputs/lang/episodes")
    e.add_argument("--model", default=None, help="GGUF path (default: LLMConfig)")
    e.add_argument("--server", default=None, help="llama-server URL (default: in-process)")
    e.add_argument("--cache", default=None, help="JSONL parse cache (read and appended)")
    e.add_argument("--parse-only", action="store_true", help="fill --cache, run no episodes")
    args = ap.parse_args()
    return {"parsers": cmd_parsers, "shapes": cmd_shapes, "episodes": cmd_episodes}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
