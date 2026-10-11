"""Run the batch evaluation under every condition in fetchbot.world.difficulty.

Writes one sub-directory per condition (episode CSV + summary) plus
``conditions_summary.csv`` / ``conditions_report.md`` with success rates and the
failure reason of every failed episode.

    python scripts/run_conditions.py --episodes 20 --seed 200 --out outputs/conditions
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from fetchbot.evaluation import BatchRunner
from fetchbot.world.difficulty import CONDITIONS


def failure_kind(reason: str) -> str:
    """Coarse failure category from the episode's failure string."""
    r = reason.lower()
    if r.startswith("step limit"):
        return "step limit"
    if r.startswith("timeout"):
        return "time limit"
    if "not inside" in r:
        return "wrong place / wrong object"
    for key in ("explor", "approach", "grab", "deliver", "path", "navig"):
        if key in r:
            return key
    return reason or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/conditions")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--parser", default="llm", choices=("llm", "rule"))
    args = ap.parse_args()
    out = Path(args.out)
    rows, fails = [], []
    for name in args.conditions:
        runner = BatchRunner(
            args.episodes,
            out / name,
            seed=args.seed,
            workers=args.workers,
            episode_overrides={"condition": name, "parser": args.parser},
        )
        m = runner.run()
        s = m.summary()
        ok_ticks = [r.ticks for r in m.results if r.success]
        kinds = Counter(failure_kind(r.failure) for r in m.results if not r.success)
        row = {
            "condition": name,
            **s,
            "with_person": sum(r.person for r in m.results),
            "false_detections": sum(r.false_detections for r in m.results),
            "max_ticks_ok": max(ok_ticks) if ok_ticks else 0,
            "failures": "; ".join(f"{k} x{v}" for k, v in kinds.most_common()),
        }
        rows.append(row)
        print(row, flush=True)
        for r in m.results:
            if not r.success:
                fails.append((name, r.episode, r.command, r.time_s, r.grabbed, r.failure))
    with open(out / "conditions_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Evaluation by condition",
        "",
        f"Seeds {args.seed}-{args.seed + args.episodes - 1}, random command per seed. "
        "Mean time/distance are over successful episodes.",
        "",
        "| condition | success | mean time (s) | mean dist (m) | mean replans | "
        "person present | false detections | failures |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['successes']}/{r['episodes']} | "
            f"{r['mean_time_s']} | {r['mean_distance_m']} | {r['mean_replans']} | "
            f"{r['with_person']}/{r['episodes']} | {r['false_detections']} | "
            f"{r['failures'] or '-'} |"
        )
    lines += [
        "",
        "## Failed episodes",
        "",
        "| condition | seed | command | sim time (s) | grabbed | reason |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {c} | {s} | {cmd} | {t} | {g or '-'} | {why} |" for c, s, cmd, t, g, why in fails]
    (out / "conditions_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'conditions_summary.csv'} and conditions_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
