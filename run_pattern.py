#!/usr/bin/env python3
"""Solve one operating pattern and record what happened.

    python run_pattern.py weekday_extended --time-limit 600

Every pattern goes through the same solver. The only thing that changes is the
configuration, which is the claim the pattern layer exists to demonstrate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.optimizer_247 import SolverTerminationError, solve_weekly_schedule
from src.patterns import (
    PATTERNS,
    build_pattern_demand,
    build_pattern_options,
    pattern_roster,
    sizing_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve one operating pattern.")
    parser.add_argument("pattern", choices=sorted(PATTERNS))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--gap", type=float, default=0.03)
    parser.add_argument("--out", default="artifacts/patterns")
    args = parser.parse_args()

    pattern = PATTERNS[args.pattern]
    sizing = sizing_report(pattern)
    print(json.dumps(sizing, indent=2), flush=True)

    agents = pattern_roster(pattern)
    options = build_pattern_options(pattern)
    demand = build_pattern_demand(pattern)

    record: dict[str, object] = {"sizing": sizing, "time_limit_sec": args.time_limit}
    started = time.perf_counter()
    try:
        schedule = solve_weekly_schedule(
            agents, options, demand,
            grid=pattern.grid,
            time_limit_sec=args.time_limit,
            gap=args.gap,
        )
        record["outcome"] = "solved"
        record["result"] = dict(schedule.summary())
    except SolverTerminationError as exc:
        # Not a crash. The gate refused to publish a schedule it could not
        # verify, which for the largest pattern is the finding rather than a
        # failure to report.
        record["outcome"] = "no_schedule"
        record["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        record["outcome"] = "error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
    record["wall_clock_sec"] = round(time.perf_counter() - started, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pattern.key}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
