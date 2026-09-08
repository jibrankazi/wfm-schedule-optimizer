#!/usr/bin/env python3
"""Render every chart for the 24/7 model from the published artifacts.

    python run_charts.py

Reads artifacts/coverage_report_247.csv and an agent assignment export, so the
figures always describe the run that is checked in rather than a fresh solve
that might differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import reporting_247 as charts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the 24/7 chart set.")
    parser.add_argument("--coverage", default="artifacts/coverage_report_247.csv")
    parser.add_argument("--assignments", default="artifacts/assignments_247.csv")
    parser.add_argument("--out", default="assets")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out = Path(args.out)
    coverage = pd.read_csv(args.coverage)

    written: list[str] = []

    def emit(name: str, figure_call) -> None:
        path = out / name
        figure_call(save_path=path, dpi=args.dpi)
        written.append(f"{path}  ({path.stat().st_size // 1024} KB)")

    emit("247_call_volume_week.png", lambda **kw: charts.plot_call_volume_week(coverage, **kw))
    emit("247_intraday_profile.png", lambda **kw: charts.plot_intraday_profile(coverage, **kw))
    emit("247_sizing_curve.png", lambda **kw: charts.plot_sizing_curve(coverage, **kw))
    emit("247_day_coverage.png", lambda **kw: charts.plot_day_coverage(coverage, day=0, **kw))
    emit("247_week_coverage.png", lambda **kw: charts.plot_week_coverage(coverage, **kw))
    emit("247_coverage_heatmap.png", lambda **kw: charts.plot_coverage_heatmap(coverage, **kw))
    emit("247_binding_constraint.png", lambda **kw: charts.plot_binding_constraint(coverage, **kw))
    emit("247_shortfall_by_hour.png", lambda **kw: charts.plot_shortfall_by_hour(coverage, **kw))

    assignments_path = Path(args.assignments)
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)
        emit("247_agent_gantt.png", lambda **kw: charts.plot_agent_gantt(assignments, day=0, **kw))
        emit("247_agent_hours.png", lambda **kw: charts.plot_agent_hours(assignments, **kw))
        emit("247_night_distribution.png",
             lambda **kw: charts.plot_night_distribution(assignments, **kw))
    else:
        print(f"note: {assignments_path} not found; roster charts skipped")

    print(f"wrote {len(written)} charts to {out}/")
    for line in written:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
