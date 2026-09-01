#!/usr/bin/env python3
"""Run the whole pipeline and write the README assets.

    python run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generator import build_shift_templates, capacity_report, generate_demand, generate_roster
from src.optimizer import coverage_lower_bound, hours_frame, solve_schedule
from src.reporting import plot_coverage, plot_shift_gantt, plot_week_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve the weekly contact centre schedule.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=120, help="solver time limit, seconds")
    parser.add_argument("--gap", type=float, default=0.03, help="relative MIP gap to stop at")
    parser.add_argument("--step", type=int, default=2, help="shift start granularity in intervals")
    parser.add_argument("--assets", default="assets", help="directory for the PNGs")
    args = parser.parse_args()

    demand = generate_demand(seed=args.seed)
    agents = generate_roster(seed=args.seed)
    templates = build_shift_templates(step=args.step)

    print(f"demand    : {len(demand)} intervals, peak {demand.required_agents.max()} agents required")
    print(f"roster    : {len(agents)} agents, {len(templates)} shift templates")
    print(f"capacity  : {json.dumps(capacity_report(agents, templates, demand))}\n")

    bound = coverage_lower_bound(agents, templates, demand)
    print(f"lower bound (agents interchangeable): {json.dumps(bound)}\n")

    print("solving the assignment model...")
    schedule = solve_schedule(agents, templates, demand, time_limit_sec=args.time_limit, gap=args.gap)
    summary = schedule.summary()
    print(json.dumps(summary, indent=2))

    attributable = summary["understaffed_agent_intervals"] - bound["min_understaffed_agent_intervals"]
    print(
        f"\nof {summary['understaffed_agent_intervals']} understaffed agent-intervals, "
        f"{bound['min_understaffed_agent_intervals']} are inherent to the demand curve and "
        f"{attributable} come from individual availability."
    )

    assets = Path(args.assets)
    plot_coverage(schedule, day=0, save_path=assets / "coverage_vs_requirement.png")
    plot_week_coverage(schedule, save_path=assets / "coverage_week.png")
    plot_shift_gantt(schedule, day=0, save_path=assets / "shift_gantt.png")
    print(f"\nwrote charts to {assets}/")

    data = Path("data")
    data.mkdir(exist_ok=True)
    schedule.to_frame().to_csv(data / "schedule.csv", index=False)
    schedule.coverage_frame().to_csv(data / "coverage.csv", index=False)
    hours_frame(schedule, agents).to_csv(data / "agent_hours.csv", index=False)
    print(f"wrote schedule.csv, coverage.csv, agent_hours.csv to {data}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
