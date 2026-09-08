#!/usr/bin/env python3
"""Build or solve the synthetic continuous 24/7 planning case.

Examples:

    python run_247.py
    python run_247.py --solve --time-limit 180 --gap 0.05
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from pathlib import Path

from src.continuous import (
    build_weekly_shift_options,
    generate_continuous_demand,
    generate_continuous_roster,
)
from src.optimizer_247 import (
    SolverTerminationError,
    build_weekly_model,
    solve_weekly_schedule,
)


MAX_ARTIFACT_GAP_PCT = 60.0
MAX_ARTIFACT_FLOOR_BREACHES = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or solve the 24/7 weekly roster.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-calls", type=float, default=70.0)
    parser.add_argument("--solve", action="store_true", help="run CBC after building the model")
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--gap", type=float, default=0.03)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="directory for measured solve JSON and interval coverage CSV",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="small seeded instance that solves in seconds, for live demonstration",
    )
    parser.add_argument(
        "--force-artifacts",
        action="store_true",
        help="write artifacts even if the run fails the quality gate",
    )
    args = parser.parse_args()

    if args.demo:
        # A reduced, seeded instance so the solve finishes while someone is
        # watching. The full run takes about 190s, which is unusable live; the
        # published baseline in artifacts/ remains the full-scale evidence.
        # 30 agents against a light queue and a floor of one. The full
        # instance needs 65 agents to hold a floor of four across 672
        # intervals; shrinking the roster without also relaxing the floor
        # produces a demo that fails its own quality gate.
        from dataclasses import replace as _replace

        from src.continuous import CONTINUOUS_247_PROFILE

        agents = generate_continuous_roster(full_time=22, part_time=8)
        demand = generate_continuous_demand(
            seed=args.seed,
            peak_calls=14.0,
            profile=_replace(CONTINUOUS_247_PROFILE, min_presence_floor=1),
        )
        time_limit, gap = 15, 0.05
    else:
        agents = generate_continuous_roster()
        demand = generate_continuous_demand(seed=args.seed, peak_calls=args.peak_calls)
        time_limit, gap = args.time_limit, args.gap
    options = build_weekly_shift_options()
    model, x, *_ = build_weekly_model(agents, options, demand)

    build_summary = {
        "agents": len(agents),
        "intervals": len(demand),
        "shift_options": len(options),
        "binary_assignment_variables": len(x),
        "total_model_variables": model.numVariables(),
        "constraints": model.numConstraints(),
        "required_agent_intervals": int(demand.required_agents.sum()),
        "presence_floor_constraint_intervals": int(
            (demand.min_presence_floor > 0).sum()
        ),
        "floor_binding_intervals": int((demand.binding_constraint == "floor").sum()),
        "peak_required_agents": int(demand.required_agents.max()),
    }
    print(json.dumps(build_summary, indent=2))

    if not args.solve:
        print("\nModel built only. Add --solve to run CBC.")
        return 0

    try:
        schedule = solve_weekly_schedule(
            agents,
            options,
            demand,
            time_limit_sec=time_limit,
            gap=gap,
            msg=args.solver_log,
        )
    except SolverTerminationError as exc:
        print(f"\nNo schedule published: {exc}")
        return 2

    print("\nVerified schedule")
    solve_summary = schedule.summary()
    print(json.dumps(solve_summary, indent=2))

    # Quality gate on the artifacts.
    #
    # The termination gate rejects schedules that break a hard invariant. It
    # does not reject one that is merely far worse than what is already on
    # disk. During benchmarking an experimental run with a 96% gap and 65 floor
    # breaches silently overwrote a checked-in baseline of 52.8% and 1. A
    # committed artifact is a claim about the model, so degraded runs are
    # reported and discarded rather than published.
    if args.demo:
        # A demonstration run must never overwrite the published baseline, and
        # the gate misreads it anyway: on a near-perfect small instance the
        # objective approaches zero, so the relative gap divides by almost
        # nothing and reads 100%.
        print("\nDemo run - artifacts not written. "
              "The published baseline in artifacts/ is the full-scale evidence.")
        return 0

    rejections: list[str] = []
    if solve_summary.get("status") not in {"Optimal", "Feasible"}:
        rejections.append(f"status is {solve_summary.get('status')!r}")
    gap = solve_summary.get("relative_gap_pct")
    if gap is not None and gap > MAX_ARTIFACT_GAP_PCT:
        rejections.append(f"relative gap {gap:.1f}% exceeds {MAX_ARTIFACT_GAP_PCT:.0f}%")
    breaches = solve_summary.get("floor_breach_intervals", 0)
    if breaches > MAX_ARTIFACT_FLOOR_BREACHES:
        rejections.append(f"{breaches} floor breaches exceed {MAX_ARTIFACT_FLOOR_BREACHES}")
    if rejections:
        print("\nArtifacts NOT written - run rejected by the quality gate:")
        for reason in rejections:
            print(f"  - {reason}")
        print("The schedule above is valid but too degraded to publish as a baseline.")
        if not args.force_artifacts:
            print("Re-run with a longer --time-limit, or pass --force-artifacts to override.")
            return 3
        print("--force-artifacts set; writing anyway.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Agent-level assignments for the same run.
    #
    # Without this the roster charts have to borrow a sector export, so the
    # Gantt and hours figures describe a different solve from every other
    # number in the README. Exporting here keeps one run behind all of them.
    roster = pd.DataFrame(
        [
            {
                "agent_id": a.agent_id,
                "agent_name": a.agent_name,
                "shift_id": a.option.id,
                "day": a.option.start_day,
                "start_slot": a.option.start_slot,
                "span_slots": a.option.span,
                "paid_hours": a.option.paid_hours,
                "coverage_slots": len(a.option.coverage_slots),
                "break_slots": len(a.option.break_slots),
                "is_night": a.option.is_night,
            }
            for a in schedule.assignments
        ]
    )
    roster_path = output_dir / "assignments_247.csv"
    roster.to_csv(roster_path, index=False)

    measurement_path = output_dir / "cbc_baseline_247.json"
    measurement_path.write_text(
        json.dumps(
            {
                "run_configuration": {
                    "seed": args.seed,
                    "peak_calls": args.peak_calls,
                    "time_limit_sec": args.time_limit,
                    "requested_gap": args.gap,
                },
                "model_build": build_summary,
                "measured_result": solve_summary,
                "metric_definitions": {
                    "relative_gap_pct": (
                        "100 * (incumbent objective - lower bound) / "
                        "abs(incumbent objective)"
                    ),
                    "coverage_compliance_pct": (
                        "share of weekly intervals where scheduled_active is at "
                        "least required_agents"
                    ),
                    "demand_fulfilment_pct": (
                        "sum(min(scheduled_active, required_agents)) divided by "
                        "sum(required_agents)"
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    coverage_report = demand.copy()
    coverage_report["scheduled_active"] = schedule.coverage
    coverage_report["net_slack"] = schedule.coverage - schedule.required
    coverage_report["demand_shortfall"] = schedule.understaffed
    coverage_report["floor_shortfall"] = schedule.floor_shortfall
    coverage_report["meets_requirement"] = schedule.coverage >= schedule.required
    coverage_report["floor_breach"] = schedule.coverage < schedule.presence_floor
    coverage_path = output_dir / "coverage_report_247.csv"
    coverage_report.to_csv(coverage_path, index=False)
    print(f"\nWrote {measurement_path} and {coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
