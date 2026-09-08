#!/usr/bin/env python3
"""Export a readable schedule for every operating pattern."""
from __future__ import annotations
import argparse
import json, sys
from pathlib import Path
import pandas as pd
from src.optimizer_247 import solve_weekly_schedule
from src.patterns import PATTERNS, build_pattern_demand, build_pattern_options, pattern_roster
from src.schedule_views import render_day_grid, write_schedule_workbook

DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
def clock(slot: int, per_day: int = 96) -> str:
    m = (slot % per_day) * 15
    return f"{m//60:02d}:{m%60:02d}"

parser = argparse.ArgumentParser(description="Export a readable schedule for every operating pattern.")
parser.add_argument("--out", default="artifacts/schedules")
parser.add_argument("--time-limit", type=int, default=300)
args = parser.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
summary = []
for key, pattern in PATTERNS.items():
    agents = pattern_roster(pattern); options = build_pattern_options(pattern)
    demand = build_pattern_demand(pattern)
    sched = solve_weekly_schedule(agents, options, demand, grid=pattern.grid,
                                  time_limit_sec=args.time_limit, gap=0.03)
    rows = []
    for a in sched.assignments:
        o = a.option
        start = o.start_slot % pattern.intervals_per_day
        end = (o.start_slot + o.span) % pattern.intervals_per_day
        rows.append({
            "pattern": key, "agent_id": a.agent_id, "agent": a.agent_name,
            "day": DAYS[o.start_day], "shift_id": o.id,
            "start": clock(start), "end": clock(end) if end else "24:00",
            "paid_hours": o.paid_hours, "on_phone_intervals": len(o.coverage_slots),
            "break_intervals": len(o.break_slots), "overnight": o.is_night,
        })
    frame = pd.DataFrame(rows).sort_values(["day","start","agent"])
    frame.to_csv(out / f"schedule_{key}.csv", index=False)

    cov = pd.DataFrame({
        "day": [DAYS[s // pattern.intervals_per_day] for s in range(pattern.grid.total_intervals)],
        "time": [clock(s, pattern.intervals_per_day) for s in range(pattern.grid.total_intervals)],
        "required": sched.required, "scheduled": sched.coverage,
    })
    cov["delta"] = cov.scheduled - cov.required
    cov.to_csv(out / f"coverage_{key}.csv", index=False)

    # A grid and a workbook alongside the CSV: the CSV is what you store, the
    # grid is what a supervisor reads.
    render_day_grid(sched, day=0, title=f"Shift plan - {pattern.label} - Monday",
                    save_path=out / f"grid_{key}.png")
    write_schedule_workbook(sched, out / f"schedule_{key}.xlsx",
                            days=pattern.days, pattern_label=pattern.label)

    s = sched.summary()
    summary.append({"pattern": key, "roster": pattern.roster_size, "shifts": len(rows),
                    "paid_hours": float(frame.paid_hours.sum()),
                    "gap_pct": s["relative_gap_pct"],
                    "coverage_pct": s["coverage_compliance_pct"],
                    "understaffed": s["understaffed_agent_intervals"],
                    "floor_breaches": s["floor_breach_intervals"]})
    print(f"{key}: {len(rows)} shifts, {frame.paid_hours.sum():.0f} paid hours", flush=True)

(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
