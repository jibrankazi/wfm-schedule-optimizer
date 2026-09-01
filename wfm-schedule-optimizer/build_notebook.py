#!/usr/bin/env python3
"""Build notebooks/schedule_optimization.ipynb.

The notebook is generated rather than hand-edited so it stays in step with
the source. Execute it afterwards to bake the outputs in:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/schedule_optimization.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("notebooks/schedule_optimization.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Contact Centre Workforce Scheduling

Two problems stacked on top of each other.

**How many agents does each 15-minute interval need?** That is a queueing question, answered
by Erlang C: given the calls arriving and how long each takes to handle, how many people must
be on the phone to answer 80% of them within 20 seconds without running the floor past 85%
occupancy.

**Which named people work which shifts?** That is a combinatorial one. Sixty agents, five days,
49 intervals a day, each person carrying their own contracted hours, booked leave and start-time
accommodations. It is solved here as a mixed-integer program.

The two are linked: the Erlang layer produces a requirement curve, and the MILP tries to lay
staffed hours on top of it without over- or under-shooting.

All data is synthetic and seeded. No real call volumes, no real employees.
"""),
    code("""
%matplotlib inline

import sys
from pathlib import Path

# make the package importable whether this runs from notebooks/ or the repo root
root = Path.cwd()
if not (root / "src").exists():
    root = root.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

from src.erlang import ServiceTarget, erlang_c, required_agents, service_level, traffic_intensity
from src.generator import build_shift_templates, capacity_report, generate_demand, generate_roster, roster_summary
from src.optimizer import coverage_lower_bound, hours_frame, solve_schedule
from src.reporting import plot_coverage, plot_shift_gantt, plot_week_coverage

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
SEED = 42
"""),
    md("""
## 1. Demand

A bimodal intraday curve: a late-morning peak, a softer mid-afternoon one, Monday heavy and
Friday light. Handle times vary around a 320-second mean.
"""),
    code("""
demand = generate_demand(seed=SEED)
demand.head(8)
"""),
    code("""
demand.groupby("day_name", sort=False).agg(
    calls=("calls", "sum"),
    peak_calls=("calls", "max"),
    mean_aht=("aht_sec", "mean"),
    peak_required=("required_agents", "max"),
    agent_intervals=("required_agents", "sum"),
).round(1)
"""),
    md("""
## 2. Erlang C sizing

Offered load in Erlangs is `calls × AHT / 900`. The queueing model turns that into a headcount
by asking, for each candidate number of agents, what fraction of calls would be answered inside
the target time.

The usual textbook formula is written with factorials and overflows a float above roughly
135 Erlangs, because `A^N` blows up long before the ratio does. `src/erlang.py` uses the
Erlang B recursion instead, which never forms that term. Same answer, no ceiling.
"""),
    code("""
target = ServiceTarget(service_level=0.80, target_time_sec=20.0, max_occupancy=0.85)

rows = []
for calls in [10, 25, 50, 75, 100, 150, 200]:
    intensity = traffic_intensity(calls, 320)
    agents = required_agents(calls, 320, target)
    rows.append({
        "calls_per_interval": calls,
        "erlangs": round(intensity, 1),
        "required_agents": agents,
        "service_level": round(service_level(agents, calls, 320), 3),
        "occupancy": round(intensity / agents, 3),
        "sl_at_one_fewer": round(service_level(agents - 1, calls, 320), 3),
    })
pd.DataFrame(rows)
"""),
    md("""
Note which constraint bites where. At low volume the service level is what forces headcount up
and occupancy sits comfortably below the cap. At high volume the queue enjoys economies of
scale — it hits 80/20 while still running agents at 89% — and the occupancy cap becomes the
binding constraint. Drop the cap and you get staffing numbers that satisfy the SLA and burn
the floor out.
"""),
    md("""
## 3. Roster and shift templates

Shifts are chosen from a set of legal templates rather than assembled from free per-interval
booleans. Unconstrained interval variables produce mathematically optimal, operationally absurd
rosters where somebody works 07:00–07:15 and again at 16:45.

Each template carries two different quantities, and conflating them is the classic way to build
a roster that looks compliant and still leaves the queue unmanned at 12:30:

- **paid hours** — what the agent is paid for, excluding the unpaid meal break
- **coverage** — the intervals they are actually on the phone, excluding meal *and* rest breaks
"""),
    code("""
templates = build_shift_templates(step=2)
agents = generate_roster(seed=SEED)

pd.DataFrame([{
    "id": t.id,
    "kind": t.kind,
    "window": t.label(),
    "on_site_intervals": t.span,
    "paid_hours": t.paid_hours,
    "on_phone_intervals": t.covered_intervals,
    "break_intervals": len(t.breaks),
} for t in templates]).head(10)
"""),
    code("""
summary = roster_summary(agents)
print(f"{len(agents)} agents | {len(templates)} shift templates")
print(summary.kind.value_counts().to_string())
print(f"\\naccommodations: {(summary.accommodation != '-').sum()}")
print(f"agents with booked leave: {(summary.blackout_intervals > 0).sum()}")
print(f"total blacked-out intervals: {summary.blackout_intervals.sum()}")
summary[summary.accommodation != "-"].head()
"""),
    md("""
## 4. Can this roster even cover the week?

Worth asking before the solver runs. If contracted hours cannot cover demand, the MILP will
still return a schedule — it will just be an understaffed one — and knowing that in advance
stops you blaming the optimiser for an establishment problem.
"""),
    code("""
capacity = capacity_report(agents, templates, demand)
for key, value in capacity.items():
    print(f"{key:38s} {value}")
"""),
    md("""
## 5. A lower bound before optimising

Solve the same coverage problem with agents treated as interchangeable — decide only *how many*
of each template run each day, with no identities, no leave, no accommodations. That relaxation
solves in under a second and bounds what any real schedule can achieve.

The gap between this bound and the full solve is the price of the workforce being made of
specific people rather than of interchangeable labour. Without it you cannot tell "the demand
curve is hard to staff" from "this particular roster cannot staff it".
"""),
    code("""
bound = coverage_lower_bound(agents, templates, demand)
bound
"""),
    md("""
## 6. The assignment model

    x[i, d, s] ∈ {0,1}   agent i works template s on day d
    under[d, t] ≥ 0      agents short of the requirement
    over[d, t]  ≥ 0      agents surplus to it
    short[i]    ≥ 0      hours below agent i's contracted minimum

    minimise  100·Σ under  +  1·Σ over  +  50·Σ short·4

Understaffing is the thing being avoided, so it is priced a hundred times a surplus agent.
Breaching a contracted minimum sits between the two.

Two deliberate departures from the textbook formulation:

**Availability is structural, not a constraint.** A variable only exists where the agent could
legally work that template, so leave and accommodations shrink the model instead of adding rows.

**The weekly minimum is soft; the maximum is hard.** A hard minimum combined with leave and
accommodations makes the model infeasible for reasons that have nothing to do with scheduling,
and an infeasible solve tells you nothing. Priced at 50 it still dominates surplus staffing, so
the solver only breaches a minimum when nothing else works — and then it reports who and by how much.
"""),
    code("""
schedule = solve_schedule(agents, templates, demand, time_limit_sec=180, gap=0.03)

for key, value in schedule.summary().items():
    print(f"{key:36s} {value}")
"""),
    md("""
### Where the residual understaffing comes from

With the relaxation bound in hand the shortfall splits into two parts with different remedies.
Structural shortfall is a demand-curve problem — it needs different shift shapes or more
establishment. Availability shortfall is a roster problem — different leave approvals would
move it.
"""),
    code("""
total_short = schedule.summary()["understaffed_agent_intervals"]
structural = bound["min_understaffed_agent_intervals"]

print(f"total understaffed agent-intervals   {total_short}")
print(f"  inherent to the demand curve       {structural}")
print(f"  caused by individual availability  {total_short - structural}")
print(f"\\nas hours: {total_short * 0.25:.2f}h short across a week of {int(demand.required_agents.sum()) * 0.25:.0f} required agent-hours")
"""),
    md("""
## 7. Coverage

The scheduled bars against the Erlang C requirement. Pink is where the schedule falls short.
"""),
    code("""
_ = plot_coverage(schedule, day=0, save_path=root / "assets" / "coverage_vs_requirement.png")
"""),
    code("""
_ = plot_week_coverage(schedule, save_path=root / "assets" / "coverage_week.png")
"""),
    md("""
## 8. What each agent actually does

Aggregate coverage can look right while the per-agent plan is unworkable, so it is worth
looking at the shifts themselves. The staggered starts the optimiser chose show up as a diagonal.
"""),
    code("""
_ = plot_shift_gantt(schedule, day=0, save_path=root / "assets" / "shift_gantt.png")
"""),
    code("""
schedule.to_frame().head(12)
"""),
    md("""
## 9. Checking the constraints actually held

An optimiser will happily return a schedule that violates something you forgot to encode.
These assertions re-derive the constraints from the assignments rather than trusting the model.
"""),
    code("""
by_id = {a.id: a for a in agents}
problems = []

seen = set()
for assignment in schedule.assignments:
    key = (assignment.agent_id, assignment.day)
    if key in seen:
        problems.append(f"{assignment.agent_name} double-booked on day {assignment.day}")
    seen.add(key)

for agent_id, hours in schedule.agent_hours.items():
    if hours > by_id[agent_id].max_weekly_hours + 1e-6:
        problems.append(f"{by_id[agent_id].name} over contracted maximum: {hours}h")

for assignment in schedule.assignments:
    agent = by_id[assignment.agent_id]
    if any(assignment.template.occupies(i) for i in agent.blackouts.get(assignment.day, set())):
        problems.append(f"{agent.name} scheduled during booked leave on day {assignment.day}")
    if not (agent.earliest_start <= assignment.template.start <= agent.latest_start):
        problems.append(f"{agent.name} start-window accommodation breached")
    if agent.kind == "part_time" and assignment.template.kind == "full_time":
        problems.append(f"{agent.name} is part-time but got a full-time shift")

rebuilt = np.zeros_like(schedule.coverage)
for assignment in schedule.assignments:
    rebuilt[assignment.day] += np.array(assignment.template.coverage, dtype=int)
if not np.array_equal(rebuilt, schedule.coverage):
    problems.append("reported coverage does not match the assignments")

print("\\n".join(problems) if problems else "All constraints hold.")
"""),
    md("""
## 10. Hours delivered against hours contracted
"""),
    code("""
hours = hours_frame(schedule, agents)
print(hours.groupby("kind").agg(
    agents=("agent", "count"),
    contracted_min=("min_hours", "sum"),
    contracted_max=("max_hours", "sum"),
    scheduled=("scheduled_hours", "sum"),
    shortfall=("shortfall", "sum"),
).round(1).to_string())

below = hours[hours.shortfall > 0]
print(f"\\nagents below their contracted minimum: {len(below)}")
below.head()
"""),
    md("""
## What this does not do

- **Shrinkage is modelled explicitly, not as a percentage uplift.** Breaks are carved out of the
  coverage vectors rather than added as a blanket factor. Unplanned absence and adherence are
  not modelled at all, so real-world coverage would run below what is shown here.
- **Demand is deterministic.** Each interval takes its expected call volume. A real forecast
  carries error, and a schedule optimised against a point estimate is fragile to it.
- **Erlang C assumes no abandonment and infinite queue patience.** Erlang A is the model that
  relaxes that, and it usually reduces required headcount.
- **Intervals are treated independently.** Erlang C is a steady-state model; calls queueing at
  10:15 and still waiting at 10:30 are not carried across.
- **Optimality is not proven.** With 60 largely interchangeable agents the model is highly
  symmetric, so the search stops at a 3% relative gap. The relaxation bound above is the honest
  measure of how much is left on the table.
"""),
]


def main() -> int:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
