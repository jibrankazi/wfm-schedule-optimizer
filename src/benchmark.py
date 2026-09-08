"""A greedy scheduler, to measure what the MILP is actually worth.

Claiming an optimiser is better than a heuristic is worthless without a
heuristic to point at. This is that heuristic: the approach a competent
person builds in a spreadsheet, applied to the same roster, the same
templates and the same demand, and returning the same :class:`Schedule`
object so every metric is computed identically.

The baseline is deliberately a *good* greedy, not a strawman:

* it scores candidate shifts by deficit closed per paid hour, so it prefers
  the shift that buys the most coverage for the least labour;
* it honours every hard constraint the MILP does - availability, blackout
  windows, start-time accommodations, part-time eligibility, one shift per
  day, and the weekly hour ceiling;
* it makes a second pass to bring agents up to contracted minimum hours,
  placing those shifts where they cause the least oversupply, because the
  MILP is penalised for leaving them short and a comparison that ignored
  this would flatter the optimiser.

What it cannot do is reconsider. Every placement is final, so an early
choice that looked locally efficient can strand demand later in the week
with no agent left able to cover it. That gap is the thing being measured.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .generator import DAYS, INTERVALS_PER_DAY, Agent, ShiftTemplate
from .optimizer import Assignment, Schedule


def _required_matrix(demand: pd.DataFrame) -> np.ndarray:
    required = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    for row in demand.itertuples():
        required[row.day, row.interval] = row.required_agents
    return required


def solve_greedy(
    agents: list[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
    fill_contracted_minimum: bool = True,
) -> Schedule:
    """Assign shifts greedily and return the result as a :class:`Schedule`."""
    started = time.perf_counter()

    required = _required_matrix(demand)
    coverage = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    masks = {t.id: np.array(t.coverage, dtype=int) for t in templates}

    assignments: list[Assignment] = []
    hours = {a.id: 0.0 for a in agents}
    booked: set[tuple[int, int]] = set()  # (agent id, day)

    def eligible(agent: Agent, day: int, template: ShiftTemplate) -> bool:
        if (agent.id, day) in booked:
            return False
        if hours[agent.id] + template.paid_hours > agent.max_weekly_hours + 1e-9:
            return False
        return agent.can_work(day, template)

    def place(agent: Agent, day: int, template: ShiftTemplate) -> None:
        assignments.append(Assignment(agent.id, agent.name, day, template))
        hours[agent.id] += template.paid_hours
        booked.add((agent.id, day))
        coverage[day] += masks[template.id]

    # --- pass 1: close the coverage deficit -------------------------------
    while True:
        deficit = np.clip(required - coverage, 0, None)
        if not deficit.any():
            break

        best: tuple[float, int, int, ShiftTemplate] | None = None
        for day in range(DAYS):
            day_deficit = deficit[day]
            if not day_deficit.any():
                continue
            for template in templates:
                gain = int(np.minimum(day_deficit, masks[template.id]).sum())
                if gain <= 0:
                    continue
                score = gain / template.paid_hours
                if best is None or score > best[0]:
                    # Prefer the agent furthest below contracted hours, which
                    # is the same pressure the MILP's shortfall term applies.
                    candidates = [a for a in agents if eligible(a, day, template)]
                    if not candidates:
                        continue
                    candidates.sort(key=lambda a: (hours[a.id] - a.min_weekly_hours, a.id))
                    best = (score, gain, day, template)
                    best_agent = candidates[0]

        if best is None:
            break  # demand remains, but nobody may legally cover it
        place(best_agent, best[2], best[3])

    # --- pass 2: honour contracted minimums --------------------------------
    if fill_contracted_minimum:
        for agent in sorted(agents, key=lambda a: (-a.min_weekly_hours, a.id)):
            while hours[agent.id] < agent.min_weekly_hours - 1e-9:
                choice: tuple[int, int, ShiftTemplate] | None = None
                for day in range(DAYS):
                    for template in templates:
                        if not eligible(agent, day, template):
                            continue
                        surplus = int(
                            np.clip(
                                coverage[day] + masks[template.id] - required[day], 0, None
                            ).sum()
                        )
                        if choice is None or surplus < choice[0]:
                            choice = (surplus, day, template)
                if choice is None:
                    break
                place(agent, choice[1], choice[2])

    elapsed = time.perf_counter() - started

    return Schedule(
        assignments=assignments,
        coverage=coverage,
        required=required,
        agent_hours=hours,
        status="Greedy",
        objective=float(np.clip(required - coverage, 0, None).sum()),
        solve_seconds=elapsed,
        variables=0,
        shortfall_hours={
            a.id: round(a.min_weekly_hours - hours[a.id], 2)
            for a in agents
            if hours[a.id] < a.min_weekly_hours - 1e-9
        },
    )


def compare(
    agents: list[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
    time_limit_sec: int = 120,
    gap: float = 0.03,
) -> pd.DataFrame:
    """Run both schedulers on identical inputs and tabulate the difference."""
    from .optimizer import solve_schedule

    greedy = solve_greedy(agents, templates, demand)
    exact = solve_schedule(agents, templates, demand, time_limit_sec=time_limit_sec, gap=gap)

    rows = []
    for label, schedule in (("Greedy heuristic", greedy), ("MILP (CBC)", exact)):
        summary = schedule.summary()
        rows.append(
            {
                "scheduler": label,
                "solve_seconds": summary["solve_seconds"],
                "shifts": summary["shifts_assigned"],
                "scheduled_hours": summary["scheduled_hours"],
                "understaffed_agent_intervals": summary["understaffed_agent_intervals"],
                "overstaffed_agent_intervals": summary["overstaffed_agent_intervals"],
                "coverage_compliance_pct": summary["coverage_compliance_pct"],
                "agents_below_contracted_minimum": summary["agents_below_contracted_minimum"],
            }
        )
    return pd.DataFrame(rows)


def summarise_gap(comparison: pd.DataFrame) -> dict[str, float]:
    """Reduce the two-row comparison to the numbers worth quoting."""
    greedy = comparison.iloc[0]
    exact = comparison.iloc[1]
    understaffed = float(greedy.understaffed_agent_intervals)
    return {
        "greedy_understaffed": understaffed,
        "milp_understaffed": float(exact.understaffed_agent_intervals),
        "understaffing_reduction_pct": (
            round(100.0 * (understaffed - exact.understaffed_agent_intervals) / understaffed, 1)
            if understaffed
            else 0.0
        ),
        "greedy_overstaffed": float(greedy.overstaffed_agent_intervals),
        "milp_overstaffed": float(exact.overstaffed_agent_intervals),
        "overstaffing_reduction_pct": (
            round(
                100.0
                * (greedy.overstaffed_agent_intervals - exact.overstaffed_agent_intervals)
                / greedy.overstaffed_agent_intervals,
                1,
            )
            if greedy.overstaffed_agent_intervals
            else 0.0
        ),
        "greedy_seconds": float(greedy.solve_seconds),
        "milp_seconds": float(exact.solve_seconds),
    }


def _main() -> int:  # pragma: no cover - console entry point
    import json

    from .generator import build_shift_templates, generate_demand, generate_roster

    agents = generate_roster()
    templates = build_shift_templates()
    demand = generate_demand()

    table = compare(agents, templates, demand)
    print(table.to_string(index=False))
    print()
    print(json.dumps(summarise_gap(table), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
