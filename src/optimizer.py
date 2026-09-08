"""Mixed-integer shift assignment.

Decision: which shift template, if any, each agent works on each day.

    x[i, d, s] in {0, 1}     agent i works template s on day d
    under[d, t] >= 0         agents short of the Erlang C requirement
    over[d, t] >= 0          agents surplus to it
    short[i]   >= 0          hours below agent i's contracted minimum

    minimise  100 * sum(under) + 1 * sum(over) + 50 * sum(short) * 4

Subject to at most one shift per agent per day, a hard weekly maximum, and
exact interval coverage with the slack variables absorbing the difference.

Two departures from the textbook formulation, both deliberate:

* Availability is enforced structurally. A variable is only created where the
  agent could legally work that template, so blackout windows and start-time
  accommodations shrink the model instead of adding constraints to it.

* The weekly *minimum* is soft, the maximum is hard. A hard minimum plus
  blackouts and accommodations makes the model infeasible for reasons that
  have nothing to do with the schedule, and an infeasible solve tells you
  nothing. Priced at 50 it still dominates surplus staffing, so the solver
  only breaches a contracted minimum when nothing else will work - and then
  it reports which agents and by how much.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from .generator import DAY_NAMES, DAYS, INTERVALS_PER_DAY, Agent, ShiftTemplate, interval_label

UNDERSTAFF_PENALTY = 100.0
OVERSTAFF_PENALTY = 1.0
MIN_HOURS_PENALTY = 50.0


@dataclass
class Assignment:
    agent_id: int
    agent_name: str
    day: int
    template: ShiftTemplate


@dataclass
class Schedule:
    """The solved roster plus everything needed to judge it."""

    assignments: list[Assignment]
    coverage: np.ndarray  # (DAYS, INTERVALS_PER_DAY) agents on phone
    required: np.ndarray  # (DAYS, INTERVALS_PER_DAY) Erlang C requirement
    agent_hours: dict[int, float]
    status: str
    objective: float
    solve_seconds: float
    variables: int
    shortfall_hours: dict[int, float] = field(default_factory=dict)

    @property
    def understaffed(self) -> np.ndarray:
        return np.clip(self.required - self.coverage, 0, None)

    @property
    def overstaffed(self) -> np.ndarray:
        return np.clip(self.coverage - self.required, 0, None)

    def summary(self) -> dict:
        under = self.understaffed
        intervals = under.size
        return {
            "status": self.status,
            "solve_seconds": round(self.solve_seconds, 2),
            "binary_variables": self.variables,
            "shifts_assigned": len(self.assignments),
            "scheduled_hours": round(sum(self.agent_hours.values()), 1),
            "understaffed_intervals": int((under > 0).sum()),
            "understaffed_agent_intervals": int(under.sum()),
            "understaffed_hours": round(float(under.sum()) * 0.25, 2),
            "overstaffed_agent_intervals": int(self.overstaffed.sum()),
            "intervals_fully_covered": int((under == 0).sum()),
            "coverage_compliance_pct": round(100.0 * (under == 0).sum() / intervals, 1),
            "agents_below_contracted_minimum": len(self.shortfall_hours),
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "agent": a.agent_name,
                    "day": DAY_NAMES[a.day],
                    "shift": a.template.id,
                    "start": interval_label(a.template.start),
                    "end": interval_label(a.template.end),
                    "paid_hours": a.template.paid_hours,
                    "on_phone_intervals": a.template.covered_intervals,
                }
                for a in sorted(self.assignments, key=lambda x: (x.agent_id, x.day))
            ]
        )

    def coverage_frame(self) -> pd.DataFrame:
        rows = []
        for day in range(DAYS):
            for interval in range(INTERVALS_PER_DAY):
                rows.append(
                    {
                        "day": day,
                        "day_name": DAY_NAMES[day],
                        "interval": interval,
                        "time": interval_label(interval),
                        "required": int(self.required[day, interval]),
                        "scheduled": int(self.coverage[day, interval]),
                        "delta": int(self.coverage[day, interval] - self.required[day, interval]),
                    }
                )
        return pd.DataFrame(rows)


def build_model(
    agents: list[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
) -> tuple[pulp.LpProblem, dict, dict, dict, dict, np.ndarray]:
    """Assemble the MILP. Returns the problem and its variable dictionaries."""
    required = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    for row in demand.itertuples():
        required[row.day, row.interval] = row.required_agents

    model = pulp.LpProblem("contact_centre_schedule", pulp.LpMinimize)
    by_id = {t.id: t for t in templates}

    # Only create variables the agent could actually work.
    allowed: dict[tuple[int, int], list[str]] = {}
    x: dict[tuple[int, int, str], pulp.LpVariable] = {}
    for agent in agents:
        for day in range(DAYS):
            ids = [t.id for t in templates if agent.can_work(day, t)]
            allowed[(agent.id, day)] = ids
            for template_id in ids:
                x[(agent.id, day, template_id)] = pulp.LpVariable(
                    f"x_{agent.id}_{day}_{template_id}", cat=pulp.LpBinary
                )

    under = {
        (d, t): pulp.LpVariable(f"under_{d}_{t}", lowBound=0)
        for d in range(DAYS)
        for t in range(INTERVALS_PER_DAY)
    }
    over = {
        (d, t): pulp.LpVariable(f"over_{d}_{t}", lowBound=0)
        for d in range(DAYS)
        for t in range(INTERVALS_PER_DAY)
    }
    short = {a.id: pulp.LpVariable(f"short_{a.id}", lowBound=0) for a in agents}

    model += (
        UNDERSTAFF_PENALTY * pulp.lpSum(under.values())
        + OVERSTAFF_PENALTY * pulp.lpSum(over.values())
        + MIN_HOURS_PENALTY * pulp.lpSum(short.values()) * 4
    )

    # One shift per agent per day at most.
    for agent in agents:
        for day in range(DAYS):
            ids = allowed[(agent.id, day)]
            if ids:
                model += pulp.lpSum(x[(agent.id, day, i)] for i in ids) <= 1, f"one_shift_{agent.id}_{day}"

    # Weekly hours: hard ceiling, soft floor.
    for agent in agents:
        worked = pulp.lpSum(
            x[(agent.id, day, i)] * by_id[i].paid_hours
            for day in range(DAYS)
            for i in allowed[(agent.id, day)]
        )
        model += worked <= agent.max_weekly_hours, f"max_hours_{agent.id}"
        model += worked + short[agent.id] >= agent.min_weekly_hours, f"min_hours_{agent.id}"

    # Interval coverage.
    for day in range(DAYS):
        for interval in range(INTERVALS_PER_DAY):
            staffed = pulp.lpSum(
                x[(agent.id, day, i)] * by_id[i].coverage[interval]
                for agent in agents
                for i in allowed[(agent.id, day)]
                if by_id[i].coverage[interval]
            )
            model += (
                staffed + under[(day, interval)] - over[(day, interval)]
                == int(required[day, interval])
            ), f"coverage_{day}_{interval}"

    return model, x, under, over, short, required


def solve_schedule(
    agents: list[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
    time_limit_sec: int = 120,
    gap: float = 0.01,
    msg: bool = False,
) -> Schedule:
    """Build and solve, returning the roster and its coverage arrays.

    `gap` stops the search once the incumbent is provably within that
    fraction of optimal. With 60 largely interchangeable agents the model is
    highly symmetric, so proving optimality costs far more than finding a
    schedule that is good enough to run.
    """
    model, x, under, over, short, required = build_model(agents, templates, demand)
    by_id = {t.id: t for t in templates}

    started = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec, gapRel=gap))
    elapsed = time.perf_counter() - started

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(
            f"CBC returned {status!r}; refusing to publish an unverified schedule"
        )
    assignments: list[Assignment] = []
    coverage = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    agent_hours = {a.id: 0.0 for a in agents}
    by_agent = {a.id: a for a in agents}

    for (agent_id, day, template_id), variable in x.items():
        value = variable.value()
        if value is None or not math.isclose(value, round(value), abs_tol=1e-7):
            raise RuntimeError("CBC returned a fractional or missing assignment value")
        if round(value) == 1:
            template = by_id[template_id]
            assignments.append(
                Assignment(agent_id, by_agent[agent_id].name, day, template)
            )
            agent_hours[agent_id] += template.paid_hours
            coverage[day] += np.array(template.coverage, dtype=int)

    shortfall = {
        agent_id: round(variable.value(), 2)
        for agent_id, variable in short.items()
        if variable.value() and variable.value() > 1e-6
    }

    return Schedule(
        assignments=assignments,
        coverage=coverage,
        required=required,
        agent_hours=agent_hours,
        status=status,
        objective=float(pulp.value(model.objective) or 0.0),
        solve_seconds=elapsed,
        variables=len(x),
        shortfall_hours=shortfall,
    )


def coverage_lower_bound(
    agents: list[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
    time_limit_sec: int = 60,
) -> dict:
    """Best achievable coverage if agents were interchangeable.

    Same coverage constraints, but the decision is only *how many* of each
    template run each day - no agent identity, no blackouts, no
    accommodations. That relaxation solves in under a second and bounds what
    any real schedule can achieve.

    The difference between this bound and the solved schedule is the price of
    the workforce being made of specific people with specific constraints,
    rather than of interchangeable labour. It separates 'the demand curve is
    hard to staff' from 'this particular roster cannot staff it'.
    """
    required = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    for row in demand.itertuples():
        required[row.day, row.interval] = row.required_agents

    model = pulp.LpProblem("coverage_bound", pulp.LpMinimize)
    counts = {
        (d, t.id): pulp.LpVariable(f"n_{d}_{t.id}", lowBound=0, cat=pulp.LpInteger)
        for d in range(DAYS)
        for t in templates
    }
    under = {
        (d, i): pulp.LpVariable(f"bu_{d}_{i}", lowBound=0)
        for d in range(DAYS)
        for i in range(INTERVALS_PER_DAY)
    }
    over = {
        (d, i): pulp.LpVariable(f"bo_{d}_{i}", lowBound=0)
        for d in range(DAYS)
        for i in range(INTERVALS_PER_DAY)
    }

    model += UNDERSTAFF_PENALTY * pulp.lpSum(under.values()) + OVERSTAFF_PENALTY * pulp.lpSum(over.values())

    for day in range(DAYS):
        for interval in range(INTERVALS_PER_DAY):
            model += (
                pulp.lpSum(counts[(day, t.id)] * t.coverage[interval] for t in templates if t.coverage[interval])
                + under[(day, interval)]
                - over[(day, interval)]
                == int(required[day, interval])
            )
        model += pulp.lpSum(counts[(day, t.id)] for t in templates) <= len(agents)

    model += (
        pulp.lpSum(counts[(d, t.id)] * t.paid_hours for d in range(DAYS) for t in templates)
        <= sum(a.max_weekly_hours for a in agents)
    )

    started = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec, gapRel=0.005))
    elapsed = time.perf_counter() - started

    understaffed = sum(v.value() or 0.0 for v in under.values())
    return {
        "status": pulp.LpStatus[model.status],
        "solve_seconds": round(elapsed, 2),
        "min_understaffed_agent_intervals": int(round(understaffed)),
        "min_overstaffed_agent_intervals": int(round(sum(v.value() or 0.0 for v in over.values()))),
        "shifts_required": int(round(sum(v.value() or 0.0 for v in counts.values()))),
    }


def hours_frame(schedule: Schedule, agents: list[Agent]) -> pd.DataFrame:
    """Per-agent contracted versus scheduled hours."""
    return pd.DataFrame(
        [
            {
                "agent": a.name,
                "kind": a.kind,
                "min_hours": a.min_weekly_hours,
                "max_hours": a.max_weekly_hours,
                "scheduled_hours": schedule.agent_hours.get(a.id, 0.0),
                "shortfall": schedule.shortfall_hours.get(a.id, 0.0),
                "shifts": sum(1 for x in schedule.assignments if x.agent_id == a.id),
                "accommodation": a.accommodation or "-",
            }
            for a in agents
        ]
    )
