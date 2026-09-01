"""Synthetic demand, roster and shift templates.

Everything here is invented. No real call volumes, no real employees, no real
contact centre. The generator is seeded so the notebook, the README chart and
the test suite all describe the same week rather than a fresh random one each
run.

Time grid: 5 days, 49 intervals per day of 15 minutes, 07:00 to 19:15.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .erlang import ServiceTarget, required_agents, traffic_intensity

DAYS = 5
INTERVALS_PER_DAY = 49
DAY_START_MINUTES = 7 * 60  # 07:00
INTERVAL_MINUTES = 15
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

DEFAULT_SEED = 42


def interval_label(index: int) -> str:
    minutes = DAY_START_MINUTES + index * INTERVAL_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def interval_labels() -> list[str]:
    return [interval_label(i) for i in range(INTERVALS_PER_DAY)]


# ---------------------------------------------------------------------------
# demand
# ---------------------------------------------------------------------------
def generate_demand(
    seed: int = DEFAULT_SEED,
    target: ServiceTarget | None = None,
    peak_calls: float = 110.0,
) -> pd.DataFrame:
    """One row per (day, interval) with calls, AHT and required headcount.

    The intraday curve is bimodal - a late-morning peak and a softer
    mid-afternoon one - which is the usual shape for an inbound queue serving
    business hours. Monday runs hot, Friday runs light.
    """
    target = target or ServiceTarget()
    rng = np.random.default_rng(seed)

    position = np.linspace(0.0, 1.0, INTERVALS_PER_DAY)
    morning = np.exp(-(((position - 0.28) / 0.16) ** 2))
    afternoon = 0.78 * np.exp(-(((position - 0.66) / 0.20) ** 2))
    shape = morning + afternoon
    shape = shape / shape.max()

    day_factor = np.array([1.14, 1.04, 0.99, 0.95, 0.84])  # Mon heavy, Fri light

    rows = []
    for day in range(DAYS):
        volume = peak_calls * day_factor[day] * shape
        volume = volume + rng.normal(0.0, 2.2, INTERVALS_PER_DAY)
        volume = np.clip(np.round(volume), 0, None)

        aht = np.clip(rng.normal(320, 22, INTERVALS_PER_DAY), 180, 600)

        for interval in range(INTERVALS_PER_DAY):
            calls = float(volume[interval])
            handle_time = float(aht[interval])
            rows.append(
                {
                    "day": day,
                    "day_name": DAY_NAMES[day],
                    "interval": interval,
                    "time": interval_label(interval),
                    "calls": calls,
                    "aht_sec": round(handle_time, 1),
                    "erlangs": round(traffic_intensity(calls, handle_time), 2),
                    "required_agents": required_agents(calls, handle_time, target),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# shift templates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShiftTemplate:
    """A legal shift shape: when it starts, and which intervals it staffs.

    `coverage` is the on-phone mask. It is not the same as paid time: rest
    breaks are paid but uncovered, and the meal break is neither. Conflating
    the two is the classic way to build a roster that looks compliant and
    still leaves the queue unmanned at 12:30.
    """

    id: str
    kind: str  # full_time | part_time
    start: int
    span: int  # intervals on site, breaks included
    paid_hours: float
    coverage: tuple[int, ...]
    breaks: tuple[int, ...]  # absolute interval indices that are off-phone

    @property
    def end(self) -> int:
        return self.start + self.span

    @property
    def covered_intervals(self) -> int:
        return sum(self.coverage)

    def label(self) -> str:
        return f"{interval_label(self.start)}-{interval_label(self.end)}"

    def occupies(self, interval: int) -> bool:
        """On site during this interval, whether on-phone or on break."""
        return self.start <= interval < self.end


# (paid hours, on-site span in intervals, rest offsets, meal offsets)
SHIFT_SHAPES = {
    "FT8": (8.0, 34, (8, 26), (16, 17)),
    "PT6": (6.0, 26, (6, 20), (12, 13)),
    "PT4": (4.0, 16, (8,), ()),
}


def build_shift_templates(step: int = 2) -> list[ShiftTemplate]:
    """Enumerate every start time on a `step`-interval grid, per shift shape.

    Solving over templates rather than free per-interval booleans is what
    keeps shifts contiguous and breaks legal. Unconstrained interval
    variables produce mathematically optimal, operationally absurd rosters
    where an agent works 07:00-07:15 and again at 16:45.
    """
    templates: list[ShiftTemplate] = []

    for name, (paid_hours, span, rest_offsets, meal_offsets) in SHIFT_SHAPES.items():
        kind = "full_time" if name.startswith("FT") else "part_time"
        off_phone = set(rest_offsets) | set(meal_offsets)

        for start in range(0, INTERVALS_PER_DAY - span + 1, step):
            coverage = [0] * INTERVALS_PER_DAY
            breaks = []
            for offset in range(span):
                interval = start + offset
                if offset in off_phone:
                    breaks.append(interval)
                else:
                    coverage[interval] = 1

            templates.append(
                ShiftTemplate(
                    id=f"{name}_{interval_label(start).replace(':', '')}",
                    kind=kind,
                    start=start,
                    span=span,
                    paid_hours=paid_hours,
                    coverage=tuple(coverage),
                    breaks=tuple(breaks),
                )
            )

    return templates


# ---------------------------------------------------------------------------
# roster
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    """One scheduleable employee."""

    id: int
    name: str
    kind: str  # full_time | part_time
    min_weekly_hours: float
    max_weekly_hours: float
    earliest_start: int = 0  # earliest interval this agent may begin
    latest_start: int = INTERVALS_PER_DAY  # latest interval this agent may begin
    accommodation: str = ""  # human-readable reason for a start-window limit
    blackouts: dict[int, set[int]] = field(default_factory=dict)  # day -> intervals

    def is_blacked_out(self, day: int, interval: int) -> bool:
        return interval in self.blackouts.get(day, set())

    def can_work(self, day: int, template: ShiftTemplate) -> bool:
        """Availability test applied before the solver ever sees the variable."""
        if template.kind == "full_time" and self.kind == "part_time":
            return False
        if not (self.earliest_start <= template.start <= self.latest_start):
            return False
        blocked = self.blackouts.get(day)
        if blocked and any(template.occupies(i) for i in blocked):
            return False
        return True


def generate_roster(
    seed: int = DEFAULT_SEED,
    full_time: int = 45,
    part_time: int = 15,
) -> list[Agent]:
    """60 agents by default: 45 full-time, 15 part-time.

    Roughly one in ten carries a start-window accommodation, and blackout
    windows (booked leave, training, appraisals) are scattered across the week.
    """
    rng = np.random.default_rng(seed + 1)
    agents: list[Agent] = []

    for index in range(full_time + part_time):
        is_full_time = index < full_time
        kind = "full_time" if is_full_time else "part_time"

        if is_full_time:
            max_hours = float(rng.choice([37.5, 40.0]))
            min_hours = 32.0
        else:
            max_hours = float(rng.choice([20.0, 24.0]))
            min_hours = 8.0

        agent = Agent(
            id=index,
            name=f"AGT-{index + 1:03d}",
            kind=kind,
            min_weekly_hours=min_hours,
            max_weekly_hours=max_hours,
        )

        # ~10% carry a start-window accommodation.
        if rng.random() < 0.10:
            if rng.random() < 0.5:
                agent.latest_start = 8  # must start by 09:00
                agent.accommodation = "must start by 09:00"
            else:
                agent.earliest_start = 12  # cannot start before 10:00
                agent.accommodation = "cannot start before 10:00"

        # Blackout windows: booked leave or training, 0-2 blocks in the week.
        for _ in range(int(rng.integers(0, 3))):
            day = int(rng.integers(0, DAYS))
            block_start = int(rng.integers(0, INTERVALS_PER_DAY - 8))
            length = int(rng.integers(4, 13))  # 1 to 3 hours
            agent.blackouts.setdefault(day, set()).update(
                range(block_start, min(block_start + length, INTERVALS_PER_DAY))
            )

        agents.append(agent)

    return agents


# ---------------------------------------------------------------------------
# feasibility diagnostics
# ---------------------------------------------------------------------------
def capacity_report(
    agents: Iterable[Agent],
    templates: list[ShiftTemplate],
    demand: pd.DataFrame,
) -> dict:
    """Compare the week's demand against the roster's theoretical ceiling.

    Worth running before the solver. If contracted hours cannot cover demand,
    the MILP will still return a schedule - it will just be an understaffed
    one - and knowing that in advance stops you blaming the solver.
    """
    agents = list(agents)
    required_agent_intervals = int(demand["required_agents"].sum())

    # Best case: every agent works their maximum hours, all of it on-phone.
    best_coverage_ratio = max(t.covered_intervals / (t.paid_hours * 4) for t in templates)
    max_paid_intervals = sum(a.max_weekly_hours * 4 for a in agents)
    supply_ceiling = int(max_paid_intervals * best_coverage_ratio)

    contracted_minimum = sum(a.min_weekly_hours * 4 for a in agents)

    return {
        "required_agent_intervals": required_agent_intervals,
        "supply_ceiling_agent_intervals": supply_ceiling,
        "utilisation_of_ceiling": round(required_agent_intervals / supply_ceiling, 3),
        "contracted_minimum_intervals": int(contracted_minimum),
        "demand_exceeds_ceiling": required_agent_intervals > supply_ceiling,
        "minimum_hours_exceed_demand": contracted_minimum > required_agent_intervals,
    }


def roster_summary(agents: Iterable[Agent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent": a.name,
                "kind": a.kind,
                "min_weekly_hours": a.min_weekly_hours,
                "max_weekly_hours": a.max_weekly_hours,
                "accommodation": a.accommodation or "-",
                "blackout_intervals": sum(len(v) for v in a.blackouts.values()),
            }
            for a in agents
        ]
    )
