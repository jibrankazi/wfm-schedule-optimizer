"""Operating patterns: the same engine, three different contact centres.

An operating pattern is data, not code. A centre is described by how many days
it runs, the window it is open, whether it holds an overnight presence floor,
and how many people it employs - and the same solver reads all of it. That is
the claim this module exists to make checkable: nothing here is hardcoded to
one room.

Two things are deliberately *not* expressed as a smaller grid. The planning
grid is always a full 24 hours, because a shift that ends at 19:00 and one
that ends at midnight have to be comparable. Opening hours instead appear in
two places: demand is zero outside them, and shift options that would run past
closing are removed from the action space. That second part is what actually
shrinks the model - a five-day daytime centre carries roughly a third of the
binaries of a continuous one, not because its grid is smaller but because far
fewer shifts are legal.

Roster sizes are the arithmetic minimum plus real slack. The minimum comes
from dividing required on-phone intervals by what one full-time line delivers,
after the measured ~20% loss between theoretical and schedulable capacity.
Every operation then carries more than that for absence, training and
vacancy. Sizing an instance below its arithmetic floor and then reducing
demand until it fits is how the earlier 65-agent 24/7 instance came about, and
it is what this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.continuous import (
    CONTINUOUS_247_PROFILE,
    TimeGrid,
    WeeklyAgent,
    WeeklyShiftOption,
    build_weekly_shift_options,
)
from src.erlang import ServiceTarget, required_agents, traffic_intensity

DEFAULT_AHT_SEC = 320.0
DAY_FACTOR = np.array([1.10, 1.02, 0.98, 0.96, 1.00, 0.78, 0.70])


@dataclass(frozen=True)
class OperatingPattern:
    """One contact centre's shape, as data."""

    key: str
    label: str
    days: int
    open_from: int  # first open interval of the day, 0-95
    open_to: int  # first closed interval, exclusive
    presence_floor: int
    peak_calls: float
    full_time: int
    part_time: int
    weekend_factor: float = 1.0
    overnight_fraction: float = 0.0
    intervals_per_day: int = 96
    start_stride: int = 1  # keep every nth legal start; >1 coarsens the action space

    def __post_init__(self) -> None:
        if not 0 <= self.open_from < self.open_to <= self.intervals_per_day:
            raise ValueError(f"{self.key}: opening window must satisfy 0 <= from < to <= day")
        if self.days < 1 or self.days > 7:
            raise ValueError(f"{self.key}: days must be 1-7")
        if self.presence_floor < 0:
            raise ValueError(f"{self.key}: presence floor cannot be negative")
        if self.full_time + self.part_time <= 0:
            raise ValueError(f"{self.key}: roster cannot be empty")

    @property
    def grid(self) -> TimeGrid:
        return TimeGrid(days=self.days, intervals_per_day=self.intervals_per_day)

    @property
    def roster_size(self) -> int:
        return self.full_time + self.part_time

    @property
    def is_continuous(self) -> bool:
        return self.open_from == 0 and self.open_to == self.intervals_per_day

    @property
    def open_intervals(self) -> int:
        """Intervals the centre actually trades in, across the week."""
        return (self.open_to - self.open_from) * self.days

    def describe(self) -> dict[str, object]:
        return {
            "pattern": self.key,
            "label": self.label,
            "days": self.days,
            "window": f"{_clock(self.open_from)}-{_clock(self.open_to)}",
            "grid_intervals": self.grid.total_intervals,
            "open_intervals": self.open_intervals,
            "presence_floor": self.presence_floor,
            "roster": self.roster_size,
        }


def _clock(index: int) -> str:
    minutes = (index % 96) * 15 if index < 96 else 0
    return "24:00" if index >= 96 else f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# the three patterns
# ---------------------------------------------------------------------------
WEEKDAY_EXTENDED = OperatingPattern(
    key="weekday_extended",
    label="Weekday extended hours, 07:00-19:00",
    days=5,
    open_from=28,  # 07:00
    open_to=76,  # 19:00
    presence_floor=0,
    peak_calls=110.0,
    full_time=48,
    part_time=17,
)

CONTINUOUS_247 = OperatingPattern(
    key="continuous_247",
    label="Continuous, 24 hours, seven days",
    days=7,
    open_from=0,
    open_to=96,
    presence_floor=4,
    peak_calls=110.0,
    full_time=113,
    part_time=37,
    weekend_factor=1.0,
    overnight_fraction=0.03,
    start_stride=4,
)

SEVEN_DAY_EXTENDED = OperatingPattern(
    key="seven_day_extended",
    label="Seven-day extended hours, 07:00-22:00",
    days=7,
    open_from=28,  # 07:00
    open_to=88,  # 22:00
    presence_floor=0,
    peak_calls=110.0,
    full_time=73,
    part_time=27,
    weekend_factor=0.5,
)

PATTERNS: dict[str, OperatingPattern] = {
    p.key: p for p in (WEEKDAY_EXTENDED, CONTINUOUS_247, SEVEN_DAY_EXTENDED)
}


# ---------------------------------------------------------------------------
# demand and action space
# ---------------------------------------------------------------------------
def build_pattern_demand(
    pattern: OperatingPattern, seed: int = 42, aht_sec: float = DEFAULT_AHT_SEC
) -> pd.DataFrame:
    """Demand for one pattern: bimodal inside the window, zero outside it."""
    target = ServiceTarget()
    rng = np.random.default_rng(seed)
    day_len = pattern.intervals_per_day

    position = np.linspace(0.0, 1.0, day_len)
    shape = np.exp(-(((position - 0.44) / 0.11) ** 2)) + 0.80 * np.exp(
        -(((position - 0.63) / 0.13) ** 2)
    )
    shape = shape / shape.max()

    open_mask = np.zeros(day_len, dtype=bool)
    open_mask[pattern.open_from : pattern.open_to] = True

    rows = []
    for day in range(pattern.days):
        weekend = day >= 5
        factor = DAY_FACTOR[day] * (pattern.weekend_factor if weekend else 1.0)
        base = pattern.peak_calls * factor * shape
        if pattern.is_continuous:
            volume = np.clip(base, pattern.peak_calls * pattern.overnight_fraction, None)
        else:
            volume = np.where(open_mask, base, 0.0)
        volume = np.clip(np.round(volume + rng.normal(0.0, 1.4, day_len)), 0, None)
        if not pattern.is_continuous:
            # Noise is applied after the window is imposed, so it has to be
            # cleared again: a closed centre receiving four calls at 02:00
            # pulls in two agents to answer them.
            volume = np.where(open_mask, volume, 0.0)

        for interval in range(day_len):
            calls = float(volume[interval])
            queueing = required_agents(calls, aht_sec, target)
            # The floor only applies where the centre is actually open.
            floor = pattern.presence_floor if (open_mask[interval] or pattern.is_continuous) else 0
            need = max(queueing, floor)
            rows.append(
                {
                    "day": day,
                    "interval": interval,
                    "week_interval": day * day_len + interval,
                    "is_open": bool(open_mask[interval]) or pattern.is_continuous,
                    "calls": calls,
                    "aht_sec": aht_sec,
                    "erlangs": round(traffic_intensity(calls, aht_sec), 2),
                    "queueing_agents": queueing,
                    "required_agents": need,
                    "min_presence_floor": min(floor, need),
                    "binding_constraint": "floor" if floor > queueing else "queueing",
                }
            )
    return pd.DataFrame(rows)


def build_pattern_options(pattern: OperatingPattern) -> list[WeeklyShiftOption]:
    """Legal shifts for a pattern.

    For a centre that closes, a shift running past closing time is not a
    scheduling choice anyone would make, so it is removed rather than left for
    the solver to price out. This is where the model actually shrinks: the
    grid stays 24 hours, but the action space collapses to the trading window.
    """
    options = build_weekly_shift_options(pattern.grid)

    if pattern.start_stride > 1:
        # A continuous desk runs a small number of rotating lines, not a shift
        # starting every fifteen minutes. Keeping every nth legal start is
        # both closer to how such an operation is actually run and the
        # difference between a model that builds and one that does not: at a
        # 150-agent roster the full start grid produces ~126,000 binaries and
        # exhausts memory during construction, before the solver is reached.
        day_len = pattern.intervals_per_day
        keep_starts = sorted(
            {s for i, s in enumerate(sorted({o.start_slot % day_len for o in options}))
             if i % pattern.start_stride == 0}
        )
        options = [o for o in options if (o.start_slot % day_len) in keep_starts]

    if pattern.is_continuous:
        return options

    day_len = pattern.intervals_per_day
    keep = []
    for option in options:
        slots = {slot % day_len for slot in option.coverage_slots}
        if all(pattern.open_from <= slot < pattern.open_to for slot in slots):
            keep.append(option)
    return keep


def pattern_roster(pattern: OperatingPattern, seed: int = 42) -> list[WeeklyAgent]:
    """The roster a pattern employs, sized in the pattern rather than here."""
    from src.continuous import generate_continuous_roster

    return generate_continuous_roster(
        full_time=pattern.full_time, part_time=pattern.part_time
    )


def sizing_report(pattern: OperatingPattern, seed: int = 42) -> dict[str, object]:
    """What the pattern needs against what its roster can deliver."""
    demand = build_pattern_demand(pattern, seed=seed)
    options = build_pattern_options(pattern)
    agents = pattern_roster(pattern, seed=seed)

    required = int(demand.required_agents.sum())
    on_phone_ratio = max(
        len(o.coverage_slots) / (o.paid_hours * 4) for o in options
    ) if options else 0.0
    ceiling = sum(a.max_weekly_hours for a in agents) * 4 * on_phone_ratio

    return {
        **pattern.describe(),
        "shift_options": len(options),
        "required_agent_intervals": required,
        "on_phone_ceiling": round(ceiling),
        "theoretical_utilisation": round(required / ceiling, 3) if ceiling else None,
        "binary_variables_estimate": len(agents) * pattern.days * (len(options) // pattern.days),
        "profile": "Erlang C 80/20, 85% occupancy",
    }
