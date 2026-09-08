"""Continuous 24/7 planning grid and synthetic weekly inputs.

The original demonstrator intentionally models five independent business
days.  A 24/7 roster cannot be represented by simply changing 49 intervals
to 96: a shift starting Sunday at 22:00 covers Monday after midnight.  This
module therefore uses one cyclic absolute index across the whole week.

The shipped workload remains synthetic.  The default is 65 agents, a
70-contact peak, 15-minute intervals, hourly starts overnight, and half-hourly
starts during the day.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd

from .profile_analysis import required_agents_for_profile
from .profiles import COMMERCIAL_INBOUND, ServiceProfile


WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class TimeGrid:
    """A fixed-resolution cyclic planning horizon."""

    days: int = 7
    intervals_per_day: int = 96
    interval_minutes: int = 15

    def __post_init__(self) -> None:
        if self.days <= 0:
            raise ValueError("days must be positive")
        if self.days > len(WEEKDAY_NAMES):
            raise ValueError("the weekly grid supports at most seven days")
        if self.intervals_per_day <= 0:
            raise ValueError("intervals_per_day must be positive")
        if self.interval_minutes <= 0 or 1_440 % self.interval_minutes:
            raise ValueError("interval_minutes must divide a 24-hour day")
        if self.intervals_per_day * self.interval_minutes != 1_440:
            raise ValueError("intervals_per_day and interval_minutes must span 24 hours")

    @property
    def total_intervals(self) -> int:
        return self.days * self.intervals_per_day

    @property
    def intervals_per_hour(self) -> int:
        return 60 // self.interval_minutes

    def time_label(self, slot_of_day: int) -> str:
        if not 0 <= slot_of_day < self.intervals_per_day:
            raise ValueError("slot_of_day is outside the grid")
        minutes = slot_of_day * self.interval_minutes
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def absolute_slot(self, day: int, slot_of_day: int) -> int:
        if not 0 <= day < self.days:
            raise ValueError("day is outside the grid")
        if not 0 <= slot_of_day < self.intervals_per_day:
            raise ValueError("slot_of_day is outside the grid")
        return day * self.intervals_per_day + slot_of_day


DEFAULT_247_GRID = TimeGrid()

# The 65-agent sizing case uses the commercial 80/20 contract with a separate
# four-person 24/7 presence invariant.  Applying the rapid-response 95/10,
# 70%-occupancy contract to the same peak creates more required work than this
# roster can physically cover, which would turn the example into a capacity
# shortage rather than a scheduling benchmark.
CONTINUOUS_247_PROFILE = replace(
    COMMERCIAL_INBOUND,
    name="24/7 standard intake",
    min_presence_floor=4,
    hours="24/7",
)


@dataclass(frozen=True)
class WeeklyAgent:
    """An employee available to the continuous-week solver."""

    id: int
    name: str
    kind: str  # full_time | part_time
    min_weekly_hours: float
    max_weekly_hours: float
    unavailable_slots: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.kind not in {"full_time", "part_time"}:
            raise ValueError("kind must be full_time or part_time")
        if self.min_weekly_hours < 0:
            raise ValueError("min_weekly_hours must be non-negative")
        if self.max_weekly_hours < self.min_weekly_hours:
            raise ValueError("max_weekly_hours must not be below the minimum")


@dataclass(frozen=True)
class WeeklyShiftOption:
    """One concrete shift start, including cross-midnight weekly coverage."""

    id: str
    shape: str
    kind: str
    start_day: int
    start_slot: int
    start_abs: int
    span: int
    paid_hours: float
    coverage_slots: tuple[int, ...]
    occupied_slots: tuple[int, ...]
    break_slots: tuple[int, ...]
    is_night: bool

    @property
    def end_abs(self) -> int:
        """Unwrapped end, used when calculating rest into the next day."""
        return self.start_abs + self.span


# paid hours, on-site span, paid/unpaid off-phone offsets
WEEKLY_SHIFT_SHAPES: Mapping[str, tuple[float, int, tuple[int, ...]]] = {
    "FT8": (8.0, 34, (8, 16, 17, 26)),
    "PT6": (6.0, 26, (6, 12, 13, 20)),
    "PT4": (4.0, 16, (8,)),
}


def allowed_start_slots(grid: TimeGrid = DEFAULT_247_GRID) -> tuple[int, ...]:
    """Hourly starts from 22:00-06:00; half-hourly from 06:00-22:00."""
    if 30 % grid.interval_minutes:
        raise ValueError("half-hourly starts require a resolution dividing 30 minutes")
    slots: list[int] = []
    for hour in range(24):
        first = hour * grid.intervals_per_hour
        overnight = hour < 6 or hour >= 22
        slots.append(first)
        if not overnight:
            slots.append(first + 30 // grid.interval_minutes)
    return tuple(slots)


def build_weekly_shift_options(
    grid: TimeGrid = DEFAULT_247_GRID,
) -> list[WeeklyShiftOption]:
    """Enumerate concrete starts on a cyclic week.

    Coverage wraps at the week boundary.  This represents a recurring weekly
    roster, so a Sunday 22:00 shift correctly covers early Monday slots.
    """
    options: list[WeeklyShiftOption] = []
    night_start = 22 * grid.intervals_per_hour
    day_start = 6 * grid.intervals_per_hour

    for day in range(grid.days):
        for start_slot in allowed_start_slots(grid):
            for shape, (paid_hours, span, off_phone) in WEEKLY_SHIFT_SHAPES.items():
                kind = "full_time" if shape.startswith("FT") else "part_time"
                raw_start = grid.absolute_slot(day, start_slot)
                occupied = tuple(
                    (raw_start + offset) % grid.total_intervals
                    for offset in range(span)
                )
                off_phone_set = set(off_phone)
                coverage = tuple(
                    (raw_start + offset) % grid.total_intervals
                    for offset in range(span)
                    if offset not in off_phone_set
                )
                breaks = tuple(
                    (raw_start + offset) % grid.total_intervals
                    for offset in off_phone
                )
                is_night = start_slot < day_start or start_slot >= night_start
                options.append(
                    WeeklyShiftOption(
                        id=(
                            f"{WEEKDAY_NAMES[day]}_{shape}_"
                            f"{grid.time_label(start_slot).replace(':', '')}"
                        ),
                        shape=shape,
                        kind=kind,
                        start_day=day,
                        start_slot=start_slot,
                        start_abs=raw_start,
                        span=span,
                        paid_hours=paid_hours,
                        coverage_slots=coverage,
                        occupied_slots=occupied,
                        break_slots=breaks,
                        is_night=is_night,
                    )
                )
    return options


def generate_continuous_roster(
    full_time: int = 49,
    part_time: int = 16,
) -> list[WeeklyAgent]:
    """Return the deterministic 65-agent default roster."""
    if full_time < 0 or part_time < 0 or full_time + part_time <= 0:
        raise ValueError("roster counts must be non-negative and not both zero")

    agents: list[WeeklyAgent] = []
    for index in range(full_time + part_time):
        is_full_time = index < full_time
        agents.append(
            WeeklyAgent(
                id=index,
                name=f"AGT-{index + 1:03d}",
                kind="full_time" if is_full_time else "part_time",
                min_weekly_hours=32.0 if is_full_time else 8.0,
                max_weekly_hours=40.0 if is_full_time else 24.0,
            )
        )
    return agents


def generate_continuous_demand(
    seed: int = 42,
    peak_calls: float = 70.0,
    profile: ServiceProfile = CONTINUOUS_247_PROFILE,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> pd.DataFrame:
    """Generate 672 synthetic demand rows and profile-sized requirements."""
    if not np.isfinite(peak_calls) or peak_calls <= 0:
        raise ValueError("peak_calls must be positive and finite")
    rng = np.random.default_rng(seed)
    quarter_hours = np.arange(grid.intervals_per_day, dtype=float)
    hours = quarter_hours * grid.interval_minutes / 60.0

    morning = np.exp(-0.5 * ((hours - 10.5) / 2.1) ** 2)
    afternoon = 0.82 * np.exp(-0.5 * ((hours - 15.5) / 2.6) ** 2)
    evening = 0.20 * np.exp(-0.5 * ((hours - 20.0) / 1.8) ** 2)
    shape = 0.018 + morning + afternoon + evening
    shape /= shape.max()
    day_factors = np.array([1.00, 0.96, 0.94, 0.92, 0.90, 0.70, 0.64])

    rows: list[dict[str, object]] = []
    for day in range(grid.days):
        calls = peak_calls * day_factors[day] * shape
        calls += rng.normal(0.0, 1.2, grid.intervals_per_day)
        calls = np.clip(np.round(calls), 0.0, peak_calls)
        aht = np.clip(
            rng.normal(320.0, 22.0, grid.intervals_per_day), 180.0, 600.0
        )

        for interval in range(grid.intervals_per_day):
            contacts = float(calls[interval])
            handle_time = float(aht[interval])
            queueing_agents = required_agents_for_profile(
                contacts, handle_time, profile
            )
            required = max(queueing_agents, profile.min_presence_floor)
            rows.append(
                {
                    "day": day,
                    "day_name": WEEKDAY_NAMES[day],
                    "interval": interval,
                    "week_interval": grid.absolute_slot(day, interval),
                    "time": grid.time_label(interval),
                    "calls": contacts,
                    "aht_sec": round(handle_time, 1),
                    "required_agents": required,
                    "min_presence_floor": profile.min_presence_floor,
                    "binding_constraint": (
                        "floor" if required > queueing_agents else "queueing"
                    ),
                    "profile": profile.name,
                }
            )

    return pd.DataFrame(rows)


def eligible_for_option(agent: WeeklyAgent, option: WeeklyShiftOption) -> bool:
    """Keep the enterprise model compact: FT uses FT8; PT uses PT4/PT6."""
    if agent.kind != option.kind:
        return False
    return not agent.unavailable_slots.intersection(option.occupied_slots)


def rest_intervals(
    earlier: WeeklyShiftOption,
    later: WeeklyShiftOption,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> int:
    """Return rest from ``earlier`` end to ``later`` start on a cyclic week."""
    later_start = later.start_abs
    while later_start <= earlier.start_abs:
        later_start += grid.total_intervals
    return later_start - earlier.end_abs
