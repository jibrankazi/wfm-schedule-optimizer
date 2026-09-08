"""Retained configuration for the standalone sector template compiler.

The active operating-pattern solver uses WeeklySolverConfig instead. These
legacy policy fields are not statutory constraints of that solver.
"""

from __future__ import annotations

import numpy as np

from .continuous import TimeGrid

from dataclasses import dataclass
from enum import Enum


class OperatingRegime(Enum):
    """Coarse label for the operating window described by ``daily_open_close``."""

    BUSINESS_HOURS = "business_hours"
    EXTENDED_WEEKDAY = "extended_weekday"
    EXTENDED_SEVEN_DAY = "extended_seven_day"
    CONTINUOUS_247 = "continuous_247"


class SectorProfile(Enum):
    PROFILE_A = "profile_a"
    PROFILE_B = "profile_b"
    PROFILE_C = "profile_c"
    GENERAL_COMMERCIAL = "general_commercial"


@dataclass(frozen=True)
class OperationalRestPolicy:
    """Legacy policy values retained for compatibility.

    Only the break trigger, break duration and daily span gate are consumed by
    shift_compiler. The rest, weekly-hours and overtime fields are metadata
    since the universal optimizer was removed. No compliance is certified.
    """

    min_daily_rest_hours: float = 11.0
    min_inter_shift_rest_hours: float = 8.0
    min_weekly_rest_hours: float = 24.0
    max_daily_work_hours: float = 8.0
    max_weekly_work_hours: float = 48.0
    overtime_threshold_weekly_hours: float = 44.0
    max_consecutive_work_hours_without_break: float = 5.0
    break_duration_hours: float = 0.5

    def __post_init__(self) -> None:
        if self.max_consecutive_work_hours_without_break <= 0:
            raise ValueError("eating-period trigger must be positive")
        if self.break_duration_hours < 0:
            raise ValueError("eating-period duration must be non-negative")
        if self.min_inter_shift_rest_hours > self.min_daily_rest_hours:
            raise ValueError("inter-shift rest cannot exceed the daily rest floor")


@dataclass(frozen=True)
class SectorConfig:
    """Operational profile for one service line."""

    sector: SectorProfile
    regime: OperatingRegime
    # 0 = Monday .. 6 = Sunday. ``None`` means the day is staffed around the clock.
    daily_open_close: dict[int, tuple[str, str] | None]
    allowed_shift_durations: list[float]  # on-site hours, breaks included
    life_safety_floor: int
    is_floor_hard_invariant: bool
    max_consecutive_nights: int
    target_sla_pct: float
    target_answer_sec: float
    allow_part_time_bridges: bool
    # Written agreement under ESA s 17(2) permitting shifts past the s 17(1)(a)
    # eight-hour daily cap. Required for any 12.5h platoon line.
    has_excess_daily_hours_agreement: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_shift_durations:
            raise ValueError("a sector needs at least one allowed shift duration")
        if self.life_safety_floor < 0:
            raise ValueError("life_safety_floor must be non-negative")
        if self.max_consecutive_nights < 0:
            raise ValueError("max_consecutive_nights must be non-negative")

    def opens_at(self, day: int) -> tuple[str, str] | None:
        return self.daily_open_close.get(day)


def get_profile_a_config() -> SectorConfig:
    """weekday_extended: extended seven-day intake with midday part-time bridges."""
    return SectorConfig(
        sector=SectorProfile.PROFILE_A,
        regime=OperatingRegime.EXTENDED_SEVEN_DAY,
        daily_open_close={day: ("07:00", "22:00") for day in range(7)},
        allowed_shift_durations=[4.0, 6.0, 8.5],
        life_safety_floor=2,
        is_floor_hard_invariant=False,
        max_consecutive_nights=3,
        target_sla_pct=80.0,
        target_answer_sec=20.0,
        allow_part_time_bridges=True,
        has_excess_daily_hours_agreement=True,  # 8.5h on-site exceeds the 8h cap
    )


def get_profile_b_config() -> SectorConfig:
    """Illustrative continuous profile with a configured 90/10 target."""
    return SectorConfig(
        sector=SectorProfile.PROFILE_B,
        regime=OperatingRegime.CONTINUOUS_247,
        daily_open_close={day: None for day in range(7)},
        allowed_shift_durations=[8.5, 12.5],
        life_safety_floor=4,
        is_floor_hard_invariant=True,
        max_consecutive_nights=2,
        target_sla_pct=90.0,
        target_answer_sec=10.0,
        allow_part_time_bridges=False,
        has_excess_daily_hours_agreement=True,
    )


def get_profile_c_config() -> SectorConfig:
    """Secondary PSAP: continuous 24/7 twelve-hour platoon dispatch."""
    return SectorConfig(
        sector=SectorProfile.PROFILE_C,
        regime=OperatingRegime.CONTINUOUS_247,
        daily_open_close={day: None for day in range(7)},
        allowed_shift_durations=[12.5],
        life_safety_floor=3,
        is_floor_hard_invariant=True,
        max_consecutive_nights=2,
        target_sla_pct=95.0,
        target_answer_sec=15.0,
        allow_part_time_bridges=False,
        has_excess_daily_hours_agreement=True,
    )


# Relocated from the removed universal solver. It describes when a service line
# is open, which is a policy property rather than a solver one, so it belongs
# beside SectorConfig.
def _parse_hhmm(value: str, interval_minutes: int) -> int:
    hours, minutes = (int(part) for part in value.split(":"))
    return (hours * 60 + minutes) // interval_minutes


def operating_mask(sector_cfg: SectorConfig, grid: TimeGrid) -> np.ndarray:
    """Boolean mask over the week, ``True`` where the service line is open."""
    mask = np.zeros(grid.total_intervals, dtype=bool)
    for day in range(grid.days):
        window = sector_cfg.opens_at(day)
        base = day * grid.intervals_per_day
        if window is None:
            mask[base : base + grid.intervals_per_day] = True
            continue
        open_slot = _parse_hhmm(window[0], grid.interval_minutes)
        close_slot = _parse_hhmm(window[1], grid.interval_minutes)
        mask[base + open_slot : base + close_slot] = True
    return mask
