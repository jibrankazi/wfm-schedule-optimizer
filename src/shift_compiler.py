"""Compile a sector's operating window into native ``WeeklyShiftOption`` records.

The compiler owns two things the solver must not have to re-derive:

1.  ESA s 20(1) eating periods. The statute requires an eating period of at
    least 30 minutes at intervals such that the employee never works more than
    five consecutive hours without one. A 12.5h on-site line therefore carries
    two 30-minute periods, not one, and those slots are excluded from
    ``coverage_slots`` while remaining in ``occupied_slots``. Paying an agent
    for a break they are not taking is a payroll error; counting them as
    on-queue during it is a service-level error. Keeping the two tuples
    distinct is what prevents both.

2.  Night classification. A start-time rule under-classifies at both ends of
    the span range: a 12.5h platoon beginning at 19:00 runs to 07:30, and an
    8.5h swing beginning at 20:00 runs to 04:30. Both are night work and
    neither starts near midnight. Lines are therefore classified on how much
    of 00:00-05:00 they actually cover, uniformly across every duration, so a
    late swing cannot slip past the rolling fatigue cap unflagged.
"""

from __future__ import annotations

import math

from .continuous import DEFAULT_247_GRID, TimeGrid, WeeklyShiftOption
from .workforce_policy import OperationalRestPolicy, SectorConfig

# Core overnight window used for fatigue classification, and how much of it a
# line must touch before it counts as a night shift.
NIGHT_CORE_START_HOUR = 0
NIGHT_CORE_END_HOUR = 5
NIGHT_CORE_THRESHOLD_HOURS = 2.0
# A line starting at or after this hour is a night line whatever its span.
LATE_START_NIGHT_HOUR = 22.0
# On-site hours at or below this are bridge lines, worked by part-time agents.
BRIDGE_MAX_HOURS = 6.0


def _parse_hhmm(value: str, interval_minutes: int) -> int:
    hours, minutes = (int(part) for part in value.split(":"))
    return (hours * 60 + minutes) // interval_minutes


def _break_offsets(duration_hours: float, span: int, esa: OperationalRestPolicy) -> frozenset[int]:
    """Return shift-relative slot offsets held as unpaid eating periods.

    Periods are spaced so that no worked run exceeds
    ``esa.max_consecutive_work_hours_without_break``.
    """
    trigger = esa.max_consecutive_work_hours_without_break
    count = max(0, math.ceil(duration_hours / trigger) - 1)
    if count == 0:
        return frozenset()

    slots_per_break = max(1, int(round(esa.break_duration_hours * span / duration_hours)))
    offsets: set[int] = set()
    # Split the span into ``count + 1`` worked runs of equal length and drop an
    # eating period at each internal boundary.
    for index in range(1, count + 1):
        boundary = int(round(index * span / (count + 1)))
        start = min(max(0, boundary - slots_per_break // 2), span - slots_per_break)
        offsets.update(range(start, start + slots_per_break))
    return frozenset(offsets)


def _longest_worked_run(span: int, break_offsets: frozenset[int]) -> int:
    longest = run = 0
    for offset in range(span):
        if offset in break_offsets:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def _is_night(start_slot: int, span: int, grid: TimeGrid) -> bool:
    """Classify a line as night work on overlap with the overnight core.

    Start time alone is not enough at either end of the span range. A 12.5h
    platoon beginning at 19:00 runs to 07:30, and an 8.5h swing beginning at
    20:00 runs to 04:30; neither starts near midnight, and both are night work.
    Classifying on how much of 00:00-05:00 the line actually covers catches
    both, and still leaves a 13:00-21:30 swing as a day line.
    """
    intervals_per_hour = 60 // grid.interval_minutes
    if start_slot / intervals_per_hour >= LATE_START_NIGHT_HOUR:
        return True

    core_start = NIGHT_CORE_START_HOUR * intervals_per_hour
    core_end = NIGHT_CORE_END_HOUR * intervals_per_hour
    threshold = int(round(NIGHT_CORE_THRESHOLD_HOURS * intervals_per_hour))
    overlap = sum(
        1
        for offset in range(span)
        if core_start <= ((start_slot + offset) % grid.intervals_per_day) < core_end
    )
    return overlap >= threshold


def compile_shift_library(
    sector_cfg: SectorConfig,
    esa_policy: OperationalRestPolicy | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> list[WeeklyShiftOption]:
    """Mint every legal shift line for ``sector_cfg`` on ``grid``.

    Raises ``ValueError`` if the sector asks for a span past the ESA s 17(1)(a)
    daily cap without recording the s 17(2) written agreement.
    """
    esa = esa_policy or OperationalRestPolicy()
    intervals_per_hour = 60 // grid.interval_minutes

    for duration_hours in sector_cfg.allowed_shift_durations:
        if duration_hours <= 0:
            raise ValueError("shift durations must be positive")
        if (
            duration_hours > esa.max_daily_work_hours
            and not sector_cfg.has_excess_daily_hours_agreement
        ):
            raise ValueError(
                f"{sector_cfg.sector.value}: a {duration_hours}h line exceeds the ESA "
                f"s 17(1)(a) {esa.max_daily_work_hours}h daily cap; set "
                "has_excess_daily_hours_agreement once the s 17(2) agreement is on file"
            )

    options: list[WeeklyShiftOption] = []
    index = 0

    for day in range(grid.days):
        window = sector_cfg.opens_at(day)
        if window is None:
            open_slot, close_slot = 0, grid.intervals_per_day
        else:
            open_slot = _parse_hhmm(window[0], grid.interval_minutes)
            close_slot = _parse_hhmm(window[1], grid.interval_minutes)

        step = 2 if sector_cfg.allow_part_time_bridges else 4
        for start_slot in range(open_slot, close_slot, step):
            for duration_hours in sector_cfg.allowed_shift_durations:
                span = int(round(duration_hours * intervals_per_hour))
                if window is not None and start_slot + span > close_slot:
                    continue

                break_offsets = _break_offsets(duration_hours, span, esa)
                worked_run_hours = _longest_worked_run(span, break_offsets) / intervals_per_hour
                if worked_run_hours > esa.max_consecutive_work_hours_without_break + 1e-9:
                    raise ValueError(
                        f"{duration_hours}h line leaves a {worked_run_hours}h worked run, "
                        "which breaches ESA s 20(1)"
                    )

                paid_hours = duration_hours - len(break_offsets) / intervals_per_hour
                start_abs = day * grid.intervals_per_day + start_slot

                coverage: list[int] = []
                occupied: list[int] = []
                breaks: list[int] = []
                for offset in range(span):
                    slot = (start_abs + offset) % grid.total_intervals
                    occupied.append(slot)
                    if offset in break_offsets:
                        breaks.append(slot)
                    else:
                        coverage.append(slot)

                is_bridge = duration_hours <= BRIDGE_MAX_HOURS
                options.append(
                    WeeklyShiftOption(
                        id=f"{sector_cfg.sector.value}_d{day}_s{start_slot}_n{span}_{index}",
                        shape=f"{duration_hours:g}h",
                        kind="part_time" if is_bridge else "full_time",
                        start_day=day,
                        start_slot=start_slot,
                        start_abs=start_abs,
                        span=span,
                        paid_hours=paid_hours,
                        coverage_slots=tuple(coverage),
                        occupied_slots=tuple(occupied),
                        break_slots=tuple(breaks),
                        is_night=_is_night(start_slot, span, grid),
                    )
                )
                index += 1

    if not options:
        raise ValueError(
            f"{sector_cfg.sector.value}: no shift fits the configured operating window"
        )
    return options
