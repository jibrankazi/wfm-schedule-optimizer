"""Readable views of a solved schedule.

A CSV of start and end times is the right thing to store and the wrong thing to
hand a supervisor. What a supervisor reads is a grid: people down the side,
the day across the top, and colour telling them at a glance who is on the
queue, who is on a break and where the floor thins out.

Two renderings live here. `render_day_grid` draws that grid as an image with a
coverage strip underneath, so a deficit is visible in the same picture as its
cause. `write_schedule_workbook` produces the same information as a real
spreadsheet with frozen panes and a legend, because a schedule that cannot be
filtered, printed or emailed does not get used.

Both derive entirely from the solved schedule object, which carries each
shift's exact break positions - the CSV export only records how many breaks a
shift has, not where they fall, which is why these read the schedule directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from src.optimizer_247 import WeeklySchedule

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# state codes used in both the image and the workbook
OFF, ON_QUEUE, BREAK, MEAL = 0, 1, 2, 3

STATE_COLOURS = {
    OFF: "#F2F3F4",
    ON_QUEUE: "#2E86C1",
    BREAK: "#F5B041",
    MEAL: "#E67E22",
}
STATE_LABELS = {
    OFF: "Off duty",
    ON_QUEUE: "On queue",
    BREAK: "Paid break",
    MEAL: "Unpaid meal",
}
# An excel fill wants RGB without the hash.
STATE_FILLS = {code: colour.lstrip("#") for code, colour in STATE_COLOURS.items()}


@dataclass(frozen=True)
class ShiftRow:
    """One agent's line on the grid, with the window they actually work."""

    agent: str
    start: str
    end: str
    paid_hours: float

    @property
    def label(self) -> str:
        return f"{self.agent}  {self.start}-{self.end}"


def interval_label(index: int, per_day: int = 96) -> str:
    minutes = (index % per_day) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def day_state_matrix(
    schedule: "WeeklySchedule", day: int, per_day: int = 96
) -> tuple[list[ShiftRow], np.ndarray]:
    """One row per shift touching this day, with values in {OFF..MEAL}.

    A shift's break slots are split into the shorter paid reliefs and the
    longer unpaid meal by run length, so the two read differently in the
    picture - a supervisor cares about the difference.
    """
    rows: list[tuple[str, str, str, float, np.ndarray]] = []
    horizon = len(schedule.coverage)
    day_start, day_end = day * per_day, (day + 1) * per_day
    for assignment in schedule.assignments:
        option = assignment.option
        occupied_today = [s for s in option.occupied_slots if day_start <= s < day_end]
        if not occupied_today:
            continue
        # A Sunday start wraps into Monday. Use the occurrence that actually
        # intersects this day's absolute interval range.
        shift_start = option.start_abs
        if shift_start >= day_end:
            shift_start -= horizon
        shift_end = shift_start + option.span
        start = interval_label(option.start_slot, per_day)
        if shift_start < day_start:
            start += " (-1d)"
        end = "24:00" if shift_end == day_end else interval_label(shift_end, per_day)
        if shift_end > day_end:
            end += " (+1d)"
        row = np.full(per_day, OFF, dtype=int)
        for slot in option.coverage_slots:
            if day_start <= slot < day_end:
                row[slot - day_start] = ON_QUEUE

        # Classify entire break runs before clipping to a calendar day, so a
        # meal straddling midnight does not become two paid relief breaks.
        offsets = sorted((slot - option.start_abs) % horizon for slot in option.break_slots)
        meal_slots_today = 0
        for run in _consecutive_runs(offsets):
            state = MEAL if len(run) > 1 else BREAK
            for offset in run:
                slot = (option.start_abs + offset) % horizon
                if day_start <= slot < day_end:
                    row[slot - day_start] = state
                    meal_slots_today += int(state == MEAL)
        paid_hours_today = (len(occupied_today) - meal_slots_today) * 0.25
        rows.append((assignment.agent_name, start, end, paid_hours_today, row))

    rows.sort(key=lambda r: (int(np.flatnonzero(r[4] != OFF)[0]), r[0]))
    labels = [ShiftRow(name, start, end, hours) for name, start, end, hours, _ in rows]
    matrix = np.array([r[4] for r in rows]) if rows else np.zeros((0, per_day), int)
    return labels, matrix


def _consecutive_runs(values: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if runs and value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def render_day_grid(
    schedule: "WeeklySchedule",
    day: int = 0,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    per_day: int = 96,
) -> Figure:
    """The grid a supervisor reads, with coverage against requirement below it."""
    rows, matrix = day_state_matrix(schedule, day, per_day)
    if matrix.size == 0:
        raise ValueError(f"no assignments on {DAY_NAMES[day]}")
    names = [r.label for r in rows]

    required = np.asarray(schedule.required)[day * per_day : (day + 1) * per_day]
    scheduled = np.asarray(schedule.coverage)[day * per_day : (day + 1) * per_day]

    height = max(6.0, 0.16 * len(names) + 3.2)
    figure, (grid_ax, cover_ax) = plt.subplots(
        2, 1, figsize=(16, height), height_ratios=[len(names) * 0.11 + 1, 2.2], sharex=True
    )

    cmap = ListedColormap([STATE_COLOURS[c] for c in (OFF, ON_QUEUE, BREAK, MEAL)])
    grid_ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    grid_ax.set_yticks(range(len(names)))
    grid_ax.set_yticklabels(names, fontsize=max(3, min(7, 420 // max(len(names), 1))))
    grid_ax.set_ylabel(f"{len(names)} agents on duty", fontweight="bold")
    grid_ax.set_title(
        title or f"Shift plan - {DAY_NAMES[day]}",
        fontsize=13, fontweight="bold", pad=12,
    )
    grid_ax.legend(
        handles=[mpatches.Patch(color=STATE_COLOURS[c], label=STATE_LABELS[c])
                 for c in (ON_QUEUE, BREAK, MEAL, OFF)],
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False, fontsize=9,
    )
    for hour in range(0, per_day, 4):
        grid_ax.axvline(hour - 0.5, color="white", linewidth=0.4, alpha=0.5)

    positions = np.arange(per_day)
    cover_ax.bar(positions, scheduled, width=0.9, color=STATE_COLOURS[ON_QUEUE],
                 alpha=0.75, label="Scheduled on queue")
    shortfall = np.clip(required - scheduled, 0, None)
    if shortfall.any():
        cover_ax.bar(positions, shortfall, bottom=scheduled, width=0.9,
                     color="#E74C3C", alpha=0.85, label="Short of requirement")
    cover_ax.step(positions, required, where="mid", color="#922B21", linewidth=2.0,
                  label="Required")
    cover_ax.set_ylabel("Agents", fontweight="bold")
    cover_ax.set_xticks(range(0, per_day, 4))
    cover_ax.set_xticklabels([interval_label(i, per_day) for i in range(0, per_day, 4)],
                             rotation=45, ha="right", fontsize=8)
    cover_ax.set_xlim(-0.5, per_day - 0.5)
    cover_ax.set_xlabel("Time of day", fontweight="bold")
    cover_ax.grid(axis="y", linestyle="--", alpha=0.35)
    cover_ax.set_axisbelow(True)
    cover_ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)

    figure.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure


def write_schedule_workbook(
    schedule: "WeeklySchedule",
    path: str | Path,
    days: int,
    pattern_label: str = "",
    per_day: int = 96,
) -> Path:
    """The same grid as a spreadsheet: one sheet per day, plus a summary.

    Frozen panes and a legend, because the first thing anyone does with a
    schedule is scroll sideways looking for one person's row.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    book = Workbook()
    book.remove(book.active)

    header_font = Font(bold=True, size=9)
    thin = Side(style="thin", color="D5D8DC")
    fills = {code: PatternFill("solid", fgColor=rgb) for code, rgb in STATE_FILLS.items()}

    for day in range(days):
        rows, matrix = day_state_matrix(schedule, day, per_day)
        sheet = book.create_sheet(DAY_NAMES[day])
        for column, heading in enumerate(("Agent", "Start", "End", "Paid h"), start=1):
            sheet.cell(row=1, column=column, value=heading).font = header_font
        offset = 5  # the grid begins after the four detail columns
        for interval in range(per_day):
            cell = sheet.cell(row=1, column=interval + offset,
                              value=interval_label(interval, per_day))
            cell.font = Font(bold=True, size=7)
            cell.alignment = Alignment(textRotation=90, horizontal="center")
            sheet.column_dimensions[get_column_letter(interval + offset)].width = 2.6

        for row_index, (shift, row) in enumerate(zip(rows, matrix), start=2):
            details: tuple[object, ...] = (
                shift.agent, shift.start, shift.end, shift.paid_hours,
            )
            for column, value in enumerate(details, start=1):
                sheet.cell(row=row_index, column=column, value=value).font = Font(size=8)
            for interval, state in enumerate(row):
                cell = sheet.cell(row=row_index, column=interval + offset)
                cell.fill = fills[int(state)]
                cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)

        sheet.column_dimensions["A"].width = 12
        for detail_column in ("B", "C", "D"):
            sheet.column_dimensions[detail_column].width = 7
        sheet.freeze_panes = "E2"

    legend = book.create_sheet("Legend", 0)
    legend["A1"] = f"Schedule{f' - {pattern_label}' if pattern_label else ''}"
    legend["A1"].font = Font(bold=True, size=13)
    legend["A3"] = "Colour"
    legend["B3"] = "Meaning"
    legend["A3"].font = legend["B3"].font = header_font
    for offset, code in enumerate((ON_QUEUE, BREAK, MEAL, OFF)):
        cell = legend.cell(row=4 + offset, column=1)
        cell.fill = fills[code]
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
        legend.cell(row=4 + offset, column=2, value=STATE_LABELS[code])
    legend.column_dimensions["A"].width = 10
    legend.column_dimensions["B"].width = 34
    legend["A9"] = "Each day sheet lists shifts touching that day, including overnight carryover."
    legend["A10"] = "Paid h is the portion paid on this day; (-1d)/(+1d) marks a date boundary."
    legend["A11"] = "Each grid column is 15 minutes. Panes are frozen after the detail columns."

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path
