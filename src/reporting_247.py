"""Charts for the continuous 24/7 model.

Everything here reads the artifacts a solve already wrote - the interval
coverage report and the agent assignment export - so the pictures always
describe the run that is published, never a fresh one that happens to differ.

The backend is deliberately not forced. Matplotlib falls back to Agg when
there is no display, so scripts and CI work headless while a notebook keeps
the inline backend and renders in the page.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
INTERVALS_PER_DAY = 96
DAYS = 7

REQUIRED = "#C0392B"
SCHEDULED = "#2E86C1"
GAP = "#F1948A"
SURPLUS = "#AED6F1"
FLOOR = "#7F8C8D"

# The 24/7 coverage report records a two-value taxonomy: an interval is sized
# either by the queueing calculation or by the presence floor. The finer
# split (service level vs occupancy cap) lives in profile_analysis.size_demand
# on the weekday path, not here.
BINDING_COLOURS = {
    "queueing": "#2E86C1",
    "floor": "#7F8C8D",
}


def interval_label(index: int) -> str:
    minutes = index * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _save(figure: Figure, save_path: str | Path | None, dpi: int) -> Figure:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure


# ---------------------------------------------------------------------------
# demand: what the week of contacts actually looks like
# ---------------------------------------------------------------------------
def plot_call_volume_week(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Raw contact arrivals across the whole week, day by day.

    This is the input, before any queueing maths. The shape is what everything
    downstream is reacting to: two daytime peaks, a deep overnight trough that
    never quite reaches zero, and lighter weekends.
    """
    figure, axis = plt.subplots(figsize=(15, 5))
    calls = coverage.sort_values("week_interval")["calls"].to_numpy()
    axis.fill_between(range(len(calls)), calls, color=SCHEDULED, alpha=0.5, linewidth=0)
    axis.plot(range(len(calls)), calls, color="#1B4F72", linewidth=0.9)

    for day in range(1, DAYS):
        axis.axvline(day * INTERVALS_PER_DAY, color="#B0B0B0", linestyle="--", linewidth=0.8)
    axis.set_xticks([day * INTERVALS_PER_DAY + 48 for day in range(DAYS)])
    axis.set_xticklabels(DAY_NAMES, fontweight="bold")
    axis.set_xlim(0, len(calls))
    axis.set_ylabel("Contacts arriving per 15 min", fontweight="bold")
    axis.set_title(
        "Contact arrivals across the week\n"
        f"{calls.sum():,.0f} contacts total, peak {calls.max():.0f} in a single interval",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_intraday_profile(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Call frequency by time of day, with the spread across the seven days.

    The band is the day-to-day variation at each interval. A forecast is a
    point estimate through the middle of it, which is why a schedule optimised
    against the mean is fragile at the edges.
    """
    pivot = coverage.pivot_table(index="interval", columns="day", values="calls")
    low, mid, high = pivot.min(axis=1), pivot.median(axis=1), pivot.max(axis=1)

    figure, axis = plt.subplots(figsize=(14, 5.5))
    axis.fill_between(pivot.index, low, high, color=SCHEDULED, alpha=0.22, label="Mon-Sun range")
    axis.plot(pivot.index, mid, color="#1B4F72", linewidth=2.2, label="Median day")
    axis.axvspan(0, 24, color=FLOOR, alpha=0.10)
    axis.axvspan(88, 96, color=FLOOR, alpha=0.10)
    axis.text(12, high.max() * 0.92, "overnight", ha="center", fontsize=9, color="#555555")

    axis.set_xticks(range(0, INTERVALS_PER_DAY, 8))
    axis.set_xticklabels([interval_label(i) for i in range(0, INTERVALS_PER_DAY, 8)],
                         rotation=45, ha="right", fontsize=9)
    axis.set_xlim(0, INTERVALS_PER_DAY - 1)
    axis.set_xlabel("Time of day", fontweight="bold")
    axis.set_ylabel("Contacts per 15 min", fontweight="bold")
    axis.set_title("Call frequency by time of day", fontsize=12, fontweight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_sizing_curve(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """How contact volume converts into required headcount.

    Not a straight line. Queueing has economies of scale, so the marginal
    agent covers more contacts as the queue grows - and the flat section at
    the bottom left is the presence floor, where staffing is set by the
    operating model rather than by arithmetic.
    """
    figure, axis = plt.subplots(figsize=(9, 6))
    floor_bound = coverage.binding_constraint == "floor"
    for label, mask, colour in (
        ("Set by service level / occupancy", ~floor_bound, SCHEDULED),
        ("Set by presence floor", floor_bound, FLOOR),
    ):
        subset = coverage[mask]
        axis.scatter(subset["calls"], subset["required_agents"], s=14, alpha=0.55,
                     color=colour, label=label, edgecolors="none")

    axis.set_xlabel("Contacts per 15 min", fontweight="bold")
    axis.set_ylabel("Agents required", fontweight="bold")
    axis.set_title("Contact volume to required headcount\nProfile queue sizing, with a presence floor",
                   fontsize=12, fontweight="bold")
    axis.grid(linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=True, loc="upper left")
    figure.tight_layout()
    return _save(figure, save_path, dpi)


# ---------------------------------------------------------------------------
# coverage: what the solver produced against what was needed
# ---------------------------------------------------------------------------
def plot_day_coverage(
    coverage: pd.DataFrame, day: int = 0, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """One full 24-hour day: scheduled headcount against the requirement."""
    frame = coverage[coverage.day == day].sort_values("interval")
    required = frame["required_agents"].to_numpy()
    scheduled = frame["scheduled_active"].to_numpy()
    floor = frame["min_presence_floor"].to_numpy()
    positions = np.arange(len(frame))

    figure, axis = plt.subplots(figsize=(15, 6))
    axis.bar(positions, scheduled, width=0.88, color=SCHEDULED, alpha=0.75,
             label="Scheduled on phone", zorder=2)
    shortfall = np.clip(required - scheduled, 0, None)
    if shortfall.any():
        axis.bar(positions, shortfall, bottom=scheduled, width=0.88, color=GAP,
                 label="Understaffed", zorder=2)
    axis.step(positions, required, where="mid", color=REQUIRED, linewidth=2.3,
              label="Required (service profile)", zorder=3)
    axis.step(positions, floor, where="mid", color=FLOOR, linewidth=1.6,
              linestyle="--", label="Presence floor", zorder=3)

    axis.set_xticks(positions[::8])
    axis.set_xticklabels([interval_label(i) for i in range(0, INTERVALS_PER_DAY, 8)],
                         rotation=45, ha="right", fontsize=9)
    axis.set_xlabel("Time of day", fontweight="bold")
    axis.set_ylabel("Concurrent agents", fontweight="bold")
    axis.set_title(
        f"Scheduled vs required staffing - {DAY_NAMES[day]} (24 hours)\n"
        f"{int((scheduled >= required).sum())} of {len(frame)} intervals fully covered",
        fontsize=12, fontweight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.set_ylim(0, max(required.max(), scheduled.max()) * 1.2)
    axis.legend(frameon=True, loc="upper right", ncol=2)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_week_coverage(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """All seven days stacked, for the shape of the whole week at once."""
    figure, axes = plt.subplots(DAYS, 1, figsize=(15, 16), sharex=True)
    for day in range(DAYS):
        frame = coverage[coverage.day == day].sort_values("interval")
        required = frame["required_agents"].to_numpy()
        scheduled = frame["scheduled_active"].to_numpy()
        positions = np.arange(len(frame))
        axis = axes[day]

        axis.bar(positions, scheduled, width=0.9, color=SCHEDULED, alpha=0.75, zorder=2)
        shortfall = np.clip(required - scheduled, 0, None)
        if shortfall.any():
            axis.bar(positions, shortfall, bottom=scheduled, width=0.9, color=GAP, zorder=2)
        axis.step(positions, required, where="mid", color=REQUIRED, linewidth=1.8, zorder=3)

        axis.set_ylabel(DAY_NAMES[day], fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
        axis.text(0.995, 0.88, f"peak {required.max()} required | {int(shortfall.sum())} short",
                  transform=axis.transAxes, ha="right", va="top", fontsize=9, color="#555555")

    axes[-1].set_xticks(range(0, INTERVALS_PER_DAY, 8))
    axes[-1].set_xticklabels([interval_label(i) for i in range(0, INTERVALS_PER_DAY, 8)],
                             rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Time of day", fontweight="bold")
    axes[0].set_title("Scheduled vs required staffing across the full week (672 intervals)",
                      fontsize=13, fontweight="bold")
    axes[0].legend(handles=[
        mpatches.Patch(color=SCHEDULED, alpha=0.75, label="Scheduled"),
        mpatches.Patch(color=GAP, label="Understaffed"),
        mpatches.Patch(color=REQUIRED, label="Required"),
    ], loc="upper right", ncol=3, frameon=True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_coverage_heatmap(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Surplus and shortfall as a day-by-time grid.

    Red is short, blue is surplus, white is exact. Reading down a column shows
    whether a time of day is chronically hard to staff, which a per-day chart
    hides.
    """
    grid = coverage.pivot_table(index="day", columns="interval", values="net_slack").to_numpy()
    limit = np.abs(grid).max()

    figure, axis = plt.subplots(figsize=(15, 4.2))
    image = axis.imshow(grid, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit,
                        interpolation="nearest")
    axis.set_yticks(range(DAYS))
    axis.set_yticklabels(DAY_NAMES, fontweight="bold")
    axis.set_xticks(range(0, INTERVALS_PER_DAY, 8))
    axis.set_xticklabels([interval_label(i) for i in range(0, INTERVALS_PER_DAY, 8)],
                         rotation=45, ha="right", fontsize=9)
    axis.set_xlabel("Time of day", fontweight="bold")
    axis.set_title("Staffing surplus and shortfall by day and interval\n"
                   "red = short of requirement, blue = surplus to it",
                   fontsize=12, fontweight="bold")
    figure.colorbar(image, ax=axis, label="scheduled - required")
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_binding_constraint(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Which rule set the headcount in every interval of the week.

    The distinctive chart in this project. Overnight is grey - staffing there
    is a floor decision, not a queueing one, and no amount of Erlang maths
    would produce it. Daytime is blue or orange depending on whether the
    service level or the occupancy ceiling bound first.
    """
    order = [name for name in BINDING_COLOURS if (coverage.binding_constraint == name).any()]
    codes = {name: index for index, name in enumerate(order)}
    grid = (
        coverage.assign(code=coverage.binding_constraint.map(codes).fillna(0))
        .pivot_table(index="day", columns="interval", values="code")
        .to_numpy()
    )
    colours = [BINDING_COLOURS[name] for name in order]
    cmap = plt.matplotlib.colors.ListedColormap(colours)

    figure, axis = plt.subplots(figsize=(15, 4.2))
    axis.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(order) - 1,
                interpolation="nearest")
    axis.set_yticks(range(DAYS))
    axis.set_yticklabels(DAY_NAMES, fontweight="bold")
    axis.set_xticks(range(0, INTERVALS_PER_DAY, 8))
    axis.set_xticklabels([interval_label(i) for i in range(0, INTERVALS_PER_DAY, 8)],
                         rotation=45, ha="right", fontsize=9)
    axis.set_xlabel("Time of day", fontweight="bold")

    counts = coverage.binding_constraint.value_counts()
    axis.set_title("Which constraint set the headcount, interval by interval",
                   fontsize=12, fontweight="bold")
    total = len(coverage)
    axis.legend(handles=[
        mpatches.Patch(
            color=BINDING_COLOURS[name],
            label=f"sized by {name} - {counts.get(name, 0)} intervals "
                  f"({100 * counts.get(name, 0) / total:.0f}%)",
        )
        for name in order
    ], loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=len(order), frameon=True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_shortfall_by_hour(
    coverage: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Where in the day the schedule falls short, summed across the week."""
    frame = coverage.copy()
    frame["hour"] = frame["interval"] // 4
    grouped = frame.groupby("hour")["demand_shortfall"].sum()

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(grouped.index, grouped.to_numpy(), color=GAP, edgecolor="#C0392B", linewidth=0.6)
    axis.set_xticks(range(0, 24, 2))
    axis.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right")
    axis.set_xlabel("Hour of day", fontweight="bold")
    axis.set_ylabel("Understaffed agent-intervals (week)", fontweight="bold")
    axis.set_title(f"When the schedule falls short\n{int(grouped.sum())} understaffed "
                   "agent-intervals across the week",
                   fontsize=12, fontweight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


# ---------------------------------------------------------------------------
# roster: what the people are doing
# ---------------------------------------------------------------------------
def plot_agent_gantt(
    assignments: pd.DataFrame, day: int = 0, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Each agent's shift for one day, sorted by start time."""
    todays = assignments[assignments.day == day].sort_values(["start_slot", "agent_name"])
    if todays.empty:
        raise ValueError(f"no assignments on day {day}")

    figure, axis = plt.subplots(figsize=(15, max(5, len(todays) * 0.22)))
    for row, record in enumerate(todays.itertuples()):
        start = int(record.start_slot) % INTERVALS_PER_DAY
        span = int(record.span_slots)
        axis.barh(row, span, left=start, height=0.7,
                  color=SCHEDULED if not record.is_night else "#5D6D7E", alpha=0.85)

    axis.set_yticks(range(len(todays)))
    axis.set_yticklabels(todays["agent_name"], fontsize=6)
    axis.invert_yaxis()
    axis.set_xticks(range(0, INTERVALS_PER_DAY + 1, 8))
    axis.set_xticklabels([interval_label(i % INTERVALS_PER_DAY)
                          for i in range(0, INTERVALS_PER_DAY + 1, 8)],
                         rotation=45, ha="right", fontsize=9)
    axis.set_xlim(0, INTERVALS_PER_DAY)
    axis.set_xlabel("Time of day", fontweight="bold")
    axis.set_title(f"Agent shift plan - {DAY_NAMES[day]} ({len(todays)} agents on duty)\n"
                   "grey = shift covering the overnight window",
                   fontsize=12, fontweight="bold")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_agent_hours(
    assignments: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """Weekly paid hours per agent, sorted, so outliers are visible."""
    hours = assignments.groupby("agent_name")["paid_hours"].sum().sort_values()

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(range(len(hours)), hours.to_numpy(), color=SCHEDULED, alpha=0.8, width=0.9)
    axis.axhline(hours.mean(), color=REQUIRED, linestyle="--", linewidth=1.6,
                 label=f"mean {hours.mean():.1f}h")
    axis.set_xlabel("Agents, sorted by weekly hours", fontweight="bold")
    axis.set_ylabel("Paid hours in the week", fontweight="bold")
    axis.set_title(f"Weekly paid hours per agent\n{len(hours)} agents scheduled, "
                   f"{hours.min():.1f}h to {hours.max():.1f}h",
                   fontsize=12, fontweight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=True)
    axis.set_xticks([])
    figure.tight_layout()
    return _save(figure, save_path, dpi)


def plot_night_distribution(
    assignments: pd.DataFrame, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """How night work is spread across the roster.

    The measured trade-off lives here: under default weights the solver
    concentrates overnight shifts on a subset rather than rotating them,
    because spreading them costs daytime coverage.
    """
    nights = (
        assignments.assign(night=assignments.is_night.astype(str).str.lower().isin(["true", "1"]))
        .groupby("agent_name")["night"].sum()
    )
    counts = nights.value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(counts.index, counts.to_numpy(), color=["#5D6D7E" if n == 0 else SCHEDULED
                                                     for n in counts.index], width=0.65)
    for x, y in zip(counts.index, counts.to_numpy()):
        axis.text(x, y + 0.4, str(int(y)), ha="center", fontsize=10, fontweight="bold")
    axis.set_xlabel("Night shifts worked in the week", fontweight="bold")
    axis.set_ylabel("Number of agents", fontweight="bold")
    axis.set_title("Distribution of overnight work across the roster",
                   fontsize=12, fontweight="bold")
    axis.set_xticks(counts.index)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, save_path, dpi)
