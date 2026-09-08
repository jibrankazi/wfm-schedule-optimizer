"""Charts.

Two views. The coverage chart answers "did we staff the curve"; the Gantt
answers "what is each person actually doing". A roster that looks right in
aggregate can still be unworkable per-agent, so both matter.
"""

from __future__ import annotations

from pathlib import Path

# The backend is deliberately not forced here. Matplotlib already falls back
# to Agg when there is no display, so scripts and CI work headless, while a
# notebook keeps the inline backend ipykernel installed and renders the
# figures in the page. Calling matplotlib.use("Agg") at import time would
# break the second of those.
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import pandas as pd

from .generator import DAY_NAMES, DAYS, INTERVALS_PER_DAY, interval_labels
from .optimizer import Schedule
from .profiles import ServiceProfile

REQUIRED_COLOUR = "#C0392B"
SCHEDULED_COLOUR = "#2E86C1"
GAP_COLOUR = "#F1948A"
SURPLUS_COLOUR = "#AED6F1"


def plot_coverage(
    schedule: Schedule,
    day: int = 0,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Scheduled headcount against the Erlang C requirement for one day."""
    labels = interval_labels()
    required = schedule.required[day]
    scheduled = schedule.coverage[day]
    positions = np.arange(INTERVALS_PER_DAY)

    figure, axis = plt.subplots(figsize=(14, 6))

    axis.bar(
        positions,
        scheduled,
        width=0.86,
        color=SCHEDULED_COLOUR,
        alpha=0.75,
        label="Scheduled agents (MILP)",
        zorder=2,
    )
    axis.step(
        positions,
        required,
        where="mid",
        color=REQUIRED_COLOUR,
        linewidth=2.4,
        label="Required agents (Erlang C 80/20)",
        zorder=3,
    )

    gap = np.clip(required - scheduled, 0, None)
    if gap.any():
        axis.bar(
            positions,
            gap,
            bottom=scheduled,
            width=0.86,
            color=GAP_COLOUR,
            alpha=0.95,
            label="Understaffed",
            zorder=2,
        )

    axis.set_xticks(positions[::4])
    axis.set_xticklabels(labels[::4], rotation=45, ha="right", fontsize=9)
    axis.set_xlabel("Interval (15 minutes)", fontweight="bold")
    axis.set_ylabel("Concurrent agents on phone", fontweight="bold")
    axis.set_title(
        f"Scheduled vs required staffing - {DAY_NAMES[day]}\n"
        f"{int((required == scheduled).sum())} of {INTERVALS_PER_DAY} intervals matched exactly, "
        f"{int((scheduled >= required).sum())} fully covered",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(frameon=True, loc="upper right")
    axis.margins(x=0.01)
    axis.set_ylim(0, max(required.max(), scheduled.max()) * 1.18)  # headroom for the legend
    figure.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_week_coverage(
    schedule: Schedule, save_path: str | Path | None = None, dpi: int = 150
) -> Figure:
    """All five days stacked, for the shape of the week at a glance."""
    figure, axes = plt.subplots(DAYS, 1, figsize=(14, 13), sharex=True)
    labels = interval_labels()
    positions = np.arange(INTERVALS_PER_DAY)

    for day in range(DAYS):
        axis = axes[day]
        required = schedule.required[day]
        scheduled = schedule.coverage[day]

        axis.bar(positions, scheduled, width=0.86, color=SCHEDULED_COLOUR, alpha=0.75, zorder=2)
        axis.step(positions, required, where="mid", color=REQUIRED_COLOUR, linewidth=2.0, zorder=3)

        gap = np.clip(required - scheduled, 0, None)
        if gap.any():
            axis.bar(positions, gap, bottom=scheduled, width=0.86, color=GAP_COLOUR, zorder=2)

        shortfall = int(gap.sum())
        axis.set_ylabel(DAY_NAMES[day], fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        axis.set_axisbelow(True)
        axis.text(
            0.995,
            0.90,
            f"peak {required.max()} required | {shortfall} agent-intervals short",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#555555",
        )

    axes[-1].set_xticks(positions[::4])
    axes[-1].set_xticklabels(labels[::4], rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Interval (15 minutes)", fontweight="bold")
    axes[0].set_title("Scheduled vs required staffing across the week", fontsize=13, fontweight="bold")
    axes[0].legend(
        handles=[
            mpatches.Patch(color=SCHEDULED_COLOUR, alpha=0.75, label="Scheduled"),
            mpatches.Patch(color=GAP_COLOUR, label="Understaffed"),
            mpatches.Patch(color=REQUIRED_COLOUR, label="Required (Erlang C)"),
        ],
        loc="upper right",
        frameon=True,
        ncol=3,
    )
    figure.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_shift_gantt(
    schedule: Schedule,
    day: int = 0,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Per-agent shift bars for one day, breaks shown as gaps.

    Sorted by start time, which makes the staggered-start pattern the
    optimiser chose visible as a diagonal.
    """
    todays = sorted(
        [a for a in schedule.assignments if a.day == day],
        key=lambda a: (a.template.start, a.agent_name),
    )
    if not todays:
        raise ValueError(f"No assignments on day {day}")

    figure, axis = plt.subplots(figsize=(14, max(6, len(todays) * 0.17)))

    for row, assignment in enumerate(todays):
        template = assignment.template
        on_phone = np.array(template.coverage)
        # Draw contiguous on-phone runs so breaks appear as gaps.
        start = None
        for interval in range(INTERVALS_PER_DAY + 1):
            covered = interval < INTERVALS_PER_DAY and on_phone[interval]
            if covered and start is None:
                start = interval
            elif not covered and start is not None:
                axis.barh(row, interval - start, left=start, height=0.72, color=SCHEDULED_COLOUR, alpha=0.85)
                start = None
        for break_interval in template.breaks:
            axis.barh(row, 1, left=break_interval, height=0.72, color="#D5D8DC")

    axis.set_yticks(range(len(todays)))
    axis.set_yticklabels([a.agent_name for a in todays], fontsize=6)
    axis.invert_yaxis()
    axis.set_xticks(np.arange(0, INTERVALS_PER_DAY, 4))
    axis.set_xticklabels(interval_labels()[::4], rotation=45, ha="right", fontsize=9)
    axis.set_xlabel("Interval (15 minutes)", fontweight="bold")
    axis.set_title(
        f"Agent shift plan - {DAY_NAMES[day]} ({len(todays)} agents scheduled)\n"
        "blue = on phone, grey = rest or meal break",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.set_xlim(-0.5, INTERVALS_PER_DAY - 0.5)
    figure.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_profile_comparison(
    demand: "pd.DataFrame",
    profiles: dict[str, "ServiceProfile"] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Same demand, four service contracts: what each one costs to staff."""
    from .profile_analysis import compare_profiles, size_demand
    from .profiles import PROFILES

    profiles = profiles or PROFILES
    summary = compare_profiles(demand, profiles)

    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})

    day_one = demand[demand.day == 0]
    palette = ["#2E86C1", "#8E44AD", "#C0392B", "#16A085"]
    for colour, profile in zip(palette, profiles.values()):
        sized = size_demand(day_one, profile)
        left.step(
            range(len(sized)), sized.required_agents, where="mid",
            linewidth=2.1, color=colour, label=f"{profile.name} ({profile.label()})",
        )
    left.set_xlabel("Interval (15 minutes)", fontweight="bold")
    left.set_ylabel("Agents required", fontweight="bold")
    left.set_title("Same demand curve, four service contracts", fontsize=12, fontweight="bold")
    left.grid(axis="y", linestyle="--", alpha=0.4)
    left.set_axisbelow(True)
    left.legend(fontsize=9, frameon=True)

    bottom = None
    for label, colour in (
        ("set_by_service_level", "#2E86C1"),
        ("set_by_abandonment", "#8E44AD"),
        ("set_by_occupancy", "#E67E22"),
        ("set_by_floor", "#7F8C8D"),
        ("set_by_multiple", "#34495E"),
    ):
        values = summary[label].to_numpy()
        right.barh(
            summary.profile, values, left=bottom, color=colour,
            label=label.replace("set_by_", "").replace("_", " "),
        )
        bottom = values if bottom is None else bottom + values

    right.set_xlabel("Intervals, by which criterion set the headcount", fontweight="bold")
    right.set_title("What actually drives the number", fontsize=12, fontweight="bold")
    right.invert_yaxis()
    right.legend(fontsize=9, loc="lower right", frameon=True)
    right.grid(axis="x", linestyle="--", alpha=0.35)
    right.set_axisbelow(True)
    for index, row in enumerate(summary.itertuples()):
        right.text(
            row.agent_intervals * 0.0 + 4, index,
            f"{row.agent_hours:,.0f}h", va="center", fontsize=9, color="white", fontweight="bold",
        )

    figure.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return figure
