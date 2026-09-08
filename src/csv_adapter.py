"""Ingest external rosters and demand matrices, and screen them before solving.

The loaders are strict on purpose. A schedule built from a silently mis-parsed
roster looks exactly like a correct one, so every column is validated at the
boundary rather than left to surface as an odd assignment three steps later.

``screen_capacity`` is a screen, not a feasibility proof. It bounds the problem
from below on headcount and total hours; it does not model the presence floor,
the ESA weekly rest day, shift granularity, or agent availability. A roster that
passes can still be infeasible. Only the solver settles that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .continuous import DEFAULT_247_GRID, TimeGrid, WeeklyAgent
from .workforce_policy import operating_mask
from .workforce_policy import SectorConfig

AGENT_COLUMNS = frozenset({"id", "name", "kind", "min_weekly_hours", "max_weekly_hours"})
DEMAND_COLUMNS = frozenset({"week_interval", "required_agents"})
# Below this share of the required workload the roster cannot cover the week
# even with perfect packing, so the solve is not worth starting.
MIN_CAPACITY_RATIO = 0.75


def _parse_slot_tokens(raw: str, grid: TimeGrid, label: str) -> frozenset[int]:
    """Parse ``"0-95;192"`` into slot indices, bounds-checked against the grid."""
    text = raw.strip()
    if not text or text.lower() == "nan":
        return frozenset()

    slots: set[int] = set()
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"range {token!r} ends before it starts")
                slots.update(range(start, end + 1))
            else:
                slots.add(int(token))
        except ValueError as error:
            raise ValueError(f"{label}: cannot parse unavailable_slots token {token!r}") from error

    out_of_range = sorted(slot for slot in slots if not 0 <= slot < grid.total_intervals)
    if out_of_range:
        raise ValueError(
            f"{label}: unavailable_slots {out_of_range[:5]} fall outside the "
            f"planning grid 0..{grid.total_intervals - 1}"
        )
    return frozenset(slots)


def load_agents_csv(path: str | Path, grid: TimeGrid = DEFAULT_247_GRID) -> list[WeeklyAgent]:
    """Read an external roster into ``WeeklyAgent`` records.

    Required columns: id, name, kind, min_weekly_hours, max_weekly_hours.
    Optional: unavailable_slots, as semicolon-separated indices or ``start-end``
    ranges over absolute week slots.
    """
    frame = pd.read_csv(path)
    missing = AGENT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"agents csv is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("agents csv contains no rows")

    duplicated = frame["id"][frame["id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"agents csv has duplicate ids: {sorted(set(duplicated))}")

    has_unavailable = "unavailable_slots" in frame.columns
    agents: list[WeeklyAgent] = []
    for row in frame.itertuples():
        label = f"agent {row.id}"
        unavailable = (
            _parse_slot_tokens(str(getattr(row, "unavailable_slots", "")), grid, label)
            if has_unavailable
            else frozenset()
        )
        # WeeklyAgent has exactly six fields; anything else the sheet carries is
        # not modelled by the solver and is dropped here rather than silently
        # accepted and ignored downstream.
        agents.append(
            WeeklyAgent(
                id=int(row.id),
                name=str(row.name),
                kind=str(row.kind).strip(),
                min_weekly_hours=float(row.min_weekly_hours),
                max_weekly_hours=float(row.max_weekly_hours),
                unavailable_slots=unavailable,
            )
        )
    return agents


def load_demand_csv(path: str | Path, grid: TimeGrid = DEFAULT_247_GRID) -> pd.DataFrame:
    """Read an external demand matrix, validated against the planning grid.

    Required columns: week_interval, required_agents. ``min_presence_floor`` and
    ``calls`` are read by the solver when present and defaulted when absent.
    """
    frame = pd.read_csv(path)
    missing = DEMAND_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"demand csv is missing required columns: {sorted(missing)}")

    ordered = frame.sort_values("week_interval").reset_index(drop=True)
    if len(ordered) != grid.total_intervals:
        raise ValueError(
            f"demand rows ({len(ordered)}) must equal the planning grid intervals "
            f"({grid.total_intervals})"
        )
    if ordered["week_interval"].tolist() != list(range(grid.total_intervals)):
        raise ValueError(
            f"week_interval must cover 0..{grid.total_intervals - 1} exactly once"
        )
    if (ordered["required_agents"] < 0).any():
        raise ValueError("required_agents must be non-negative")
    if "min_presence_floor" in ordered.columns:
        if (ordered["min_presence_floor"] < 0).any():
            raise ValueError("min_presence_floor must be non-negative")
        if (ordered["min_presence_floor"] > ordered["required_agents"]).any():
            raise ValueError("min_presence_floor cannot exceed required_agents")
    return ordered


def screen_capacity(
    agents: list[WeeklyAgent],
    demand: pd.DataFrame,
    sector_cfg: SectorConfig | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> tuple[bool, str, dict[str, float]]:
    """Bound the problem before compiling a model.

    Passing this screen does not mean the instance is feasible; it means no
    obvious capacity obstruction was found. When ``sector_cfg`` is supplied,
    intervals outside the sector's operating window are excluded, because the
    solver zeroes them and counting them overstates the workload. On a weekday_extended week
    that difference is roughly a quarter of the total.
    """
    if not agents:
        return False, "No agents supplied.", {}

    required = np.zeros(grid.total_intervals, dtype=int)
    for row in demand.itertuples():
        required[int(row.week_interval)] = int(row.required_agents)

    if sector_cfg is not None:
        required = np.where(operating_mask(sector_cfg, grid), required, 0)

    capacity_hours = float(sum(agent.max_weekly_hours for agent in agents))
    required_hours = float(required.sum()) * grid.interval_minutes / 60.0
    peak = int(required.max())
    ratio = capacity_hours / max(1.0, required_hours)

    metrics = {
        "total_available_hours": round(capacity_hours, 1),
        "total_required_hours": round(required_hours, 1),
        "capacity_ratio": round(ratio, 3),
        "peak_agents_needed": float(peak),
        "roster_headcount": float(len(agents)),
    }

    if peak > len(agents):
        return (
            False,
            f"Capacity screen failed: peak requirement of {peak} agents exceeds the "
            f"roster headcount of {len(agents)}.",
            metrics,
        )
    if ratio < MIN_CAPACITY_RATIO:
        return (
            False,
            f"Capacity screen failed: {capacity_hours:.1f}h of contracted capacity covers "
            f"only {ratio * 100:.1f}% of the {required_hours:.1f}h workload.",
            metrics,
        )
    return (
        True,
        f"Capacity screen passed: {ratio * 100:.1f}% of the {required_hours:.1f}h workload "
        "is covered by contracted capacity. This is a bound, not a feasibility proof.",
        metrics,
    )
