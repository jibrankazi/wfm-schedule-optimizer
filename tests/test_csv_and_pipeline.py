"""Tests for the external ingestion layer and the sector CLI.

``run_pipeline.py`` and ``.github/workflows/ci-cd.yml`` are deliberately not
exercised or modified here; the sector runner lives in ``run_sector.py`` so the
existing pipeline entry point and its CI step keep working unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.continuous import (
    DEFAULT_247_GRID,
    generate_continuous_demand,
    generate_continuous_roster,
)
from src.csv_adapter import load_agents_csv, load_demand_csv, screen_capacity
from src.workforce_policy import get_profile_a_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agents_csv_parses_ranges_and_drops_unmodelled_columns(tmp_path: Path):
    """Ranges expand, blanks are empty, and columns the solver cannot read are dropped."""
    path = tmp_path / "agents.csv"
    path.write_text(
        "id,name,kind,min_weekly_hours,max_weekly_hours,unavailable_slots,max_consecutive_days\n"
        "0,Alice,full_time,32.0,40.0,0-95;192,5\n"
        "1,Bob,part_time,16.0,24.0,,4\n",
        encoding="utf-8",
    )

    agents = load_agents_csv(path)
    assert [a.id for a in agents] == [0, 1]
    assert agents[0].unavailable_slots == frozenset(range(0, 96)) | {192}
    assert agents[1].unavailable_slots == frozenset()
    # max_consecutive_days is not a WeeklyAgent field and must not be smuggled on.
    assert not hasattr(agents[0], "max_consecutive_days")


def test_agents_csv_rejects_slots_outside_the_grid(tmp_path: Path):
    path = tmp_path / "agents.csv"
    path.write_text(
        "id,name,kind,min_weekly_hours,max_weekly_hours,unavailable_slots\n"
        "0,Alice,full_time,32.0,40.0,700\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the planning grid"):
        load_agents_csv(path)


def test_demand_csv_validates_grid_dimensions(tmp_path: Path):
    path = tmp_path / "demand.csv"
    path.write_text("week_interval,required_agents\n0,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="planning grid intervals"):
        load_demand_csv(path, grid=DEFAULT_247_GRID)


def test_demand_csv_round_trips_the_generator(tmp_path: Path):
    path = tmp_path / "demand.csv"
    generate_continuous_demand(seed=42, peak_calls=25.0).to_csv(path, index=False)
    loaded = load_demand_csv(path)
    assert len(loaded) == DEFAULT_247_GRID.total_intervals
    assert loaded["week_interval"].tolist() == list(range(DEFAULT_247_GRID.total_intervals))


def test_capacity_screen_catches_a_headcount_deficit():
    demand = generate_continuous_demand(seed=42, peak_calls=50.0)
    passed, message, metrics = screen_capacity(
        generate_continuous_roster(full_time=2, part_time=0), demand
    )
    assert passed is False
    assert "exceeds the roster headcount" in message
    assert metrics["roster_headcount"] == 2.0


def test_capacity_screen_masks_closed_hours_for_windowed_sectors():
    """Counting the closed 22:00-07:00 stretch overstates a seven-day extended week's workload."""
    demand = generate_continuous_demand(seed=42, peak_calls=25.0)
    agents = generate_continuous_roster(full_time=26, part_time=10)

    _, _, unmasked = screen_capacity(agents, demand)
    _, _, masked = screen_capacity(agents, demand, sector_cfg=get_profile_a_config())

    assert masked["total_required_hours"] < unmasked["total_required_hours"]
    assert masked["capacity_ratio"] > unmasked["capacity_ratio"]
    # The seven-day extended pattern is open 15 of every 24 hours, so the masked workload should be well under
    # the round-the-clock figure.
    assert masked["total_required_hours"] < 0.9 * unmasked["total_required_hours"]



