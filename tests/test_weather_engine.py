"""Tests for the weather elasticity workload transformer.

`src/weather_engine.py` sits upstream of the solver: it modifies `calls`
and `required_agents` based on meteorological conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.continuous import (
    CONTINUOUS_247_PROFILE,
    DEFAULT_247_GRID,
    generate_continuous_demand,
)
from src.profile_analysis import required_agents_for_profile
from src.weather_engine import (
    WeatherConditions,
    WeatherElasticity,
    apply_weather_demand,
    detect_freeze_thaw,
    load_weather_fixture,
    weather_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "weather_sample.json"


def _demand():
    return generate_continuous_demand(seed=42, peak_calls=25.0)


def test_weather_elasticity_parameter_guards():
    with pytest.raises(ValueError, match="alpha_rain"):
        WeatherElasticity(alpha_rain=-0.01)
    with pytest.raises(ValueError, match="alpha_snow"):
        WeatherElasticity(alpha_snow=-0.01)
    with pytest.raises(ValueError, match="delta_freeze_thaw"):
        WeatherElasticity(delta_freeze_thaw=-0.1)
    with pytest.raises(ValueError, match="max_multiplier"):
        WeatherElasticity(max_multiplier=0.5)


def test_multiplier_logic_and_capping():
    elasticity = WeatherElasticity(
        alpha_rain=0.03, alpha_snow=0.05, delta_freeze_thaw=0.20, max_multiplier=2.0
    )
    assert elasticity.multiplier(0.0, 0.0, False) == pytest.approx(1.0)
    assert elasticity.multiplier(10.0, 0.0, False) == pytest.approx(1.30)
    assert elasticity.multiplier(0.0, 10.0, True) == pytest.approx(1.70)
    assert elasticity.multiplier(50.0, 50.0, True) == pytest.approx(2.0)


def test_detect_freeze_thaw_identifies_zero_crossings():
    assert detect_freeze_thaw(np.full(32, -5.0)).sum() == 0
    assert detect_freeze_thaw(np.full(32, 10.0)).sum() == 0

    crossing = np.concatenate([np.full(16, -3.0), np.full(16, 2.0)])
    transitions = detect_freeze_thaw(crossing, window_intervals=8)
    assert transitions[:16].sum() == 0
    assert transitions[16:24].sum() > 0


def test_zero_weather_leaves_demand_and_headcount_unchanged():
    demand = _demand()
    adjusted = apply_weather_demand(demand, weather=None)
    ordered = demand.sort_values("week_interval")
    assert np.array_equal(
        adjusted["required_agents"].to_numpy(), ordered["required_agents"].to_numpy()
    )
    assert np.allclose(adjusted["calls"].to_numpy(), ordered["calls"].to_numpy())
    assert (adjusted["weather_multiplier"] == 1.0).all()


def test_weather_demand_is_monotonic_and_respects_floor():
    demand = _demand()
    n = DEFAULT_247_GRID.total_intervals
    conditions = WeatherConditions(
        rain_mm=np.full(n, 5.0),
        snow_cm=np.zeros(n),
        freeze_thaw=np.zeros(n, dtype=int),
    )
    adjusted = apply_weather_demand(demand, weather=conditions)

    ordered = demand.sort_values("week_interval")
    before = ordered["required_agents"].to_numpy()
    after = adjusted["required_agents"].to_numpy()
    floor = ordered["min_presence_floor"].to_numpy()

    assert (after >= before).all()
    assert (after >= floor).all()
    assert after.sum() > before.sum()


def test_caller_frame_is_not_mutated():
    demand = _demand()
    before_calls = demand["calls"].to_numpy().copy()
    before_agents = demand["required_agents"].to_numpy().copy()

    n = DEFAULT_247_GRID.total_intervals
    apply_weather_demand(
        demand,
        weather=WeatherConditions(
            rain_mm=np.full(n, 10.0),
            snow_cm=np.zeros(n),
            freeze_thaw=np.zeros(n, dtype=int),
        ),
    )

    assert np.array_equal(demand["calls"].to_numpy(), before_calls)
    assert np.array_equal(demand["required_agents"].to_numpy(), before_agents)
    assert "weather_multiplier" not in demand.columns


def test_fixture_loading_and_tiling(tmp_path):
    fixture_file = tmp_path / "test_fixture.json"
    fixture_file.write_text(
        json.dumps({"rain_mm": [2.0] * 96, "snow_cm": [0.0] * 96, "freeze_thaw": [1] * 96}),
        encoding="utf-8",
    )

    conditions = load_weather_fixture(fixture_file, target_intervals=672)
    assert len(conditions.rain_mm) == 672
    assert (conditions.rain_mm == 2.0).all()
    assert (conditions.freeze_thaw == 1).all()


def test_shipped_sample_fixture_loads_and_derives_freeze_thaw():
    """The repo fixture carries temperature, not a freeze_thaw flag."""
    conditions = load_weather_fixture(SAMPLE_FIXTURE)
    assert len(conditions.rain_mm) == DEFAULT_247_GRID.total_intervals
    assert conditions.rain_mm.max() == pytest.approx(14.2)
    assert conditions.snow_cm.max() == pytest.approx(4.0)
    assert 0 < conditions.freeze_thaw.sum() < DEFAULT_247_GRID.total_intervals


def test_fixture_rejects_lengths_that_cannot_tile(tmp_path):
    fixture_file = tmp_path / "odd.json"
    fixture_file.write_text(
        json.dumps({"rain_mm": [1.0] * 100, "snow_cm": [0.0] * 100}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot be evenly tiled"):
        load_weather_fixture(fixture_file, target_intervals=672)


def test_missing_fixture_raises():
    with pytest.raises(FileNotFoundError):
        load_weather_fixture("data/fixtures/does_not_exist.json")


def test_weather_report_quantifies_surge():
    demand = _demand()
    n = DEFAULT_247_GRID.total_intervals
    conditions = WeatherConditions(
        rain_mm=np.concatenate([np.full(100, 10.0), np.zeros(n - 100)]),
        snow_cm=np.zeros(n),
        freeze_thaw=np.zeros(n, dtype=int),
    )
    adjusted = apply_weather_demand(demand, weather=conditions)
    report = weather_report(demand, adjusted)

    assert report["intervals_impacted"] == 100
    assert report["added_calls"] > 0
    assert report["added_agent_intervals"] > 0
    assert report["added_agent_hours"] == pytest.approx(
        report["added_agent_intervals"] * 0.25
    )
    assert report["peak_after"] >= report["peak_before"]


def test_weather_frame_loads_through_csv_adapter(tmp_path):
    from src.csv_adapter import load_demand_csv

    adjusted = apply_weather_demand(_demand(), weather=None)
    csv_file = tmp_path / "weather_demand.csv"
    adjusted.to_csv(csv_file, index=False)

    loaded = load_demand_csv(csv_file)
    assert len(loaded) == DEFAULT_247_GRID.total_intervals
    assert (loaded["required_agents"] >= loaded["min_presence_floor"]).all()


def test_weather_sizing_matches_hand_computed_interval():
    demand = _demand()
    n = DEFAULT_247_GRID.total_intervals
    rain = np.zeros(n)
    rain[200] = 20.0  # +50% at alpha=0.025
    adjusted = apply_weather_demand(
        demand,
        weather=WeatherConditions(
            rain_mm=rain, snow_cm=np.zeros(n), freeze_thaw=np.zeros(n, dtype=int)
        ),
    )

    row = demand.sort_values("week_interval").iloc[200]
    expected_calls = float(row["calls"]) * 1.50
    expected_agents = required_agents_for_profile(
        expected_calls, float(row["aht_sec"]), CONTINUOUS_247_PROFILE
    )

    assert int(adjusted["required_agents"].iloc[200]) == max(
        expected_agents, int(row["required_agents"]), int(row["min_presence_floor"])
    )


