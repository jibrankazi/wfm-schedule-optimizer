"""Translate weather shocks (rain, snow, freeze-thaw) into demand surges.

Weather events do not generate uniform call volume; they create localized or
sustained surges in contact arrivals (e.g., storm drain backups from heavy rain,
road clearance and salting requests during snowfall, and water main breaks or
potholes triggered by freeze-thaw cycles).

Like `src/concurrency.py`, this module sits upstream of the solver:
1. It scales arrival volumes (`calls`) according to meteorological elasticity.
2. It resizes `required_agents` using the `ServiceProfile` that generated the
   frame (dispatching to Erlang A or Erlang C as appropriate).
3. It enforces monotonicity: adding weather shocks can never reduce headcount
   below the baseline or the presence floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .continuous import CONTINUOUS_247_PROFILE, DEFAULT_247_GRID, TimeGrid
from .profile_analysis import required_agents_for_profile
from .profiles import ServiceProfile

# Default elasticity coefficients (illustrative priors, subject to empirical calibration)
DEFAULT_ALPHA_RAIN = 0.025   # +2.5% arrival volume per mm of precipitation
DEFAULT_ALPHA_SNOW = 0.060   # +6.0% arrival volume per cm of snowfall
DEFAULT_DELTA_FREEZE = 0.200  # +20.0% surge during active freeze-thaw transition
DEFAULT_MAX_MULTIPLIER = 3.0  # Cap to protect against sensor anomalies or extreme data


@dataclass(frozen=True)
class WeatherElasticity:
    """Elasticity coefficients governing how weather conditions scale call arrivals."""

    alpha_rain: float = DEFAULT_ALPHA_RAIN
    alpha_snow: float = DEFAULT_ALPHA_SNOW
    delta_freeze_thaw: float = DEFAULT_DELTA_FREEZE
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER

    def __post_init__(self) -> None:
        if self.alpha_rain < 0.0:
            raise ValueError("alpha_rain must be non-negative")
        if self.alpha_snow < 0.0:
            raise ValueError("alpha_snow must be non-negative")
        if self.delta_freeze_thaw < 0.0:
            raise ValueError("delta_freeze_thaw must be non-negative")
        if self.max_multiplier < 1.0:
            raise ValueError("max_multiplier must be at least 1.0")

    def multiplier(self, rain_mm: float, snow_cm: float, freeze_thaw: bool | float) -> float:
        """Calculate the combined demand multiplier for a single interval."""
        raw = (
            1.0
            + (self.alpha_rain * max(0.0, rain_mm))
            + (self.alpha_snow * max(0.0, snow_cm))
            + (self.delta_freeze_thaw * (1.0 if freeze_thaw else 0.0))
        )
        return float(min(raw, self.max_multiplier))


@dataclass(frozen=True, eq=False)
class WeatherConditions:
    """Interval-aligned weather observations or forecast series."""

    rain_mm: np.ndarray
    snow_cm: np.ndarray
    freeze_thaw: np.ndarray

    def __post_init__(self) -> None:
        if len(self.rain_mm) != len(self.snow_cm) or len(self.rain_mm) != len(self.freeze_thaw):
            raise ValueError("All weather condition arrays must have identical length")
        if (self.rain_mm < 0).any():
            raise ValueError("rain_mm values must be non-negative")
        if (self.snow_cm < 0).any():
            raise ValueError("snow_cm values must be non-negative")


def detect_freeze_thaw(
    temperature_c: np.ndarray,
    freezing_threshold: float = 0.0,
    window_intervals: int = 16,  # 4 hours at 15-minute intervals
) -> np.ndarray:
    """Detect freeze-thaw events where temperature crosses 0C within a rolling window.

    A freeze-thaw transition occurs if the current temperature is above freezing
    while recent intervals were sub-zero, or vice versa.
    """
    temp = np.asarray(temperature_c, dtype=float)
    n = len(temp)
    is_freezing = temp < freezing_threshold
    freeze_thaw = np.zeros(n, dtype=int)

    for i in range(n):
        start = max(0, i - window_intervals)
        window = is_freezing[start : i + 1]
        if window.any() and not window.all():
            freeze_thaw[i] = 1

    return freeze_thaw


def load_weather_fixture(
    fixture_path: Path | str,
    target_intervals: int = DEFAULT_247_GRID.total_intervals,
) -> WeatherConditions:
    """Load offline weather fixture JSON and broadcast/validate to target intervals."""
    path = Path(fixture_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather fixture not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("rain_mm", "snow_cm"):
        if key not in data:
            raise ValueError(f"Fixture missing required key: {key}")

    rain = np.asarray(data["rain_mm"], dtype=float)
    snow = np.asarray(data["snow_cm"], dtype=float)

    if "freeze_thaw" in data:
        freeze = np.asarray(data["freeze_thaw"], dtype=int)
    elif "temperature_c" in data:
        freeze = detect_freeze_thaw(np.asarray(data["temperature_c"], dtype=float))
    else:
        freeze = np.zeros_like(rain, dtype=int)

    # Allow 1-day fixtures (96 intervals) to tile across the full weekly grid (672)
    if len(rain) != target_intervals:
        if target_intervals % len(rain) == 0:
            reps = target_intervals // len(rain)
            rain = np.tile(rain, reps)
            snow = np.tile(snow, reps)
            freeze = np.tile(freeze, reps)
        else:
            raise ValueError(
                f"Fixture length ({len(rain)}) does not match target ({target_intervals}) "
                f"and cannot be evenly tiled."
            )

    return WeatherConditions(rain_mm=rain, snow_cm=snow, freeze_thaw=freeze)


def apply_weather_demand(
    demand: pd.DataFrame,
    weather: WeatherConditions | Path | str | None = None,
    elasticity: WeatherElasticity | None = None,
    profile: ServiceProfile = CONTINUOUS_247_PROFILE,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> pd.DataFrame:
    """Return a copy of ``demand`` scaled for weather shocks and resized.

    The caller's frame is never modified. ``required_agents`` is re-sized using
    ``profile`` and floored at ``max(sized, baseline, min_presence_floor)``.
    """
    required_columns = {"week_interval", "calls", "aht_sec", "required_agents"}
    missing = required_columns.difference(demand.columns)
    if missing:
        raise ValueError(f"demand is missing required columns: {sorted(missing)}")

    model_elasticity = elasticity or WeatherElasticity()
    ordered = demand.sort_values("week_interval").reset_index(drop=True)
    if len(ordered) != grid.total_intervals:
        raise ValueError(
            f"demand rows ({len(ordered)}) must equal planning grid intervals "
            f"({grid.total_intervals})"
        )

    if weather is None:
        conditions = WeatherConditions(
            rain_mm=np.zeros(grid.total_intervals),
            snow_cm=np.zeros(grid.total_intervals),
            freeze_thaw=np.zeros(grid.total_intervals, dtype=int),
        )
    elif isinstance(weather, (str, Path)):
        conditions = load_weather_fixture(weather, target_intervals=grid.total_intervals)
    elif isinstance(weather, WeatherConditions):
        if len(weather.rain_mm) != grid.total_intervals:
            raise ValueError(
                f"WeatherConditions length ({len(weather.rain_mm)}) does not match "
                f"grid ({grid.total_intervals})"
            )
        conditions = weather
    else:
        raise TypeError(f"Unsupported weather input type: {type(weather)}")

    calls = ordered["calls"].to_numpy(dtype=float)
    aht = ordered["aht_sec"].to_numpy(dtype=float)
    baseline = ordered["required_agents"].to_numpy(dtype=int)
    floor = (
        ordered["min_presence_floor"].to_numpy(dtype=int)
        if "min_presence_floor" in ordered.columns
        else np.zeros(grid.total_intervals, dtype=int)
    )

    multipliers = np.zeros(grid.total_intervals, dtype=float)
    scaled_calls = np.zeros(grid.total_intervals, dtype=float)
    headcount = np.zeros(grid.total_intervals, dtype=int)

    for i in range(grid.total_intervals):
        mult = model_elasticity.multiplier(
            float(conditions.rain_mm[i]),
            float(conditions.snow_cm[i]),
            bool(conditions.freeze_thaw[i]),
        )
        multipliers[i] = mult
        new_calls = calls[i] * mult
        scaled_calls[i] = round(new_calls, 4)

        sized = required_agents_for_profile(new_calls, float(aht[i]), profile)
        headcount[i] = max(int(sized), int(baseline[i]), int(floor[i]))

    adjusted = ordered.copy()
    adjusted["base_calls"] = calls
    adjusted["calls"] = scaled_calls
    adjusted["weather_multiplier"] = np.round(multipliers, 4)
    adjusted["rain_mm"] = conditions.rain_mm
    adjusted["snow_cm"] = conditions.snow_cm
    adjusted["freeze_thaw"] = conditions.freeze_thaw
    adjusted["required_agents"] = headcount
    return adjusted


def weather_report(baseline: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, float]:
    """Summarise call volume and staffing impacts from weather scaling."""
    before_calls = baseline.sort_values("week_interval")["calls"].to_numpy(dtype=float)
    after_calls = adjusted.sort_values("week_interval")["calls"].to_numpy(dtype=float)
    before_agents = baseline.sort_values("week_interval")["required_agents"].to_numpy(dtype=int)
    after_agents = adjusted.sort_values("week_interval")["required_agents"].to_numpy(dtype=int)

    multipliers = (
        adjusted.get("weather_multiplier", pd.Series(np.ones(len(after_calls))))
        .to_numpy(dtype=float)
    )
    impacted = int(np.sum(multipliers > 1.0001))

    total_before = float(before_calls.sum())
    total_after = float(after_calls.sum())
    call_increase_pct = (
        ((total_after - total_before) / total_before * 100.0) if total_before > 0 else 0.0
    )

    return {
        "intervals_impacted": impacted,
        "max_weather_multiplier": round(float(multipliers.max()), 3),
        "baseline_total_calls": round(total_before, 1),
        "adjusted_total_calls": round(total_after, 1),
        "added_calls": round(total_after - total_before, 1),
        "call_increase_pct": round(call_increase_pct, 2),
        "baseline_agent_intervals": int(before_agents.sum()),
        "adjusted_agent_intervals": int(after_agents.sum()),
        "added_agent_intervals": int(after_agents.sum() - before_agents.sum()),
        "added_agent_hours": round(float(after_agents.sum() - before_agents.sum()) * 0.25, 2),
        "peak_before": int(before_agents.max()),
        "peak_after": int(after_agents.max()),
    }
