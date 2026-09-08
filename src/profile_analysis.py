"""Compare what different service contracts cost against identical demand.

The question this answers: hand four operations the same call arrivals and
the same handle times, and how differently must they staff?

The answer is not "a bit more for the strict one". Tightening service,
abandonment or occupancy targets can change the required integer headcount,
and the presence floor binds at the opposite end of the demand curve. A
single headline number therefore hides what is actually happening.
"""

from __future__ import annotations

import pandas as pd

from .erlang import required_agents, service_level, traffic_intensity
from .erlang_advanced import (
    calculate_erlang_a_metrics,
    erlang_a_service_level,
    required_agents_erlang_a,
)
from .profiles import PROFILES, ServiceProfile


def required_agents_for_profile(
    calls: float, aht: float, profile: ServiceProfile
) -> int:
    if not profile.abandons:
        return required_agents(calls, aht, profile.target)
    return required_agents_erlang_a(
        arrival_rate=calls,
        aht_sec=aht,
        mean_patience_sec=profile.mean_patience_sec,
        target_sl=profile.target_sl,
        target_sec=profile.target_time_sec,
        max_abandon_rate=profile.max_abandon_rate,
        max_occupancy=profile.max_occupancy,
        min_presence_floor=0,
    )


def profile_interval_metrics(
    agents: int,
    calls: float,
    aht: float,
    profile: ServiceProfile,
) -> tuple[float, float, float]:
    """Return service level, abandonment and occupancy for one interval."""
    intensity = traffic_intensity(calls, aht)
    if calls <= 0:
        return 1.0, 0.0, 0.0
    if agents <= 0:
        return 0.0, 1.0 if profile.abandons else 0.0, 1.0
    if not profile.abandons:
        return (
            service_level(agents, calls, aht, profile.target_time_sec),
            0.0,
            min(1.0, intensity / agents),
        )

    metrics = calculate_erlang_a_metrics(
        calls / 900.0,
        aht,
        profile.mean_patience_sec,
        agents,
    )
    achieved_sl = erlang_a_service_level(
        calls / 900.0,
        aht,
        profile.mean_patience_sec,
        agents,
        profile.target_time_sec,
    )
    return achieved_sl, metrics["abandonment_rate"], metrics["occupancy"]


def size_demand(demand: pd.DataFrame, profile: ServiceProfile) -> pd.DataFrame:
    """Re-size an existing demand frame under a different service contract.

    Adds the profile's presence floor, and records which of the three
    criteria - service level, occupancy cap, or floor - set each interval.
    """
    rows = []
    for row in demand.itertuples():
        calls = float(row.calls)
        aht = float(row.aht_sec)
        intensity = traffic_intensity(calls, aht)

        queueing_agents = required_agents_for_profile(calls, aht, profile)
        staffed = max(queueing_agents, profile.min_presence_floor)

        if staffed > queueing_agents:
            binding = "floor"
        elif queueing_agents == 0:
            binding = "no demand"
        else:
            previous_sl, previous_abandonment, previous_occupancy = profile_interval_metrics(
                queueing_agents - 1, calls, aht, profile
            )
            failures = []
            if previous_sl < profile.target_sl:
                failures.append("service level")
            if profile.abandons and previous_abandonment > profile.max_abandon_rate:
                failures.append("abandonment")
            if previous_occupancy > profile.max_occupancy:
                failures.append("occupancy")
            binding = failures[0] if len(failures) == 1 else "multiple"

        rows.append(
            {
                "day": row.day,
                "interval": row.interval,
                "calls": calls,
                "erlangs": round(intensity, 2),
                "required_agents": staffed,
                "queueing_agents": queueing_agents,
                "binding_constraint": binding,
            }
        )
    return pd.DataFrame(rows)


def compare_profiles(
    demand: pd.DataFrame,
    profiles: dict[str, ServiceProfile] | None = None,
) -> pd.DataFrame:
    """One row per profile: what the same week costs under each contract."""
    profiles = profiles or PROFILES
    baseline_key = "commercial" if "commercial" in profiles else next(iter(profiles))
    baseline_total: float | None = None
    rows = []

    for key, profile in profiles.items():
        sized = size_demand(demand, profile)
        total = int(sized.required_agents.sum())
        if key == baseline_key:
            baseline_total = float(total)

        binding = sized.binding_constraint.value_counts()
        rows.append(
            {
                "profile": profile.name,
                "target": profile.label(),
                "max_occupancy": profile.max_occupancy,
                "presence_floor": profile.min_presence_floor,
                "agent_intervals": total,
                "agent_hours": round(total * 0.25, 1),
                "peak_agents": int(sized.required_agents.max()),
                "set_by_service_level": int(binding.get("service level", 0)),
                "set_by_abandonment": int(binding.get("abandonment", 0)),
                "set_by_occupancy": int(binding.get("occupancy", 0)),
                "set_by_floor": int(binding.get("floor", 0)),
                "set_by_multiple": int(binding.get("multiple", 0)),
            }
        )

    frame = pd.DataFrame(rows)
    if baseline_total:
        frame["vs_commercial"] = (frame.agent_intervals / baseline_total).round(3)
    return frame


def marginal_cost_of_target(
    calls: float,
    aht_sec: float,
    profiles: dict[str, ServiceProfile] | None = None,
) -> pd.DataFrame:
    """Headcount for one interval under each contract, at a fixed offered load.

    Strips away the demand curve so the contract is the only thing varying.
    """
    profiles = profiles or PROFILES
    intensity = traffic_intensity(calls, aht_sec)
    rows = []
    for profile in profiles.values():
        agents = max(
            required_agents_for_profile(calls, aht_sec, profile),
            profile.min_presence_floor,
        )
        achieved_sl, abandonment, occupancy = profile_interval_metrics(
            agents, calls, aht_sec, profile
        )
        rows.append(
            {
                "profile": profile.name,
                "target": profile.label(),
                "erlangs": round(intensity, 2),
                "agents": agents,
                "occupancy": round(occupancy, 3),
                "achieved_sl": round(achieved_sl, 4),
                "abandonment_rate": round(abandonment, 4),
                "agents_per_erlang": round(agents / intensity, 3) if intensity else 0.0,
            }
        )
    return pd.DataFrame(rows)
