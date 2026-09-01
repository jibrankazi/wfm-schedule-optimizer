"""Erlang C staffing model.

Answers one question per interval: how many agents must be on the phone
concurrently to answer `target_sl` of calls within `target_time_sec`, without
pushing agent occupancy past `max_occupancy`?

Implementation note. The textbook Erlang C formula is written with factorials:

    P(wait) = [A^N/N! * N/(N-A)] / [sum(A^k/k!, k<N) + A^N/N! * N/(N-A)]

Evaluated literally in floating point that raises OverflowError once traffic
intensity passes roughly 135 Erlangs, because A^N overflows a float long
before the ratio does. This module uses the Erlang B recursion instead, which
never forms A^N at all:

    B(0, A) = 1
    B(n, A) = A * B(n-1, A) / (n + A * B(n-1, A))
    C(N, A) = B / (1 - rho * (1 - B))            where rho = A / N

Same answer to machine precision, O(N) time, no overflow at any load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

INTERVAL_SECONDS = 900  # 15 minutes


def traffic_intensity(calls: float, aht_sec: float, interval_sec: int = INTERVAL_SECONDS) -> float:
    """Offered load in Erlangs: the agent-hours of work arriving per interval."""
    if calls <= 0 or aht_sec <= 0:
        return 0.0
    return (calls * aht_sec) / interval_sec


def erlang_b(agents: int, intensity: float) -> float:
    """Blocking probability, via the recursion that avoids A^N."""
    if intensity <= 0:
        return 0.0
    inv = 1.0
    for n in range(1, agents + 1):
        inv = 1.0 + inv * n / intensity
    return 1.0 / inv


def erlang_c(agents: int, intensity: float) -> float:
    """Probability an arriving call has to queue."""
    if agents <= 0:
        return 1.0
    if intensity <= 0:
        return 0.0
    if agents <= intensity:
        return 1.0  # offered load exceeds capacity; every call queues
    blocking = erlang_b(agents, intensity)
    utilisation = intensity / agents
    return blocking / (1.0 - utilisation * (1.0 - blocking))


def service_level(
    agents: int,
    calls: float,
    aht_sec: float,
    target_time_sec: float = 20.0,
    interval_sec: int = INTERVAL_SECONDS,
) -> float:
    """Fraction of calls answered within `target_time_sec`."""
    intensity = traffic_intensity(calls, aht_sec, interval_sec)
    if intensity <= 0:
        return 1.0
    if agents <= intensity:
        return 0.0
    queued = erlang_c(agents, intensity)
    return 1.0 - queued * math.exp(-(agents - intensity) * (target_time_sec / aht_sec))


def average_speed_of_answer(agents: int, calls: float, aht_sec: float) -> float:
    """Mean queue time in seconds, for reporting."""
    intensity = traffic_intensity(calls, aht_sec)
    if intensity <= 0 or agents <= intensity:
        return float("inf")
    return erlang_c(agents, intensity) * aht_sec / (agents - intensity)


@dataclass(frozen=True)
class ServiceTarget:
    """The staffing contract to hit in every interval."""

    service_level: float = 0.80  # answer 80% of calls...
    target_time_sec: float = 20.0  # ...within 20 seconds
    max_occupancy: float = 0.85  # and keep agents below 85% utilised

    def label(self) -> str:
        return f"{self.service_level:.0%}/{self.target_time_sec:.0f}s"


def required_agents(
    calls: float,
    aht_sec: float,
    target: ServiceTarget | None = None,
    ceiling: int = 500,
) -> int:
    """Smallest headcount meeting both the service level and the occupancy cap.

    Both criteria improve monotonically as headcount rises, so the first
    N that satisfies them is the minimum. `ceiling` is a guard against a
    runaway loop on absurd inputs rather than a modelling choice.
    """
    target = target or ServiceTarget()
    intensity = traffic_intensity(calls, aht_sec)
    if intensity <= 0:
        return 0

    agents = max(1, math.ceil(intensity))
    while agents <= ceiling:
        meets_sl = service_level(agents, calls, aht_sec, target.target_time_sec) >= target.service_level
        meets_occupancy = (intensity / agents) <= target.max_occupancy
        if meets_sl and meets_occupancy:
            return agents
        agents += 1

    raise ValueError(
        f"No headcount up to {ceiling} satisfies {target.label()} "
        f"for {calls:.0f} calls at {aht_sec:.0f}s AHT ({intensity:.1f} Erlangs)."
    )
