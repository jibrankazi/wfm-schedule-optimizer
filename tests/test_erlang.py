import math

import pytest

from src.erlang import (
    ServiceTarget,
    average_speed_of_answer,
    erlang_b,
    erlang_c,
    required_agents,
    service_level,
    traffic_intensity,
)


def factorial_erlang_c(agents: int, intensity: float) -> float:
    """The textbook formula, for cross-checking the recursion at safe loads."""
    if agents <= intensity:
        return 1.0
    partial = sum((intensity**k) / math.factorial(k) for k in range(agents))
    tail = ((intensity**agents) / math.factorial(agents)) * (agents / (agents - intensity))
    return tail / (partial + tail)


# --- traffic ---------------------------------------------------------------
def test_traffic_intensity_is_calls_times_aht_over_the_interval():
    # 45 calls of 320s in a 900s interval = 16 Erlangs of offered load
    assert traffic_intensity(45, 320) == pytest.approx(16.0)


def test_zero_demand_is_zero_load():
    assert traffic_intensity(0, 320) == 0.0
    assert required_agents(0, 320) == 0


# --- the recursion agrees with the textbook formula ------------------------
@pytest.mark.parametrize("calls,aht", [(10, 320), (30, 320), (50, 320), (70, 400), (100, 250)])
def test_recursion_matches_factorial_form(calls, aht):
    intensity = traffic_intensity(calls, aht)
    agents = math.ceil(intensity) + 3
    assert erlang_c(agents, intensity) == pytest.approx(factorial_erlang_c(agents, intensity), abs=1e-12)


def test_factorial_form_overflows_where_the_recursion_does_not():
    """The reason this module does not use the textbook formula."""
    intensity = traffic_intensity(400, 320)  # ~142 Erlangs
    agents = math.ceil(intensity) + 10
    with pytest.raises(OverflowError):
        factorial_erlang_c(agents, intensity)
    assert 0.0 < erlang_c(agents, intensity) < 1.0


# --- boundary behaviour ----------------------------------------------------
def test_every_call_queues_when_capacity_equals_load():
    assert erlang_c(16, 16.0) == 1.0
    assert service_level(16, 45, 320) == 0.0


def test_queue_probability_falls_as_agents_are_added():
    intensity = 16.0
    probabilities = [erlang_c(n, intensity) for n in range(17, 30)]
    assert all(later < earlier for earlier, later in zip(probabilities, probabilities[1:]))


def test_erlang_b_is_between_zero_and_one():
    assert 0.0 < erlang_b(20, 16.0) < 1.0


# --- required headcount ----------------------------------------------------
def test_required_headcount_meets_both_criteria():
    target = ServiceTarget(service_level=0.80, target_time_sec=20.0, max_occupancy=0.85)
    for calls in (5, 20, 45, 70, 110):
        agents = required_agents(calls, 320, target)
        intensity = traffic_intensity(calls, 320)
        assert service_level(agents, calls, 320, target.target_time_sec) >= target.service_level
        assert intensity / agents <= target.max_occupancy


def test_one_fewer_agent_would_breach_the_target():
    """Confirms the search returns the minimum, not merely a sufficient number."""
    target = ServiceTarget()
    for calls in (20, 45, 70):
        agents = required_agents(calls, 320, target)
        intensity = traffic_intensity(calls, 320)
        breaches_sl = service_level(agents - 1, calls, 320) < target.service_level
        breaches_occupancy = (intensity / (agents - 1)) > target.max_occupancy
        assert breaches_sl or breaches_occupancy


def test_headcount_rises_with_volume():
    counts = [required_agents(c, 320) for c in (10, 30, 60, 100)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_a_tighter_target_needs_more_agents():
    relaxed = required_agents(50, 320, ServiceTarget(service_level=0.70, target_time_sec=30))
    strict = required_agents(50, 320, ServiceTarget(service_level=0.90, target_time_sec=10))
    assert strict > relaxed


def test_occupancy_cap_binds_only_at_high_load():
    """Which constraint bites depends on queue size, and both are needed.

    Small queues hit the 80/20 service level well below 85% occupancy, so the
    cap is slack. Large queues enjoy economies of scale in queueing and meet
    the service level while still running agents into the ground, so the cap
    becomes the binding constraint. Dropping it would produce staffing numbers
    that satisfy the SLA and burn the floor out.
    """
    # 24.9 Erlangs: service level binds, occupancy cap is slack
    assert required_agents(70, 320, ServiceTarget(max_occupancy=0.99)) == required_agents(
        70, 320, ServiceTarget(max_occupancy=0.85)
    )

    # 53.3 Erlangs: the cap now forces extra headcount
    uncapped = required_agents(150, 320, ServiceTarget(max_occupancy=0.99))
    capped = required_agents(150, 320, ServiceTarget(max_occupancy=0.85))
    assert capped > uncapped
    assert traffic_intensity(150, 320) / uncapped > 0.85


def test_impossible_target_raises_rather_than_looping():
    with pytest.raises(ValueError, match="No headcount"):
        required_agents(500, 600, ServiceTarget(), ceiling=10)


# --- reporting -------------------------------------------------------------
def test_asa_is_finite_when_staffed_and_infinite_when_not():
    assert average_speed_of_answer(25, 45, 320) < 60
    assert average_speed_of_answer(10, 45, 320) == float("inf")
