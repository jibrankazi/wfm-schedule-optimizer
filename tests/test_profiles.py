import pandas as pd
import pytest

from src.erlang import required_agents, service_level, traffic_intensity
from src.profile_analysis import (
    compare_profiles,
    marginal_cost_of_target,
    profile_interval_metrics,
    size_demand,
)
from src.profiles import (
    COMMERCIAL_INBOUND,
    PROFILES,
    RAPID_RESPONSE,
    DutyCodes,
    ServiceProfile,
    get_profile,
)


# --- duty codes ------------------------------------------------------------
def test_primary_on_queue_is_explicit_not_derived():
    """Iteration order over a frozenset is not stable between processes.

    Deriving the primary code with next(iter(on_queue)) would mark a drafted
    agent OVERTIME on one run and QUEUE on the next.
    """
    codes = DutyCodes()
    assert codes.primary_on_queue == "QUEUE"
    assert codes.primary_on_queue in codes.on_queue


def test_primary_must_be_a_member_of_on_queue():
    with pytest.raises(ValueError, match="must be one of"):
        DutyCodes(primary_on_queue="PHONE", on_queue=frozenset({"QUEUE"}))


def test_a_code_cannot_be_both_on_queue_and_preemptable():
    with pytest.raises(ValueError, match="both on-queue and preemptable"):
        DutyCodes(on_queue=frozenset({"QUEUE"}), preemptable=("QUEUE", "TRAINING"))


def test_preemptable_must_not_repeat():
    with pytest.raises(ValueError, match="must not repeat"):
        DutyCodes(preemptable=("QUALITY", "QUALITY"))


def test_codes_normalise_whitespace_and_case():
    codes = DutyCodes()
    assert codes.is_on_queue("  queue ")
    assert not codes.is_on_queue("quality")


def test_custom_taxonomy_is_accepted():
    codes = DutyCodes(
        primary_on_queue="CALL_TAKER",
        on_queue=frozenset({"CALL_TAKER", "RADIO"}),
        preemptable=("ADMIN", "TRAINING"),
    )
    assert codes.is_on_queue("radio")
    assert "ADMIN" in codes.all_codes()


# --- service profiles ------------------------------------------------------
def test_all_shipped_profiles_are_valid():
    assert set(PROFILES) == {"commercial", "complex", "rapid", "blended"}
    for profile in PROFILES.values():
        assert 0 < profile.target_sl <= 1
        assert 0 < profile.max_occupancy <= 1
        assert profile.min_presence_floor >= 0


def test_profile_exposes_an_erlang_target():
    target = COMMERCIAL_INBOUND.target
    assert target.service_level == 0.80
    assert target.target_time_sec == 20.0
    assert target.max_occupancy == 0.85


def test_urgent_intake_does_not_model_abandonment():
    """Callers with an emergency hold and redial; they do not hang up.

    Modelling them with an abandonment term would understate capacity.
    """
    assert RAPID_RESPONSE.abandons is False
    assert COMMERCIAL_INBOUND.abandons is True


def test_abandoning_profile_needs_positive_patience():
    with pytest.raises(ValueError, match="positive mean_patience_sec"):
        ServiceProfile(
            name="bad", target_sl=0.8, target_time_sec=20, max_occupancy=0.85,
            abandons=True, mean_patience_sec=0.0,
        )


def test_invalid_targets_are_rejected():
    with pytest.raises(ValueError, match="target_sl"):
        ServiceProfile(name="x", target_sl=1.5, target_time_sec=20, max_occupancy=0.85)
    with pytest.raises(ValueError, match="max_occupancy"):
        ServiceProfile(name="x", target_sl=0.8, target_time_sec=20, max_occupancy=0.0)


def test_get_profile_and_its_error():
    assert get_profile("RAPID ") is RAPID_RESPONSE
    with pytest.raises(KeyError, match="unknown profile"):
        get_profile("fire-department")


def test_with_codes_returns_a_new_profile():
    codes = DutyCodes(primary_on_queue="RADIO", on_queue=frozenset({"RADIO"}))
    swapped = COMMERCIAL_INBOUND.with_codes(codes)
    assert swapped.duty_codes.primary_on_queue == "RADIO"
    assert COMMERCIAL_INBOUND.duty_codes.primary_on_queue == "QUEUE"


# --- staffing economics ----------------------------------------------------
@pytest.fixture(scope="module")
def demand():
    from src.generator import generate_demand
    return generate_demand(seed=42)


def test_a_stricter_target_costs_more_against_identical_demand(demand):
    frame = compare_profiles(demand).set_index("profile")
    commercial = frame.loc["Commercial inbound", "agent_intervals"]
    rapid = frame.loc["Rapid response intake", "agent_intervals"]
    blended = frame.loc["Blended back office", "agent_intervals"]
    assert rapid > commercial > blended


def test_comparison_reports_which_criterion_set_each_interval(demand):
    frame = compare_profiles(demand).set_index("profile")
    row = frame.loc["Complex support"]
    # A 0.75 occupancy cap binds frequently under this synthetic workload.
    assert row.set_by_occupancy > row.set_by_service_level
    commercial = frame.loc["Commercial inbound"]
    # Finite patience now influences the result rather than being metadata.
    assert commercial.set_by_abandonment + commercial.set_by_multiple > 0


def test_sized_headcount_never_falls_below_the_presence_floor(demand):
    sized = size_demand(demand, RAPID_RESPONSE)
    assert (sized.required_agents >= RAPID_RESPONSE.min_presence_floor).all()


def test_floor_binds_when_demand_is_negligible():
    quiet = pd.DataFrame(
        [{"day": 0, "interval": i, "calls": 0.0, "aht_sec": 320.0} for i in range(4)]
    )
    sized = size_demand(quiet, RAPID_RESPONSE)
    assert (sized.required_agents == RAPID_RESPONSE.min_presence_floor).all()
    assert (sized.binding_constraint == "floor").all()


def test_every_sized_interval_meets_its_own_target(demand):
    for profile in PROFILES.values():
        sized = size_demand(demand.head(40), profile)
        for row in sized.itertuples():
            if row.calls <= 0:
                continue
            achieved, abandonment, occupancy = profile_interval_metrics(
                int(row.required_agents),
                row.calls,
                float(demand.loc[row.Index, "aht_sec"]),
                profile,
            )
            assert achieved >= profile.target_sl - 1e-9
            assert occupancy <= profile.max_occupancy + 1e-6
            if profile.abandons:
                assert abandonment <= profile.max_abandon_rate + 1e-6


def test_queues_get_cheaper_per_erlang_as_they_grow():
    """Economies of scale in queueing: bigger queues need less relative slack."""
    small = marginal_cost_of_target(10, 320).set_index("profile")
    large = marginal_cost_of_target(240, 320).set_index("profile")
    for name in small.index:
        assert large.loc[name, "agents_per_erlang"] < small.loc[name, "agents_per_erlang"]
