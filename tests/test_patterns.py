"""Tests for the operating-pattern configuration layer.

The claim this layer makes is that a contact centre is data, not code: the
same solver runs a weekday desk, a seven-day extended one and a continuous
24/7 operation with nothing changing but the configuration. These check that
the configuration actually controls what it says it controls, and that a
pattern cannot be defined in a way that quietly under-sizes itself - which is
how the original 65-agent 24/7 instance came about.
"""

from __future__ import annotations

import pytest

from src.patterns import (
    CONTINUOUS_247,
    PATTERNS,
    SEVEN_DAY_EXTENDED,
    WEEKDAY_EXTENDED,
    OperatingPattern,
    build_pattern_demand,
    build_pattern_options,
    pattern_roster,
    sizing_report,
)


# --- the configurations themselves ----------------------------------------
def test_all_three_patterns_are_registered():
    assert set(PATTERNS) == {"weekday_extended", "continuous_247", "seven_day_extended"}


@pytest.mark.parametrize("pattern", PATTERNS.values(), ids=list(PATTERNS))
def test_every_pattern_carries_enough_roster_to_meet_its_own_demand(pattern):
    """The check the original 24/7 instance would have failed.

    Sizing a roster below what its demand requires, then reducing the demand
    until it fits, is a modelling error rather than a solver result. A pattern
    whose theoretical utilisation exceeds 1.0 cannot be covered by anyone.
    """
    report = sizing_report(pattern)
    assert report["theoretical_utilisation"] < 1.0, report


@pytest.mark.parametrize("pattern", PATTERNS.values(), ids=list(PATTERNS))
def test_the_grid_is_always_a_full_day(pattern):
    """Opening hours live in demand and the action space, never in the grid.

    A shift ending at 19:00 and one ending at midnight have to be comparable,
    which they are not if the grid itself is truncated.
    """
    assert pattern.grid.intervals_per_day == 96
    assert pattern.grid.total_intervals == pattern.days * 96


def test_a_pattern_with_an_impossible_window_is_rejected():
    with pytest.raises(ValueError, match="opening window"):
        OperatingPattern(
            key="bad", label="", days=5, open_from=76, open_to=28,
            presence_floor=0, peak_calls=110.0, full_time=10, part_time=0,
        )


def test_a_pattern_with_an_empty_roster_is_rejected():
    with pytest.raises(ValueError, match="roster cannot be empty"):
        OperatingPattern(
            key="bad", label="", days=5, open_from=28, open_to=76,
            presence_floor=0, peak_calls=110.0, full_time=0, part_time=0,
        )


# --- demand follows the window --------------------------------------------
def test_a_closed_centre_has_no_demand_outside_its_hours():
    demand = build_pattern_demand(WEEKDAY_EXTENDED)
    closed = demand[~demand.is_open]
    assert not closed.empty
    assert (closed.calls == 0).all()
    assert (closed.required_agents == 0).all()


def test_a_continuous_centre_is_never_closed_and_never_unstaffed():
    demand = build_pattern_demand(CONTINUOUS_247)
    assert demand.is_open.all()
    assert (demand.required_agents >= CONTINUOUS_247.presence_floor).all()


def test_the_floor_only_applies_where_the_centre_is_open():
    """A closed centre needs nobody present, however high its floor."""
    pattern = OperatingPattern(
        key="floored", label="", days=2, open_from=28, open_to=76,
        presence_floor=5, peak_calls=40.0, full_time=20, part_time=5,
    )
    demand = build_pattern_demand(pattern)
    assert (demand[~demand.is_open].required_agents == 0).all()
    assert (demand[demand.is_open].required_agents >= 5).all()


def test_weekend_demand_is_lighter_where_the_pattern_says_so():
    demand = build_pattern_demand(SEVEN_DAY_EXTENDED)
    weekday_peak = demand[demand.day < 5].calls.max()
    weekend_peak = demand[demand.day >= 5].calls.max()
    assert weekend_peak < weekday_peak


# --- the action space follows the window ----------------------------------
def test_a_closed_centre_offers_no_shift_running_past_closing():
    """This is where the model actually shrinks, not in the grid."""
    options = build_pattern_options(WEEKDAY_EXTENDED)
    assert options
    for option in options:
        for slot in option.coverage_slots:
            assert WEEKDAY_EXTENDED.open_from <= slot % 96 < WEEKDAY_EXTENDED.open_to


def test_a_continuous_centre_keeps_overnight_shifts():
    options = build_pattern_options(CONTINUOUS_247)
    assert any(any(s % 96 < 24 for s in o.coverage_slots) for o in options)


def test_coarsening_the_start_grid_shrinks_the_action_space():
    """The fix that made the 150-agent instance buildable.

    At the full start grid a 150-agent continuous roster produces roughly
    126,000 binaries and exhausts memory during model construction. Keeping
    every fourth start is both closer to how a continuous desk is actually
    run and the difference between a model that builds and one that does not.
    """
    from dataclasses import replace

    fine = build_pattern_options(replace(CONTINUOUS_247, start_stride=1))
    coarse = build_pattern_options(CONTINUOUS_247)
    assert len(coarse) < len(fine)
    assert CONTINUOUS_247.start_stride > 1


# --- the roster the pattern asks for --------------------------------------
@pytest.mark.parametrize("pattern", PATTERNS.values(), ids=list(PATTERNS))
def test_the_roster_matches_the_declared_size(pattern):
    agents = pattern_roster(pattern)
    assert len(agents) == pattern.roster_size == pattern.full_time + pattern.part_time


def test_the_three_patterns_are_sized_differently():
    """A single roster figure reused across operating patterns is the bug
    this layer exists to prevent."""
    sizes = {p.key: p.roster_size for p in PATTERNS.values()}
    assert len(set(sizes.values())) == 3, sizes
    assert sizes["continuous_247"] > sizes["seven_day_extended"] > sizes["weekday_extended"]
