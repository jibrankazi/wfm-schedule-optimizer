"""Tests for the one-for-one shift-swap validator."""

import pytest

from src.swap_validator import ShiftAssignment, validate_shift_swap


def shift(agent, day, start=32, duration=34, hours=8.0, kind="FT"):
    return ShiftAssignment(agent, day, start, duration, hours, kind)


@pytest.fixture
def early_and_late():
    """A works 08:00-16:30 on Mon and Tue; B works 14:00-22:30 on the same days."""
    return (
        [shift("A", 0), shift("A", 1)],
        [shift("B", 0, start=56), shift("B", 1, start=56)],
    )


# --- accepted -------------------------------------------------------------
def test_a_clean_swap_is_accepted():
    a, b = shift("A", 0), shift("B", 2)
    result = validate_shift_swap([a], [b], a, b)
    assert result.is_valid
    assert result.reasons == []


# --- the four things a swap can break -------------------------------------
def test_turnaround_rest_is_enforced(early_and_late):
    """B's Monday shift ends at 22:30; taking A's Tuesday 08:00 gives 9.5h rest."""
    roster_a, roster_b = early_and_late
    result = validate_shift_swap(roster_a, roster_b, roster_a[1], roster_b[1])
    assert not result.is_valid
    assert any("Agent B violates turnaround rest" in r for r in result.reasons)
    assert any("9.50h provided, 10.0h required" in r for r in result.reasons)


def test_one_shift_per_day_is_enforced():
    """Taking a shift on a day the agent already works must be rejected."""
    a_mon, a_wed = shift("A", 0), shift("A", 2)
    b_wed = shift("B", 2, start=48)
    result = validate_shift_swap([a_mon, a_wed], [b_wed], a_mon, b_wed)
    assert not result.is_valid
    assert any("multiple shifts on day 2 (2 shifts)" in r for r in result.reasons)


def test_weekly_hour_ceiling_is_enforced():
    roster_a = [shift("A", day) for day in range(5)]  # 40.0h exactly
    longer = shift("B", 5, duration=42, hours=10.0)
    result = validate_shift_swap(roster_a, [longer], roster_a[0], longer)
    assert not result.is_valid
    assert any("Agent A would exceed weekly cap: 42.0h > 40.0h" in r for r in result.reasons)


def test_contract_partition_is_enforced():
    full_time = shift("A", 0)
    part_time = shift("B", 0, start=40, duration=16, hours=4.0, kind="PT")
    result = validate_shift_swap([full_time], [part_time], full_time, part_time)
    assert not result.is_valid
    assert any("Contract mismatch" in r for r in result.reasons)


# --- guards ---------------------------------------------------------------
def test_self_swap_is_rejected():
    first, second = shift("A", 0), shift("A", 2)
    result = validate_shift_swap([first, second], [first, second], first, second)
    assert not result.is_valid
    assert any("Self-swap rejected" in r for r in result.reasons)


def test_a_shift_the_agent_does_not_hold_is_rejected():
    a, b, stranger = shift("A", 0), shift("B", 2), shift("A", 4)
    result = validate_shift_swap([a], [b], stranger, b)
    assert not result.is_valid
    assert any("is not assigned to Agent A" in r for r in result.reasons)


# --- the invariant this module relies on ----------------------------------
def test_interval_coverage_is_invariant_under_a_swap():
    """Why there is no presence-floor check.

    A one-for-one trade exchanges who works two envelopes; both stay staffed.
    Interval headcount cannot change, so no swap can breach a floor.
    """
    a, b = shift("A", 0, start=32), shift("B", 1, start=56, duration=20)

    def covered(shifts):
        counts: dict[int, int] = {}
        for s in shifts:
            for t in range(s.global_start, s.global_end):
                counts[t] = counts.get(t, 0) + 1
        return counts

    before = covered([a, b])
    after = covered([
        ShiftAssignment("A", b.day, b.start_interval, b.duration_intervals, b.paid_hours, "FT"),
        ShiftAssignment("B", a.day, a.start_interval, a.duration_intervals, a.paid_hours, "FT"),
    ])
    assert before == after


# --- input validation -----------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"day": 7}, "day must be 0-6"),
        ({"start": 96}, "start_interval must be within the day"),
        ({"duration": 0}, "duration_intervals must be positive"),
        ({"hours": 0.0}, "paid_hours must be positive"),
        ({"kind": "CASUAL"}, "contract_kind must be"),
    ],
)
def test_malformed_assignments_are_rejected(kwargs, message):
    base = {"agent": "A", "day": 0, "start": 32, "duration": 34, "hours": 8.0, "kind": "FT"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        shift(**base)


def test_a_rest_violation_reports_the_configured_window():
    """The message quotes the window in force, not a hardcoded 10 hours."""
    roster_a = [shift("A", 0), shift("A", 1)]
    roster_b = [shift("B", 0, start=56), shift("B", 1, start=56)]
    result = validate_shift_swap(
        roster_a, roster_b, roster_a[1], roster_b[1], min_rest_intervals=48
    )
    assert any("12.0h required" in r for r in result.reasons)


def test_swap_checks_sunday_to_monday_rest():
    a = [shift("A", 0), shift("A", 6)]
    b = [shift("B", 6, start=56)]
    result = validate_shift_swap(a, b, a[1], b[0])
    assert not result.is_valid
    assert any("day 6 and day 0" in reason for reason in result.reasons)


def test_swap_rejects_a_roster_containing_another_agents_shifts():
    a, b = shift("A", 0), shift("B", 2)
    result = validate_shift_swap([a, shift("C", 3)], [b], a, b)
    assert not result.is_valid
    assert "another agent" in result.reasons[0]
