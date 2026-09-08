import numpy as np
import pytest

from src.benchmark import compare, solve_greedy, summarise_gap
from src.generator import DAYS, INTERVALS_PER_DAY, build_shift_templates, generate_demand, generate_roster


@pytest.fixture(scope="module")
def inputs():
    return generate_roster(seed=42), build_shift_templates(step=4), generate_demand(seed=42)


@pytest.fixture(scope="module")
def greedy(inputs):
    agents, templates, demand = inputs
    return solve_greedy(agents, templates, demand)


# --- the baseline must obey every constraint the MILP obeys ----------------
def test_greedy_never_double_books_a_day(greedy):
    seen = set()
    for assignment in greedy.assignments:
        key = (assignment.agent_id, assignment.day)
        assert key not in seen
        seen.add(key)


def test_greedy_respects_the_weekly_ceiling(greedy, inputs):
    agents, _, _ = inputs
    by_id = {a.id: a for a in agents}
    for agent_id, worked in greedy.agent_hours.items():
        assert worked <= by_id[agent_id].max_weekly_hours + 1e-9


def test_greedy_never_schedules_over_booked_leave(greedy, inputs):
    agents, _, _ = inputs
    by_id = {a.id: a for a in agents}
    for assignment in greedy.assignments:
        blocked = by_id[assignment.agent_id].blackouts.get(assignment.day, set())
        assert not any(assignment.template.occupies(i) for i in blocked)


def test_greedy_honours_start_window_accommodations(greedy, inputs):
    agents, _, _ = inputs
    by_id = {a.id: a for a in agents}
    for assignment in greedy.assignments:
        agent = by_id[assignment.agent_id]
        assert agent.earliest_start <= assignment.template.start <= agent.latest_start


def test_greedy_does_not_give_part_timers_full_time_shifts(greedy, inputs):
    agents, _, _ = inputs
    by_id = {a.id: a for a in agents}
    for assignment in greedy.assignments:
        if by_id[assignment.agent_id].kind == "part_time":
            assert assignment.template.kind == "part_time"


def test_greedy_coverage_matches_its_own_assignments(greedy):
    rebuilt = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    for assignment in greedy.assignments:
        rebuilt[assignment.day] += np.array(assignment.template.coverage, dtype=int)
    assert np.array_equal(rebuilt, greedy.coverage)


def test_greedy_is_deterministic(inputs):
    agents, templates, demand = inputs
    first = solve_greedy(agents, templates, demand)
    second = solve_greedy(agents, templates, demand)
    assert np.array_equal(first.coverage, second.coverage)
    assert len(first.assignments) == len(second.assignments)


def test_greedy_is_fast(greedy):
    """The whole point of a heuristic: it answers immediately."""
    assert greedy.solve_seconds < 5.0


def test_min_hours_pass_can_be_disabled(inputs):
    agents, templates, demand = inputs
    without = solve_greedy(agents, templates, demand, fill_contracted_minimum=False)
    with_fill = solve_greedy(agents, templates, demand, fill_contracted_minimum=True)
    assert sum(with_fill.agent_hours.values()) >= sum(without.agent_hours.values())


# --- the comparison --------------------------------------------------------
def test_milp_covers_at_least_as_well_as_the_heuristic(inputs):
    agents, templates, demand = inputs
    table = compare(agents, templates, demand, time_limit_sec=90, gap=0.05)
    gap = summarise_gap(table)
    assert gap["milp_understaffed"] <= gap["greedy_understaffed"]
    assert gap["understaffing_reduction_pct"] >= 0


def test_the_heuristic_is_the_faster_of_the_two(inputs):
    agents, templates, demand = inputs
    table = compare(agents, templates, demand, time_limit_sec=90, gap=0.05)
    gap = summarise_gap(table)
    assert gap["greedy_seconds"] < gap["milp_seconds"]


def test_comparison_table_shape(inputs):
    agents, templates, demand = inputs
    table = compare(agents, templates, demand, time_limit_sec=60, gap=0.05)
    assert list(table.scheduler) == ["Greedy heuristic", "MILP (CBC)"]
    assert (table.scheduled_hours > 0).all()


def test_summarise_gap_handles_a_perfect_baseline():
    import pandas as pd

    perfect = pd.DataFrame(
        [
            {"scheduler": "Greedy heuristic", "solve_seconds": 0.1, "shifts": 1,
             "scheduled_hours": 8.0, "understaffed_agent_intervals": 0,
             "overstaffed_agent_intervals": 0, "coverage_compliance_pct": 100.0,
             "agents_below_contracted_minimum": 0},
            {"scheduler": "MILP (CBC)", "solve_seconds": 5.0, "shifts": 1,
             "scheduled_hours": 8.0, "understaffed_agent_intervals": 0,
             "overstaffed_agent_intervals": 0, "coverage_compliance_pct": 100.0,
             "agents_below_contracted_minimum": 0},
        ]
    )
    gap = summarise_gap(perfect)
    assert gap["understaffing_reduction_pct"] == 0.0
    assert gap["overstaffing_reduction_pct"] == 0.0
