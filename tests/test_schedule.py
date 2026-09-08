import numpy as np
import pulp
import pytest

from src.generator import (
    DAYS,
    INTERVALS_PER_DAY,
    SHIFT_SHAPES,
    Agent,
    build_shift_templates,
    capacity_report,
    generate_demand,
    generate_roster,
    interval_label,
)
from src.optimizer import coverage_lower_bound, hours_frame, solve_schedule


@pytest.fixture(scope="module")
def demand():
    return generate_demand(seed=42)


@pytest.fixture(scope="module")
def agents():
    return generate_roster(seed=42)


@pytest.fixture(scope="module")
def templates():
    return build_shift_templates(step=2)


@pytest.fixture(scope="module")
def schedule(agents, templates, demand):
    # Coarser templates and a looser gap keep the suite quick; the quality
    # assertions below are deliberately written to hold at either setting.
    return solve_schedule(agents, build_shift_templates(step=4), demand, time_limit_sec=90, gap=0.05)


# --- time grid -------------------------------------------------------------
def test_grid_runs_0700_to_1915():
    assert interval_label(0) == "07:00"
    assert interval_label(INTERVALS_PER_DAY - 1) == "19:00"
    assert INTERVALS_PER_DAY == 49


# --- demand ----------------------------------------------------------------
def test_demand_is_reproducible():
    assert generate_demand(seed=7).equals(generate_demand(seed=7))


def test_different_seeds_give_different_weeks():
    assert not generate_demand(seed=1).equals(generate_demand(seed=2))


def test_demand_covers_every_interval(demand):
    assert len(demand) == DAYS * INTERVALS_PER_DAY
    assert set(demand.day) == set(range(DAYS))


def test_demand_is_bimodal_and_peaks_late_morning(demand):
    monday = demand[demand.day == 0].reset_index(drop=True)
    peak = monday.required_agents.idxmax()
    assert 10 <= peak <= 22  # roughly 09:30 to 12:30


def test_monday_is_the_heaviest_day(demand):
    totals = demand.groupby("day").required_agents.sum()
    assert totals.idxmax() == 0
    assert totals.idxmin() == DAYS - 1


def test_aht_stays_in_a_plausible_band(demand):
    assert demand.aht_sec.between(180, 600).all()


# --- shift templates -------------------------------------------------------
def test_templates_fit_inside_the_day(templates):
    assert templates
    for template in templates:
        assert template.start >= 0
        assert template.end <= INTERVALS_PER_DAY


def test_paid_hours_exclude_the_unpaid_meal(templates):
    for template in templates:
        _hours, span, rests, meals = SHIFT_SHAPES[template.id.split("_")[0]]
        assert template.paid_hours == pytest.approx((span - len(meals)) * 0.25)
        # rest breaks are paid but off-phone, so coverage is shorter still
        assert template.covered_intervals == span - len(rests) - len(meals)


def test_coverage_is_contiguous_apart_from_breaks(templates):
    """No template may staff 07:00, vanish, and reappear at 16:45."""
    for template in templates:
        on = [i for i, flag in enumerate(template.coverage) if flag]
        gaps = [b - a - 1 for a, b in zip(on, on[1:]) if b - a > 1]
        assert sum(gaps) == len(template.breaks)


def test_breaks_fall_inside_the_shift(templates):
    for template in templates:
        for interval in template.breaks:
            assert template.start <= interval < template.end
            assert template.coverage[interval] == 0


def test_finer_granularity_produces_more_templates():
    assert len(build_shift_templates(step=2)) > len(build_shift_templates(step=4))


# --- roster ----------------------------------------------------------------
def test_roster_shape(agents):
    assert len(agents) == 60
    assert sum(1 for a in agents if a.kind == "full_time") == 45
    assert sum(1 for a in agents if a.kind == "part_time") == 15


def test_roster_is_reproducible():
    first = generate_roster(seed=11)
    second = generate_roster(seed=11)
    assert [a.max_weekly_hours for a in first] == [a.max_weekly_hours for a in second]
    assert [a.blackouts for a in first] == [a.blackouts for a in second]


def test_minimum_never_exceeds_maximum(agents):
    assert all(a.min_weekly_hours <= a.max_weekly_hours for a in agents)


def test_part_time_agents_are_refused_full_time_templates(agents, templates):
    part_timer = next(a for a in agents if a.kind == "part_time")
    full_shift = next(t for t in templates if t.kind == "full_time")
    assert not part_timer.can_work(0, full_shift)


def test_blackout_blocks_any_overlapping_template(templates):
    agent = Agent(id=0, name="T", kind="full_time", min_weekly_hours=0, max_weekly_hours=40)
    agent.blackouts[2] = {20, 21, 22}
    overlapping = [t for t in templates if t.occupies(21)]
    assert overlapping
    assert all(not agent.can_work(2, t) for t in overlapping)
    assert any(agent.can_work(2, t) for t in templates if not t.occupies(21))


def test_start_window_accommodation_is_respected(templates):
    agent = Agent(
        id=0, name="T", kind="full_time", min_weekly_hours=0, max_weekly_hours=40, latest_start=8
    )
    assert all(t.start <= 8 for t in templates if agent.can_work(0, t))


def test_blackout_on_one_day_does_not_affect_another(templates):
    agent = Agent(id=0, name="T", kind="full_time", min_weekly_hours=0, max_weekly_hours=40)
    agent.blackouts[1] = set(range(0, INTERVALS_PER_DAY))
    assert not any(agent.can_work(1, t) for t in templates)
    assert any(agent.can_work(0, t) for t in templates)


# --- capacity diagnostics --------------------------------------------------
def test_capacity_report_flags_a_roster_that_cannot_meet_demand(agents, templates, demand):
    report = capacity_report(agents, templates, demand)
    assert not report["demand_exceeds_ceiling"]
    assert 0 < report["utilisation_of_ceiling"] < 1

    tiny = agents[:5]
    assert capacity_report(tiny, templates, demand)["demand_exceeds_ceiling"]


# --- the solved schedule ---------------------------------------------------
def test_solver_returns_a_schedule(schedule):
    assert schedule.status == "Optimal"
    assert schedule.assignments


def test_solver_refuses_a_not_solved_allocation(monkeypatch, agents, demand):
    def leave_unsolved(model, solver):
        model.status = pulp.LpStatusNotSolved
        return model.status

    monkeypatch.setattr(pulp.LpProblem, "solve", leave_unsolved)
    with pytest.raises(RuntimeError, match="refusing to publish"):
        solve_schedule(
            agents[:2],
            build_shift_templates(step=8),
            demand,
            time_limit_sec=1,
        )


def test_no_agent_works_two_shifts_in_one_day(schedule):
    seen = set()
    for assignment in schedule.assignments:
        key = (assignment.agent_id, assignment.day)
        assert key not in seen, f"{assignment.agent_name} double-booked on day {assignment.day}"
        seen.add(key)


def test_weekly_maximum_is_never_breached(schedule, agents):
    by_id = {a.id: a for a in agents}
    for agent_id, hours in schedule.agent_hours.items():
        assert hours <= by_id[agent_id].max_weekly_hours + 1e-6


def test_no_agent_is_scheduled_during_a_blackout(schedule, agents):
    by_id = {a.id: a for a in agents}
    for assignment in schedule.assignments:
        agent = by_id[assignment.agent_id]
        blocked = agent.blackouts.get(assignment.day, set())
        assert not any(assignment.template.occupies(i) for i in blocked)


def test_accommodated_start_windows_are_honoured(schedule, agents):
    by_id = {a.id: a for a in agents}
    accommodated = [a for a in agents if a.accommodation]
    assert accommodated, "fixture should contain at least one accommodation"
    for assignment in schedule.assignments:
        agent = by_id[assignment.agent_id]
        assert agent.earliest_start <= assignment.template.start <= agent.latest_start


def test_part_time_agents_only_get_part_time_shifts(schedule, agents):
    by_id = {a.id: a for a in agents}
    for assignment in schedule.assignments:
        if by_id[assignment.agent_id].kind == "part_time":
            assert assignment.template.kind == "part_time"


def test_reported_coverage_matches_the_assignments(schedule):
    rebuilt = np.zeros((DAYS, INTERVALS_PER_DAY), dtype=int)
    for assignment in schedule.assignments:
        rebuilt[assignment.day] += np.array(assignment.template.coverage, dtype=int)
    assert np.array_equal(rebuilt, schedule.coverage)


def test_the_schedule_actually_covers_most_of_the_curve(schedule):
    assert schedule.summary()["coverage_compliance_pct"] > 80


def test_understaffing_is_never_negative(schedule):
    assert (schedule.understaffed >= 0).all()
    assert (schedule.overstaffed >= 0).all()


def test_frames_line_up_with_the_schedule(schedule, agents):
    assert len(schedule.to_frame()) == len(schedule.assignments)
    assert len(schedule.coverage_frame()) == DAYS * INTERVALS_PER_DAY
    assert len(hours_frame(schedule, agents)) == len(agents)


# --- the relaxation bound --------------------------------------------------
def test_lower_bound_is_a_bound(agents, demand):
    coarse = build_shift_templates(step=4)
    bound = coverage_lower_bound(agents, coarse, demand)
    solved = solve_schedule(agents, coarse, demand, time_limit_sec=90, gap=0.05)
    assert (
        bound["min_understaffed_agent_intervals"]
        <= solved.summary()["understaffed_agent_intervals"]
    )
