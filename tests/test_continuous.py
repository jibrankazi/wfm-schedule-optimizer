import pandas as pd

from src.continuous import (
    DEFAULT_247_GRID,
    TimeGrid,
    WeeklyAgent,
    allowed_start_slots,
    build_weekly_shift_options,
    eligible_for_option,
    generate_continuous_demand,
    generate_continuous_roster,
    rest_intervals,
)
from src.optimizer_247 import (
    WeeklySolverConfig,
    build_weekly_model,
    conflicting_option_pairs,
    solve_weekly_schedule,
)


def test_default_grid_is_a_continuous_672_interval_week():
    grid = DEFAULT_247_GRID
    assert grid.days == 7
    assert grid.intervals_per_day == 96
    assert grid.total_intervals == 672
    assert grid.time_label(0) == "00:00"
    assert grid.time_label(95) == "23:45"


def test_start_cadence_is_hourly_overnight_and_half_hourly_daytime():
    starts = allowed_start_slots()
    assert len(starts) == 40
    labels = {DEFAULT_247_GRID.time_label(slot) for slot in starts}
    assert {"02:00", "06:00", "06:30", "21:30", "22:00", "23:00"} <= labels
    assert "02:30" not in labels
    assert "22:30" not in labels


def test_weekly_options_and_default_roster_produce_about_24k_binaries():
    options = build_weekly_shift_options()
    agents = generate_continuous_roster()
    assert len(options) == 7 * 40 * 3
    assert len({option.id for option in options}) == len(options)
    assert len(agents) == 65

    eligible_variables = sum(
        eligible_for_option(agent, option)
        for agent in agents
        for option in options
    )
    assert eligible_variables == 22_680


def test_sunday_night_shift_wraps_into_monday_without_truncation():
    options = {option.id: option for option in build_weekly_shift_options()}
    sunday_night = options["Sun_FT8_2200"]
    assert sunday_night.end_abs > DEFAULT_247_GRID.total_intervals
    assert 0 in sunday_night.occupied_slots
    assert max(sunday_night.occupied_slots) == 671
    assert len(sunday_night.occupied_slots) == sunday_night.span


def test_ten_hour_rest_conflicts_are_precomputed_across_days_and_week_end():
    options = build_weekly_shift_options()
    by_id = {option.id: option for option in options}
    assert rest_intervals(by_id["Mon_FT8_2200"], by_id["Tue_FT8_0600"]) < 40
    assert rest_intervals(by_id["Sun_FT8_2200"], by_id["Mon_FT8_0600"]) < 40

    full_time = [option for option in options if option.kind == "full_time"]
    conflicts = set(conflicting_option_pairs(full_time))
    assert ("Mon_FT8_2200", "Tue_FT8_0600") in conflicts
    assert ("Sun_FT8_2200", "Mon_FT8_0600") in conflicts


def test_peak_70_workload_has_672_rows_and_a_binding_presence_floor():
    demand = generate_continuous_demand(seed=42, peak_calls=70)
    assert len(demand) == 672
    assert demand.week_interval.tolist() == list(range(672))
    assert demand.calls.max() <= 70
    assert 8_400 <= int(demand.required_agents.sum()) <= 8_600
    assert (demand.required_agents >= demand.min_presence_floor).all()
    floor_rows = demand[demand.binding_constraint == "floor"]
    assert not floor_rows.empty
    assert (floor_rows.required_agents == 4).all()


def test_model_contains_penalized_presence_floor_and_rotation_constraints():
    grid = TimeGrid(days=2)
    options = build_weekly_shift_options(grid)
    agents = [
        WeeklyAgent(0, "AGT-001", "full_time", 0.0, 8.0),
    ]
    demand = pd.DataFrame(
        {
            "week_interval": range(grid.total_intervals),
            "required_agents": [1] * grid.total_intervals,
            "min_presence_floor": [1] * grid.total_intervals,
        }
    )
    config = WeeklySolverConfig(
        min_rest_hours=10,
        max_consecutive_nights=1,
        full_time_night_min=1,
        full_time_night_max=1,
    )
    model, *_ = build_weekly_model(agents, options, demand, config, grid)
    floor_constraint = model.get_constraint_by_name("presence_floor_0")
    assert floor_constraint is not None
    assert "floor_under_0" in str(floor_constraint)
    assert model.get_constraint_by_name("night_run_0_0") is not None
    assert model.get_constraint_by_name("rest_0_0") is not None
    assert "night_short_0" in str(model.get_constraint_by_name("night_min_0"))


def test_small_zero_demand_week_solves_and_is_verified():
    grid = TimeGrid(days=2)
    options = build_weekly_shift_options(grid)
    agents = [WeeklyAgent(0, "AGT-001", "full_time", 0.0, 0.0)]
    demand = pd.DataFrame(
        {
            "week_interval": range(grid.total_intervals),
            "required_agents": [0] * grid.total_intervals,
            "min_presence_floor": [0] * grid.total_intervals,
        }
    )
    config = WeeklySolverConfig(
        max_consecutive_nights=1,
        full_time_night_min=0,
        full_time_night_max=0,
    )
    schedule = solve_weekly_schedule(
        agents,
        options,
        demand,
        config,
        grid,
        time_limit_sec=10,
        gap=0,
    )
    assert schedule.status == "Optimal"
    assert not schedule.assignments
    assert schedule.summary()["floor_breach_intervals"] == 0


def test_floor_slack_makes_an_unstaffable_floor_measurable():
    grid = TimeGrid(days=2)
    options = build_weekly_shift_options(grid)
    agents = [WeeklyAgent(0, "AGT-001", "full_time", 0.0, 0.0)]
    demand = pd.DataFrame(
        {
            "week_interval": range(grid.total_intervals),
            "required_agents": [1] * grid.total_intervals,
            "min_presence_floor": [1] * grid.total_intervals,
        }
    )
    config = WeeklySolverConfig(
        max_consecutive_nights=1,
        full_time_night_min=0,
        full_time_night_max=0,
    )
    schedule = solve_weekly_schedule(
        agents,
        options,
        demand,
        config,
        grid,
        time_limit_sec=10,
        gap=0,
    )
    summary = schedule.summary()
    assert summary["status"] == "Optimal"
    assert summary["floor_breach_intervals"] == grid.total_intervals
    assert summary["floor_breach_agent_intervals"] == grid.total_intervals
    assert summary["understaffed_agent_intervals"] == grid.total_intervals


def test_night_minimum_is_soft_so_the_all_slack_point_is_feasible():
    grid = TimeGrid(days=2)
    options = build_weekly_shift_options(grid)
    agents = [WeeklyAgent(0, "AGT-001", "full_time", 0.0, 0.0)]
    demand = pd.DataFrame(
        {
            "week_interval": range(grid.total_intervals),
            "required_agents": [0] * grid.total_intervals,
            "min_presence_floor": [0] * grid.total_intervals,
        }
    )
    config = WeeklySolverConfig(
        max_consecutive_nights=1,
        full_time_night_min=1,
        full_time_night_max=1,
    )
    schedule = solve_weekly_schedule(
        agents,
        options,
        demand,
        config,
        grid,
        time_limit_sec=10,
        gap=0,
    )
    assert schedule.summary()["weekly_night_shortfall_shifts"] == 1
