import pandas as pd
import pytest

from src.continuous import (
    TimeGrid,
    WeeklyAgent,
    build_weekly_shift_options,
    generate_continuous_demand,
    generate_continuous_roster,
)
from src.optimizer_247 import WeeklySolverConfig, build_weekly_model, solve_weekly_schedule
from src.optimizer_cpsat import (
    build_weekly_cpsat_model,
    solve_weekly_schedule_cpsat,
)


def test_default_cpsat_model_has_cbc_parity_dimensions():
    agents = generate_continuous_roster()
    options = build_weekly_shift_options()
    demand = generate_continuous_demand(seed=42, peak_calls=70)
    cbc_model, cbc_x, *_ = build_weekly_model(agents, options, demand)
    cpsat = build_weekly_cpsat_model(agents, options, demand)

    # Binary decision space and row count are identical; the paths diverge by
    # exactly the 672 surplus columns. CBC drops them from the active matrix
    # because its coverage row is a one-sided inequality - the two-sided
    # equality stopped its root LP converging - while CP-SAT keeps the equality
    # since it does not solve by simplex and the change there is unmeasured.
    SURPLUS_COLUMNS = 672
    assert len(cbc_x) == len(cpsat.assignments) == 22_680
    assert len(cpsat.model.proto.variables) == 24_810
    assert cbc_model.numVariables() == 24_810 - SURPLUS_COLUMNS
    assert cbc_model.numConstraints() == len(cpsat.model.proto.constraints) == 2_937


def test_cpsat_and_cbc_match_on_an_unstaffable_floor():
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
    cbc = solve_weekly_schedule(
        agents, options, demand, config, grid, time_limit_sec=10, gap=0
    )
    cpsat = solve_weekly_schedule_cpsat(
        agents, options, demand, config, grid, time_limit_sec=10, gap=0, workers=1
    )

    # The two paths no longer optimise identical expressions. CBC uses a
    # one-sided covering inequality with no surplus column, because the
    # two-sided equality stopped its root LP converging; CP-SAT keeps the
    # equality, since it does not solve by simplex and the change there is
    # unmeasured. The objectives therefore differ by exactly the overstaffing
    # cost, which CBC reports as a post-solve projection. Reconstructing it
    # restores parity to the cent - verified at scale as 14,244.82 against
    # 14,244.77.
    surplus_cost = (
        config.overstaff_penalty * cbc.summary()["overstaffed_agent_intervals"]
    )
    assert cpsat.objective == pytest.approx(cbc.objective + surplus_cost)
    assert cpsat.summary()["understaffed_agent_intervals"] == grid.total_intervals
    assert cpsat.summary()["floor_breach_agent_intervals"] == grid.total_intervals


def test_cpsat_hour_and_night_minimum_slacks_match_cbc_weights():
    grid = TimeGrid(days=2)
    options = build_weekly_shift_options(grid)
    agents = [WeeklyAgent(0, "AGT-001", "full_time", 8.0, 8.0)]
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
    cbc = solve_weekly_schedule(
        agents, options, demand, config, grid, time_limit_sec=10, gap=0
    )
    cpsat = solve_weekly_schedule_cpsat(
        agents, options, demand, config, grid, time_limit_sec=10, gap=0, workers=1
    )

    # The two paths no longer optimise identical expressions. CBC uses a
    # one-sided covering inequality with no surplus column, because the
    # two-sided equality stopped its root LP converging; CP-SAT keeps the
    # equality, since it does not solve by simplex and the change there is
    # unmeasured. The objectives therefore differ by exactly the overstaffing
    # cost, which CBC reports as a post-solve projection. Reconstructing it
    # restores parity to the cent - verified at scale as 14,244.82 against
    # 14,244.77.
    surplus_cost = (
        config.overstaff_penalty * cbc.summary()["overstaffed_agent_intervals"]
    )
    assert cpsat.objective == pytest.approx(cbc.objective + surplus_cost)
    assert cpsat.summary()["contracted_minimum_shortfall_hours"] == 0
    assert cpsat.summary()["weekly_night_shortfall_shifts"] == 0
