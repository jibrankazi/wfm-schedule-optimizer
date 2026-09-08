"""Google OR-Tools CP-SAT parity model for the continuous 24/7 roster.

The feasible region and objective mirror :mod:`src.optimizer_247`. Shift
starts remain enumerated fixed options; changing them to flexible interval
variables would create a different scheduling problem rather than a solver
benchmark.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from .continuous import (
    DEFAULT_247_GRID,
    TimeGrid,
    WeeklyAgent,
    WeeklyShiftOption,
    eligible_for_option,
)
from .optimizer_247 import (
    SolverTerminationError,
    WeeklyAssignment,
    WeeklySchedule,
    WeeklySolverConfig,
    _demand_arrays,
    _validate_schedule,
)


@dataclass
class CpSatModelBuild:
    """Variables and inputs required to solve and verify one CP-SAT model."""

    model: cp_model.CpModel
    assignments: dict[tuple[int, str], cp_model.IntVar]
    demand_under: dict[int, cp_model.IntVar]
    demand_over: dict[int, cp_model.IntVar]
    floor_under: dict[int, cp_model.IntVar]
    hour_short_slots: dict[int, cp_model.IntVar]
    night_short: dict[int, cp_model.IntVar]
    required: np.ndarray
    floor: np.ndarray


def _integer_weight(value: float, label: str) -> int:
    """Return an exact CP-SAT objective weight for the calibrated model."""
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(f"{label} must be an integer for CP-SAT parity")
    return int(rounded)


def _paid_quarter_hours(hours: float, label: str) -> int:
    slots = hours * 4.0
    if not math.isclose(slots, round(slots), abs_tol=1e-9):
        raise ValueError(f"{label} must align to 15-minute paid increments")
    return int(round(slots))


def build_weekly_cpsat_model(
    agents: list[WeeklyAgent],
    options: list[WeeklyShiftOption],
    demand: pd.DataFrame,
    config: WeeklySolverConfig | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> CpSatModelBuild:
    """Build the CP-SAT equivalent of ``build_weekly_model`` without solving."""
    if not agents:
        raise ValueError("agents must not be empty")
    if not options:
        raise ValueError("options must not be empty")
    config = config or WeeklySolverConfig()
    config.validate(grid)
    required, floor = _demand_arrays(demand, grid)

    option_by_id = {option.id: option for option in options}
    if len(option_by_id) != len(options):
        raise ValueError("shift option ids must be unique")
    if len({agent.id for agent in agents}) != len(agents):
        raise ValueError("agent ids must be unique")

    allowed: dict[tuple[int, int], list[str]] = {}
    for agent in agents:
        for day in range(grid.days):
            allowed[(agent.id, day)] = [
                option.id
                for option in options
                if option.start_day == day and eligible_for_option(agent, option)
            ]

    model = cp_model.CpModel()
    x: dict[tuple[int, str], cp_model.IntVar] = {}
    for agent in agents:
        for day in range(grid.days):
            for option_id in allowed[(agent.id, day)]:
                x[(agent.id, option_id)] = model.new_bool_var(
                    f"x_{agent.id}_{option_id}"
                )

    demand_under = {
        slot: model.new_int_var(0, int(required[slot]), f"under_{slot}")
        for slot in range(grid.total_intervals)
    }
    demand_over = {
        slot: model.new_int_var(0, len(agents), f"over_{slot}")
        for slot in range(grid.total_intervals)
    }
    floor_under = {
        slot: model.new_int_var(0, int(floor[slot]), f"floor_under_{slot}")
        for slot in range(grid.total_intervals)
        if floor[slot] > 0
    }
    hour_short_slots = {
        agent.id: model.new_int_var(
            0,
            _paid_quarter_hours(
                agent.min_weekly_hours, f"agent {agent.id} minimum hours"
            ),
            f"short_slots_{agent.id}",
        )
        for agent in agents
    }
    night_short = {
        agent.id: model.new_int_var(
            0, config.full_time_night_min, f"night_short_{agent.id}"
        )
        for agent in agents
        if agent.kind == "full_time" and config.full_time_night_min > 0
    }

    # At most one shift start per employee per calendar day.
    for agent in agents:
        for day in range(grid.days):
            model.add_at_most_one(
                x[(agent.id, option_id)]
                for option_id in allowed[(agent.id, day)]
            )

    # Weekly maximum is hard. The minimum uses paid-quarter-hour slack so its
    # objective cost exactly equals CBC's 4 * min_hours_penalty per hour.
    for agent in agents:
        worked_slots = sum(
            _paid_quarter_hours(
                option_by_id[option_id].paid_hours,
                f"option {option_id} paid hours",
            )
            * x[(agent.id, option_id)]
            for day in range(grid.days)
            for option_id in allowed[(agent.id, day)]
        )
        min_slots = _paid_quarter_hours(
            agent.min_weekly_hours, f"agent {agent.id} minimum hours"
        )
        max_slots = _paid_quarter_hours(
            agent.max_weekly_hours, f"agent {agent.id} maximum hours"
        )
        model.add(worked_slots <= max_slots)
        model.add(worked_slots + hour_short_slots[agent.id] >= min_slots)

    # Use the same bounded adjacent-day rest inequality as CBC, including the
    # cyclic Sunday-to-Monday boundary.
    min_rest = config.rest_intervals(grid)
    max_span = max(option.span for option in options)
    rest_big_m = grid.intervals_per_day + max_span + min_rest
    for agent in agents:
        for day in range(grid.days):
            next_day = (day + 1) % grid.days
            previous_ids = allowed[(agent.id, day)]
            next_ids = allowed[(agent.id, next_day)]
            previous_worked = sum(x[(agent.id, option_id)] for option_id in previous_ids)
            next_worked = sum(x[(agent.id, option_id)] for option_id in next_ids)
            previous_end = sum(
                x[(agent.id, option_id)]
                * (option_by_id[option_id].start_slot + option_by_id[option_id].span)
                for option_id in previous_ids
            )
            next_start = sum(
                x[(agent.id, option_id)] * option_by_id[option_id].start_slot
                for option_id in next_ids
            )
            model.add(
                grid.intervals_per_day
                + next_start
                - previous_end
                + rest_big_m * (2 - previous_worked - next_worked)
                >= min_rest
            )

    # Exact cyclic k-in-(k+1) night window used by the CBC formulation.
    window = config.max_consecutive_nights + 1
    if window > 1:
        for agent in agents:
            for start_day in range(grid.days):
                days = {
                    (start_day + offset) % grid.days for offset in range(window)
                }
                night_ids = [
                    option_id
                    for day in days
                    for option_id in allowed[(agent.id, day)]
                    if option_by_id[option_id].is_night
                ]
                if night_ids:
                    model.add(
                        sum(x[(agent.id, option_id)] for option_id in night_ids)
                        <= config.max_consecutive_nights
                    )

    for agent in (candidate for candidate in agents if candidate.kind == "full_time"):
        night_ids = [
            option_id
            for day in range(grid.days)
            for option_id in allowed[(agent.id, day)]
            if option_by_id[option_id].is_night
        ]
        nights = sum(x[(agent.id, option_id)] for option_id in night_ids)
        if config.full_time_night_min > 0:
            model.add(
                nights + night_short[agent.id] >= config.full_time_night_min
            )
        model.add(nights <= config.full_time_night_max)

    covering: dict[int, list[str]] = {
        slot: [] for slot in range(grid.total_intervals)
    }
    for option in options:
        for slot in option.coverage_slots:
            covering[slot].append(option.id)

    for slot in range(grid.total_intervals):
        staffed = sum(
            x[(agent.id, option_id)]
            for agent in agents
            for option_id in covering[slot]
            if (agent.id, option_id) in x
        )
        model.add(
            staffed + demand_under[slot] - demand_over[slot]
            == int(required[slot])
        )
        if floor[slot] > 0:
            model.add(staffed + floor_under[slot] >= int(floor[slot]))

    understaff_weight = _integer_weight(
        config.understaff_penalty, "understaff_penalty"
    )
    floor_weight = _integer_weight(
        config.floor_breach_penalty, "floor_breach_penalty"
    )
    overstaff_weight = _integer_weight(
        config.overstaff_penalty, "overstaff_penalty"
    )
    min_hours_weight = _integer_weight(
        config.min_hours_penalty, "min_hours_penalty"
    )
    night_min_weight = _integer_weight(
        config.night_min_penalty, "night_min_penalty"
    )
    model.minimize(
        understaff_weight * sum(demand_under.values())
        + floor_weight * sum(floor_under.values())
        + overstaff_weight * sum(demand_over.values())
        + min_hours_weight * sum(hour_short_slots.values())
        + night_min_weight * sum(night_short.values())
    )

    return CpSatModelBuild(
        model=model,
        assignments=x,
        demand_under=demand_under,
        demand_over=demand_over,
        floor_under=floor_under,
        hour_short_slots=hour_short_slots,
        night_short=night_short,
        required=required,
        floor=floor,
    )


def solve_weekly_schedule_cpsat(
    agents: list[WeeklyAgent],
    options: list[WeeklyShiftOption],
    demand: pd.DataFrame,
    config: WeeklySolverConfig | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
    time_limit_sec: float = 200.0,
    gap: float = 0.03,
    workers: int = 8,
    random_seed: int = 42,
    msg: bool = False,
) -> WeeklySchedule:
    """Solve the parity model and return only an invariant-checked incumbent."""
    if time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive")
    if not 0.0 <= gap < 1.0:
        raise ValueError("gap must be in [0, 1)")
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = config or WeeklySolverConfig()
    build = build_weekly_cpsat_model(agents, options, demand, config, grid)
    option_by_id = {option.id: option for option in options}
    agent_by_id = {agent.id: agent for agent in agents}

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.relative_gap_limit = gap
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = random_seed
    solver.parameters.log_search_progress = msg

    started = time.perf_counter()
    status_code = solver.Solve(build.model)
    elapsed = time.perf_counter() - started
    raw_status = solver.StatusName(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise SolverTerminationError(
            f"CP-SAT returned {raw_status!r}; no schedule was published"
        )

    objective = float(solver.ObjectiveValue())
    objective_bound = float(solver.BestObjectiveBound())
    relative_gap_pct = max(
        0.0,
        100.0 * (objective - objective_bound) / max(abs(objective), 1e-12),
    )
    if math.isclose(objective, objective_bound, abs_tol=1e-7):
        status = "Optimal"
        termination_reason = "optimal solution found"
        relative_gap_pct = 0.0
    else:
        status = "Feasible"
        termination_reason = (
            "relative gap limit"
            if status_code == cp_model.OPTIMAL
            else "time limit" if elapsed >= 0.98 * time_limit_sec else "search stopped"
        )

    assignments: list[WeeklyAssignment] = []
    coverage = np.zeros(grid.total_intervals, dtype=int)
    agent_hours = {agent.id: 0.0 for agent in agents}
    for (agent_id, option_id), variable in build.assignments.items():
        if solver.Value(variable) == 1:
            option = option_by_id[option_id]
            assignments.append(
                WeeklyAssignment(agent_id, agent_by_id[agent_id].name, option)
            )
            agent_hours[agent_id] += option.paid_hours
            coverage[list(option.coverage_slots)] += 1

    actual_floor_shortfall = np.clip(build.floor - coverage, 0, None)
    reported_floor_shortfall = np.zeros(grid.total_intervals, dtype=int)
    for slot, variable in build.floor_under.items():
        reported_floor_shortfall[slot] = solver.Value(variable)
    if not np.array_equal(reported_floor_shortfall, actual_floor_shortfall):
        raise SolverTerminationError("CP-SAT floor slack does not match coverage")

    shortfall_hours: dict[int, float] = {}
    for agent in agents:
        missing_hours = max(
            0.0, agent.min_weekly_hours - agent_hours[agent.id]
        )
        reported_slots = solver.Value(build.hour_short_slots[agent.id])
        if not math.isclose(reported_slots / 4.0, missing_hours, abs_tol=1e-7):
            raise SolverTerminationError(
                "CP-SAT weekly-hour slack does not match assignments"
            )
        if missing_hours > 1e-7:
            shortfall_hours[agent.id] = missing_hours

    assigned_nights = {agent.id: 0 for agent in agents}
    for assignment in assignments:
        if assignment.option.is_night:
            assigned_nights[assignment.agent_id] += 1
    night_shortfall: dict[int, int] = {}
    for agent_id, variable in build.night_short.items():
        missing_nights = max(
            0, config.full_time_night_min - assigned_nights[agent_id]
        )
        if solver.Value(variable) != missing_nights:
            raise SolverTerminationError(
                "CP-SAT weekly-night slack does not match assignments"
            )
        if missing_nights:
            night_shortfall[agent_id] = missing_nights

    schedule = WeeklySchedule(
        assignments=assignments,
        coverage=coverage,
        required=build.required,
        presence_floor=build.floor,
        agent_hours=agent_hours,
        status=status,
        solver_status=raw_status,
        termination_reason=termination_reason,
        objective=objective,
        objective_bound=objective_bound,
        relative_gap_pct=relative_gap_pct,
        solve_seconds=elapsed,
        variables=len(build.assignments),
        total_variables=len(build.model.proto.variables),
        constraints=len(build.model.proto.constraints),
        shortfall_hours=shortfall_hours,
        night_shortfall=night_shortfall,
    )
    _validate_schedule(schedule, agents, config, grid)
    return schedule
