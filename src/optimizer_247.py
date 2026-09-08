"""PuLP/CBC formulation for a continuous cyclic 24/7 roster.

Decision variables select concrete weekly shift starts. Coverage is indexed
over one 672-slot week, so overnight shifts cross day and week boundaries
without truncation. The model adds penalized minimum-presence slack, aggregated
inter-day rest rules, and inexpensive night-rotation bounds.
"""

from __future__ import annotations

import math
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from .continuous import (
    DEFAULT_247_GRID,
    TimeGrid,
    WeeklyAgent,
    WeeklyShiftOption,
    eligible_for_option,
    rest_intervals,
)


class SolverTerminationError(RuntimeError):
    """Raised when CBC does not return a verified integer allocation."""


@dataclass(frozen=True)
class WeeklySolverConfig:
    min_rest_hours: float = 10.0
    max_consecutive_nights: int = 3
    full_time_night_min: int = 1
    full_time_night_max: int = 3
    understaff_penalty: float = 100.0
    floor_breach_penalty: float = 1_000.0
    overstaff_penalty: float = 1.0  # Used by the CP-SAT variant, omitted by CBC.
    min_hours_penalty: float = 50.0
    night_min_penalty: float = 50.0

    def validate(self, grid: TimeGrid) -> None:
        if self.min_rest_hours < 0:
            raise ValueError("min_rest_hours must be non-negative")
        if not 0 <= self.max_consecutive_nights < grid.days:
            raise ValueError("max_consecutive_nights must be between 0 and days-1")
        if not 0 <= self.full_time_night_min <= self.full_time_night_max:
            raise ValueError("invalid full-time weekly night band")
        if min(
            self.understaff_penalty,
            self.floor_breach_penalty,
            self.overstaff_penalty,
            self.min_hours_penalty,
            self.night_min_penalty,
        ) < 0:
            raise ValueError("objective penalties must be non-negative")

    def rest_intervals(self, grid: TimeGrid) -> int:
        intervals = self.min_rest_hours * 60.0 / grid.interval_minutes
        if not math.isclose(intervals, round(intervals)):
            raise ValueError("min_rest_hours must align to the grid resolution")
        return int(round(intervals))


@dataclass(frozen=True)
class WeeklyAssignment:
    agent_id: int
    agent_name: str
    option: WeeklyShiftOption


@dataclass
class WeeklySchedule:
    assignments: list[WeeklyAssignment]
    coverage: np.ndarray
    required: np.ndarray
    presence_floor: np.ndarray
    agent_hours: dict[int, float]
    status: str
    solver_status: str
    termination_reason: str
    objective: float
    objective_bound: float | None
    relative_gap_pct: float | None
    solve_seconds: float
    variables: int
    total_variables: int
    constraints: int
    shortfall_hours: dict[int, float] = field(default_factory=dict)
    night_shortfall: dict[int, int] = field(default_factory=dict)

    @property
    def understaffed(self) -> np.ndarray:
        return np.clip(self.required - self.coverage, 0, None)

    @property
    def overstaffed(self) -> np.ndarray:
        return np.clip(self.coverage - self.required, 0, None)

    @property
    def floor_shortfall(self) -> np.ndarray:
        return np.clip(self.presence_floor - self.coverage, 0, None)

    def summary(self) -> dict[str, float | int | str | None]:
        return {
            "status": self.status,
            "solver_status": self.solver_status,
            "termination_reason": self.termination_reason,
            "solve_seconds": round(self.solve_seconds, 2),
            "objective": round(self.objective, 3),
            "objective_bound": (
                round(self.objective_bound, 3)
                if self.objective_bound is not None
                else None
            ),
            "relative_gap_pct": (
                round(self.relative_gap_pct, 3)
                if self.relative_gap_pct is not None
                else None
            ),
            "binary_assignment_variables": self.variables,
            "total_model_variables": self.total_variables,
            "constraints": self.constraints,
            "shifts_assigned": len(self.assignments),
            "scheduled_hours": round(sum(self.agent_hours.values()), 2),
            "understaffed_agent_intervals": int(self.understaffed.sum()),
            "overstaffed_agent_intervals": int(self.overstaffed.sum()),
            "coverage_compliance_pct": round(
                100.0 * float(np.count_nonzero(self.coverage >= self.required))
                / self.required.size,
                2,
            ),
            "demand_fulfilment_pct": round(
                100.0
                * float(np.minimum(self.coverage, self.required).sum())
                / max(1, int(self.required.sum())),
                2,
            ),
            "floor_breach_intervals": int(np.count_nonzero(self.floor_shortfall)),
            "floor_breach_agent_intervals": int(self.floor_shortfall.sum()),
            "agents_below_contracted_minimum": len(self.shortfall_hours),
            "contracted_minimum_shortfall_hours": round(
                sum(self.shortfall_hours.values()), 2
            ),
            "agents_below_weekly_night_minimum": len(self.night_shortfall),
            "weekly_night_shortfall_shifts": sum(self.night_shortfall.values()),
        }


def _demand_arrays(
    demand: pd.DataFrame,
    grid: TimeGrid,
) -> tuple[np.ndarray, np.ndarray]:
    needed = {"week_interval", "required_agents"}
    missing = needed.difference(demand.columns)
    if missing:
        raise ValueError(f"demand is missing columns: {sorted(missing)}")
    if demand.week_interval.duplicated().any():
        raise ValueError("demand has duplicate week_interval rows")
    expected = set(range(grid.total_intervals))
    actual = {int(value) for value in demand.week_interval}
    if actual != expected:
        raise ValueError("demand must contain every interval in the planning grid")

    required = np.zeros(grid.total_intervals, dtype=int)
    floor = np.zeros(grid.total_intervals, dtype=int)
    has_floor = "min_presence_floor" in demand.columns
    for row in demand.itertuples():
        slot = int(row.week_interval)
        required[slot] = int(row.required_agents)
        floor[slot] = int(row.min_presence_floor) if has_floor else 0
    if (required < 0).any() or (floor < 0).any():
        raise ValueError("headcount requirements must be non-negative")
    if (floor > required).any():
        raise ValueError("presence floor cannot exceed required_agents")
    return required, floor


def conflicting_option_pairs(
    options: list[WeeklyShiftOption],
    grid: TimeGrid = DEFAULT_247_GRID,
    min_rest_intervals: int = 40,
) -> list[tuple[str, str]]:
    """Precompute sparse adjacent-day shift pairs that violate rest."""
    conflict_sets = conflicting_option_sets(
        options, grid=grid, min_rest_intervals=min_rest_intervals
    )
    return [
        (earlier_id, later_id)
        for earlier_id, later_ids in conflict_sets.items()
        for later_id in later_ids
    ]


def conflicting_option_sets(
    options: list[WeeklyShiftOption],
    grid: TimeGrid = DEFAULT_247_GRID,
    min_rest_intervals: int = 40,
) -> dict[str, list[str]]:
    """Group conflicting next-day starts by their preceding shift.

    Because the model already permits at most one shift on the following day,
    ``x[earlier] + sum(x[conflicting later starts]) <= 1`` is exactly
    equivalent to emitting one pairwise inequality for every combination and
    is dramatically smaller for CBC presolve.
    """
    by_day: dict[int, list[WeeklyShiftOption]] = {
        day: [] for day in range(grid.days)
    }
    for option in options:
        by_day[option.start_day].append(option)

    conflicts: dict[str, list[str]] = {}
    for day in range(grid.days):
        next_day = (day + 1) % grid.days
        for earlier in by_day[day]:
            later_ids = []
            for later in by_day[next_day]:
                if rest_intervals(earlier, later, grid) < min_rest_intervals:
                    later_ids.append(later.id)
            if later_ids:
                conflicts[earlier.id] = later_ids
    return conflicts


def build_weekly_model(
    agents: list[WeeklyAgent],
    options: list[WeeklyShiftOption],
    demand: pd.DataFrame,
    config: WeeklySolverConfig | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
) -> tuple[
    pulp.LpProblem,
    dict[tuple[int, str], pulp.LpVariable],
    dict[int, pulp.LpVariable],
    dict[int, pulp.LpVariable],
    dict[int, pulp.LpVariable],
    dict[int, pulp.LpVariable],
    dict[int, pulp.LpVariable],
    np.ndarray,
    np.ndarray,
]:
    """Build the continuous-week MILP without running CBC."""
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

    model = pulp.LpProblem("continuous_weekly_roster", pulp.LpMinimize)
    x: dict[tuple[int, str], pulp.LpVariable] = {}
    for agent in agents:
        for day in range(grid.days):
            for option_id in allowed[(agent.id, day)]:
                x[(agent.id, option_id)] = pulp.LpVariable(
                    f"x_{agent.id}_{option_id}", cat=pulp.LpBinary
                )

    under = {
        slot: pulp.LpVariable(f"under_{slot}", lowBound=0)
        for slot in range(grid.total_intervals)
    }
    over = {
        slot: pulp.LpVariable(f"over_{slot}", lowBound=0)
        for slot in range(grid.total_intervals)
    }
    short = {
        agent.id: pulp.LpVariable(f"short_{agent.id}", lowBound=0)
        for agent in agents
    }
    floor_under = {
        slot: pulp.LpVariable(f"floor_under_{slot}", lowBound=0, cat=pulp.LpInteger)
        for slot in range(grid.total_intervals)
        if floor[slot] > 0
    }
    night_short = {
        agent.id: pulp.LpVariable(
            f"night_short_{agent.id}", lowBound=0, cat=pulp.LpInteger
        )
        for agent in agents
        if agent.kind == "full_time" and config.full_time_night_min > 0
    }
    model += (
        config.understaff_penalty * pulp.lpSum(under.values())
        + config.floor_breach_penalty * pulp.lpSum(floor_under.values())
        + config.min_hours_penalty * 4.0 * pulp.lpSum(short.values())
        + config.night_min_penalty * pulp.lpSum(night_short.values())
    )

    # At most one shift start per employee per calendar day.
    for agent in agents:
        for day in range(grid.days):
            ids = allowed[(agent.id, day)]
            if ids:
                model += (
                    pulp.lpSum(x[(agent.id, option_id)] for option_id in ids) <= 1,
                    f"one_shift_{agent.id}_{day}",
                )

    # Weekly maximum is hard; contracted minimum remains an explicit soft gap.
    for agent in agents:
        worked = pulp.lpSum(
            x[(agent.id, option_id)] * option_by_id[option_id].paid_hours
            for day in range(grid.days)
            for option_id in allowed[(agent.id, day)]
        )
        model += worked <= agent.max_weekly_hours, f"max_hours_{agent.id}"
        model += worked + short[agent.id] >= agent.min_weekly_hours, f"min_hours_{agent.id}"

    # Exact inter-day rest with one bounded inequality per agent/day.  Because
    # one_shift makes each weighted start/end expression select at most one
    # option, this is equivalent to every pairwise conflict exclusion while
    # avoiding hundreds of thousands of rows in CBC presolve.  The M value is
    # tight to one day plus the longest shift and required rest.
    min_rest = config.rest_intervals(grid)
    max_span = max(option.span for option in options)
    rest_big_m = grid.intervals_per_day + max_span + min_rest
    for agent in agents:
        for day in range(grid.days):
            next_day = (day + 1) % grid.days
            previous_ids = allowed[(agent.id, day)]
            next_ids = allowed[(agent.id, next_day)]
            previous_worked = pulp.lpSum(
                x[(agent.id, option_id)] for option_id in previous_ids
            )
            next_worked = pulp.lpSum(
                x[(agent.id, option_id)] for option_id in next_ids
            )
            previous_end = pulp.lpSum(
                x[(agent.id, option_id)]
                * (option_by_id[option_id].start_slot + option_by_id[option_id].span)
                for option_id in previous_ids
            )
            next_start = pulp.lpSum(
                x[(agent.id, option_id)] * option_by_id[option_id].start_slot
                for option_id in next_ids
            )
            model += (
                grid.intervals_per_day
                + next_start
                - previous_end
                + rest_big_m * (2 - previous_worked - next_worked)
                >= min_rest,
                f"rest_{agent.id}_{day}",
            )

    # Sparse fairness: no more than k nights in any cyclic k+1 day window.
    window = config.max_consecutive_nights + 1
    if window > 1:
        for agent in agents:
            for start_day in range(grid.days):
                days = {(start_day + offset) % grid.days for offset in range(window)}
                night_ids = [
                    option_id
                    for day in days
                    for option_id in allowed[(agent.id, day)]
                    if option_by_id[option_id].is_night
                ]
                if night_ids:
                    model += (
                        pulp.lpSum(x[(agent.id, option_id)] for option_id in night_ids)
                        <= config.max_consecutive_nights,
                        f"night_run_{agent.id}_{start_day}",
                    )

    for agent in (candidate for candidate in agents if candidate.kind == "full_time"):
        night_ids = [
            option_id
            for day in range(grid.days)
            for option_id in allowed[(agent.id, day)]
            if option_by_id[option_id].is_night
        ]
        nights = pulp.lpSum(x[(agent.id, option_id)] for option_id in night_ids)
        if config.full_time_night_min > 0:
            model += (
                nights + night_short[agent.id] >= config.full_time_night_min,
                f"night_min_{agent.id}",
            )
        model += nights <= config.full_time_night_max, f"night_max_{agent.id}"

    covering: dict[int, list[str]] = {
        slot: [] for slot in range(grid.total_intervals)
    }
    for option in options:
        for slot in option.coverage_slots:
            covering[slot].append(option.id)

    # Every profile builds `required` as max(queueing, floor), so the floor can
    # never exceed the requirement. The floor row depends on that; assert it
    # rather than trust it, so a profile that breaks the invariant fails loudly
    # instead of silently mispricing safety coverage.
    violations = [
        slot for slot in range(grid.total_intervals) if floor[slot] > required[slot]
    ]
    if violations:
        raise ValueError(
            f"Profile invariant violated: floor > required at slots {violations[:5]}."
        )

    # CBC deliberately omits surplus columns and their cost. This changes the
    # objective from the CP-SAT variant; overstaffing is reported after solving.
    # Similar bounds on a sample do not establish equivalent objectives.
    for slot in range(grid.total_intervals):
        staffed = pulp.lpSum(
            x[(agent.id, option_id)]
            for agent in agents
            for option_id in covering[slot]
            if (agent.id, option_id) in x
        )
        model += (
            staffed + under[slot] >= int(required[slot]),
            f"coverage_{slot}",
        )
        # The floor needs its own row. Dropping it was safe for the feasible
        # region but only while u_t = 0: once a deficit is accepted, nothing
        # stops coverage falling through the floor, and the breach is priced at
        # the understaffing weight instead of the floor weight. Measured, that
        # tenfold discount took floor breaches from 4 to 51.
        if floor[slot] > 0:
            model += (
                staffed + floor_under[slot] >= int(floor[slot]),
                f"presence_floor_{slot}",
            )

    for surplus in over.values():
        surplus.upBound = 0

    return model, x, under, over, floor_under, short, night_short, required, floor


def _validate_schedule(
    schedule: WeeklySchedule,
    agents: list[WeeklyAgent],
    config: WeeklySolverConfig,
    grid: TimeGrid,
) -> None:
    """Reject any extracted incumbent that violates a hard invariant."""
    by_agent = {agent.id: agent for agent in agents}
    rebuilt = np.zeros(grid.total_intervals, dtype=int)
    assignments_by_agent: dict[int, list[WeeklyAssignment]] = {
        agent.id: [] for agent in agents
    }

    for assignment in schedule.assignments:
        agent = by_agent[assignment.agent_id]
        if not eligible_for_option(agent, assignment.option):
            raise SolverTerminationError("solver returned an ineligible assignment")
        assignments_by_agent[agent.id].append(assignment)
        rebuilt[list(assignment.option.coverage_slots)] += 1

    if not np.array_equal(rebuilt, schedule.coverage):
        raise SolverTerminationError("extracted coverage does not match assignments")

    min_rest = config.rest_intervals(grid)
    for agent in agents:
        assigned = sorted(
            assignments_by_agent[agent.id], key=lambda item: item.option.start_abs
        )
        start_days = [item.option.start_day for item in assigned]
        if len(start_days) != len(set(start_days)):
            raise SolverTerminationError("agent has more than one shift on a day")
        hours = sum(item.option.paid_hours for item in assigned)
        if hours > agent.max_weekly_hours + 1e-7:
            raise SolverTerminationError("agent exceeds the weekly hour ceiling")

        if len(assigned) > 1:
            for index, earlier in enumerate(assigned):
                later = assigned[(index + 1) % len(assigned)]
                if rest_intervals(earlier.option, later.option, grid) < min_rest:
                    raise SolverTerminationError("agent violates inter-shift rest")

        night_by_day = [0] * grid.days
        for item in assigned:
            if item.option.is_night:
                night_by_day[item.option.start_day] = 1
        window = config.max_consecutive_nights + 1
        if window > 1:
            for start_day in range(grid.days):
                count = sum(
                    night_by_day[(start_day + offset) % grid.days]
                    for offset in range(window)
                )
                if count > config.max_consecutive_nights:
                    raise SolverTerminationError("agent exceeds consecutive-night cap")
        if agent.kind == "full_time":
            night_count = sum(night_by_day)
            if night_count > config.full_time_night_max:
                raise SolverTerminationError("agent exceeds the weekly night maximum")


def _parse_cbc_log(log_text: str) -> tuple[str, float | None]:
    """Extract the termination reason and best bound from CBC's text log."""
    reason = "CBC completed"
    if "Stopped on time limit" in log_text:
        reason = "time limit"
    elif "within gap tolerance" in log_text:
        reason = "relative gap limit reached"
    elif "Optimal solution found" in log_text:
        reason = "optimal solution found"
    elif "Problem is infeasible" in log_text:
        reason = "infeasible"

    match = re.search(r"^Lower bound:\s*([-+0-9.eE]+)\s*$", log_text, re.MULTILINE)
    bound = float(match.group(1)) if match else None
    return reason, bound


def solve_weekly_schedule(
    agents: list[WeeklyAgent],
    options: list[WeeklyShiftOption],
    demand: pd.DataFrame,
    config: WeeklySolverConfig | None = None,
    grid: TimeGrid = DEFAULT_247_GRID,
    time_limit_sec: int = 180,
    gap: float = 0.03,
    msg: bool = False,
) -> WeeklySchedule:
    """Solve and return only a verified CBC allocation.

    Unlike the legacy demonstrator, ``Not Solved`` is never interpreted as a
    publishable roster without checking the solution state. A time-limited
    integer incumbent is returned as ``Feasible`` after independent invariant
    validation. Infeasible models, missing or fractional incumbents, and failed
    post-solve invariants raise :class:`SolverTerminationError`.
    """
    if time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive")
    if not 0.0 <= gap < 1.0:
        raise ValueError("gap must be in [0, 1)")
    config = config or WeeklySolverConfig()
    model, x, _, _, floor_under, short, night_short, required, floor = build_weekly_model(
        agents, options, demand, config, grid
    )
    option_by_id = {option.id: option for option in options}
    agent_by_id = {agent.id: agent for agent in agents}

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="wfm-cbc-") as temp_directory:
        log_path = Path(temp_directory) / "cbc.log"
        model.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                timeLimit=time_limit_sec,
                gapRel=gap,
                logPath=str(log_path),
            )
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if msg:
        print(log_text, end="")
    elapsed = time.perf_counter() - started
    solver_status = pulp.LpStatus[model.status]
    solution_status = pulp.LpSolution[getattr(model, "sol_status", 0)]
    termination_reason, objective_bound = _parse_cbc_log(log_text)

    if (
        model.sol_status == pulp.LpSolutionOptimal
        and termination_reason == "optimal solution found"
    ):
        status = "Optimal"
    elif model.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        status = "Feasible"
    else:
        raise SolverTerminationError(
            f"CBC returned solver status {solver_status!r} and solution status "
            f"{solution_status!r}; no schedule was published"
        )

    assignments: list[WeeklyAssignment] = []
    coverage = np.zeros(grid.total_intervals, dtype=int)
    agent_hours = {agent.id: 0.0 for agent in agents}
    for (agent_id, option_id), variable in x.items():
        value = variable.value()
        if value is None or not math.isclose(value, round(value), abs_tol=1e-7):
            raise SolverTerminationError("CBC returned an unverified fractional value")
        if round(value) == 1:
            option = option_by_id[option_id]
            assignments.append(
                WeeklyAssignment(
                    agent_id,
                    agent_by_id[agent_id].name,
                    option,
                )
            )
            agent_hours[agent_id] += option.paid_hours
            coverage[list(option.coverage_slots)] += 1

    shortfall = {
        agent_id: round(float(variable.value() or 0.0), 2)
        for agent_id, variable in short.items()
        if float(variable.value() or 0.0) > 1e-7
    }
    night_shortfall = {
        agent_id: int(round(float(variable.value() or 0.0)))
        for agent_id, variable in night_short.items()
        if float(variable.value() or 0.0) > 1e-7
    }
    # The floor slack is no longer cross-checkable against the allocation now
    # that coverage is one-sided, so the shortfall is derived from the solved
    # coverage directly. Deriving it from the assignments is strictly stronger
    # than trusting a solver-reported slack.
    actual_floor_shortfall = np.clip(floor - coverage, 0, None)

    objective = float(pulp.value(model.objective) or 0.0)
    if status == "Optimal":
        objective_bound = objective
        relative_gap_pct = 0.0
    elif objective_bound is not None:
        relative_gap_pct = max(
            0.0,
            100.0 * (objective - objective_bound) / max(abs(objective), 1e-12),
        )
    else:
        relative_gap_pct = None
    schedule = WeeklySchedule(
        assignments=assignments,
        coverage=coverage,
        required=required,
        presence_floor=floor,
        agent_hours=agent_hours,
        status=status,
        solver_status=solver_status,
        termination_reason=termination_reason,
        objective=objective,
        objective_bound=objective_bound,
        relative_gap_pct=relative_gap_pct,
        solve_seconds=elapsed,
        variables=len(x),
        total_variables=model.numVariables(),
        constraints=model.numConstraints(),
        shortfall_hours=shortfall,
        night_shortfall=night_shortfall,
    )
    _validate_schedule(schedule, agents, config, grid)
    return schedule
