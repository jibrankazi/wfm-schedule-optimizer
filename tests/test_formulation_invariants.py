"""Guards on the solver formulation itself.

The v0.5.0 build shipped with a two-sided coverage equality that had already
been measured as defective: its root LP did not converge in 180s under either
CBC or HiGHS, and the integer incumbent was 43,988 against 29,900 for the
one-sided form. Nothing in a 197-test suite caught the regression, because
every test covered a feature and none covered the formulation.

These do. They are cheap and they fail loudly if the equality ever returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
import pytest

from src.continuous import (
    DEFAULT_247_GRID,
    build_weekly_shift_options,
    generate_continuous_demand,
    generate_continuous_roster,
)
from src.optimizer_247 import build_weekly_model


@pytest.fixture(scope="module")
def model_parts():
    agents = generate_continuous_roster()
    options = build_weekly_shift_options()
    demand = generate_continuous_demand(seed=42, peak_calls=70.0)
    return build_weekly_model(agents, options, demand)


def test_coverage_rows_are_one_sided_inequalities(model_parts):
    """No coverage row may be an equality, and none may carry a surplus term.

    A two-sided equality lets any epsilon shift between overlapping templates
    be absorbed by the two slacks at zero objective change. That neutral-pivot
    plateau is what stopped the root LP converging.
    """
    model = model_parts[0]
    coverage_rows = [c for name, c in model.constraints.items() if name.startswith("coverage_")]
    assert coverage_rows, "expected one coverage row per interval"
    for row in coverage_rows:
        assert row.sense == pulp.LpConstraintGE, "coverage must be >=, never an equality"


def test_surplus_variables_are_out_of_the_active_matrix(model_parts):
    """Overstaffing is an affine projection of the solved coverage, not a column."""
    over = model_parts[3]
    assert over, "the surplus dictionary is retained for API compatibility"
    for surplus in over.values():
        assert surplus.upBound == 0


def test_surplus_does_not_appear_in_the_objective(model_parts):
    model, _x, _under, over = model_parts[0], model_parts[1], model_parts[2], model_parts[3]
    objective_vars = set(model.objective.keys())
    assert not objective_vars.intersection(over.values())


def test_presence_floor_keeps_its_own_row(model_parts):
    """Dropping it is safe for the feasible region but only while u_t = 0.

    Once a deficit is accepted, nothing stops coverage falling through the
    floor and the breach is priced at the understaffing weight instead of the
    floor weight. Measured, that discount took breaches from 4 to 51.
    """
    model, floor = model_parts[0], model_parts[8]
    rows = [n for n in model.constraints if n.startswith("presence_floor_")]
    expected = sum(1 for slot in range(DEFAULT_247_GRID.total_intervals) if floor[slot] > 0)
    assert len(rows) == expected


def test_floor_never_exceeds_required_in_the_shipped_profile(model_parts):
    required, floor = model_parts[7], model_parts[8]
    assert (np.asarray(floor) <= np.asarray(required)).all()


def test_a_profile_that_breaks_the_invariant_is_rejected():
    """The floor row depends on required >= floor. Assert it, do not trust it.

    Demand validation already rejects this upstream; the assertion inside the
    model builder is a second line of defence for callers that construct the
    requirement and floor arrays directly. Either message is acceptable - what
    matters is that the model is never built with the invariant broken.
    """
    grid = DEFAULT_247_GRID
    agents = generate_continuous_roster(full_time=2, part_time=1)
    options = build_weekly_shift_options()
    demand = pd.DataFrame(
        {
            "week_interval": range(grid.total_intervals),
            "required_agents": [1] * grid.total_intervals,
            "min_presence_floor": [5] * grid.total_intervals,  # floor above requirement
        }
    )
    with pytest.raises(ValueError, match="floor|Profile invariant violated"):
        build_weekly_model(agents, options, demand)
