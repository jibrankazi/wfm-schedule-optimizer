"""Compare CBC's unpriced surplus with the former priced-surplus variant.

The small fixture has similar bounds under both objectives. That is a sample
regression check, not an equivalence or general solver-performance proof.
"""

from __future__ import annotations

import pulp
import pytest

from src.continuous import (
    DEFAULT_247_GRID,
    build_weekly_shift_options,
    generate_continuous_demand,
    generate_continuous_roster,
)
from src.optimizer_247 import WeeklySolverConfig, build_weekly_model


def _two_sided(model, x, under, over, required, covering, overstaff_penalty: float) -> None:
    """Rebuild coverage the old way: equality with a priced surplus column.

    The surplus must be restored to the objective as well as the matrix.
    Leaving it free lets the relaxation buy coverage at zero cost, which is a
    defect of the reconstruction rather than of the original formulation - it
    produced a bound 2.6% below the shipped model on the first attempt.
    """
    for name in [c for c in list(model.constraints) if c.startswith("coverage_")]:
        del model.constraints[name]
    for slot in range(len(required)):
        model += (
            pulp.lpSum(x[k] for k in covering[slot]) + under[slot] - over[slot]
            == int(required[slot]),
            f"coverage_{slot}",
        )
    for surplus in over.values():
        surplus.upBound = None
    model.objective += overstaff_penalty * pulp.lpSum(over.values())


@pytest.fixture(scope="module")
def small_instance():
    """A reduced instance so the LP finishes in seconds rather than minutes."""
    agents = generate_continuous_roster(full_time=4, part_time=2)
    options = build_weekly_shift_options()
    demand = generate_continuous_demand(seed=42, peak_calls=6.0)
    return agents, options, demand


def _covering_map(x, options):
    import collections

    by_id = {o.id: o for o in options}
    covering = collections.defaultdict(list)
    for key in x:
        for slot in by_id[key[1]].coverage_slots:
            covering[slot].append(key)
    return covering


def test_the_shipped_formulation_is_one_sided(small_instance):
    """Guards the regression: an equality must never come back."""
    agents, options, demand = small_instance
    model, *_ = build_weekly_model(agents, options, demand)
    for name, row in model.constraints.items():
        if name.startswith("coverage_"):
            assert row.sense == pulp.LpConstraintGE, f"{name} is not a covering inequality"


def test_small_instance_objectives_remain_within_recorded_tolerance(small_instance):
    """This sample's bounds are close; the objectives are not equivalent."""
    agents, options, demand = small_instance

    bounds = {}
    for label in ("one_sided", "two_sided"):
        model, x, under, over, *rest = build_weekly_model(agents, options, demand)
        required = rest[3]
        if label == "two_sided":
            _two_sided(
                model, x, under, over, required, _covering_map(x, options),
                WeeklySolverConfig().overstaff_penalty,
            )
        for variable in x.values():
            variable.cat = pulp.LpContinuous
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
        assert pulp.LpStatus[model.status] == "Optimal", f"{label} LP did not converge"
        bounds[label] = pulp.value(model.objective)

    # A regression bound for this sample, not an equivalence proof.
    assert bounds["one_sided"] == pytest.approx(bounds["two_sided"], rel=0.02), bounds


def test_the_surplus_column_is_absent_from_the_shipped_matrix(small_instance):
    """672 continuous columns removed, and the objective no longer prices them."""
    agents, options, demand = small_instance
    model, x, under, over, *_ = build_weekly_model(agents, options, demand)
    assert all(surplus.upBound == 0 for surplus in over.values())
    assert not set(model.objective.keys()).intersection(over.values())
