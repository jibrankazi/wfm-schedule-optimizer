import pandas as pd
import pytest

from src.profiles import DutyCodes
from src.smart_rebalancer import AutonomousQueueRebalancer as Rebalancer


def test_original_frame_is_not_mutated():
    original = pd.DataFrame(
        {"A": ["QUEUE"], "B": ["TRAINING"], "C": ["CORRESPONDENCE"], "D": ["PROJECT"], "E": ["QUALITY"]}
    )
    updated, _ = Rebalancer.rebalance_interval(original, 0, 3)
    assert original.iloc[0].to_dict() == {
        "A": "QUEUE", "B": "TRAINING", "C": "CORRESPONDENCE", "D": "PROJECT", "E": "QUALITY"
    }
    assert updated.iloc[0].to_dict() != original.iloc[0].to_dict()


def test_preemption_follows_the_configured_priority_order():
    grid = pd.DataFrame({"A": ["QUALITY"], "B": ["PROJECT"], "C": ["CORRESPONDENCE"], "D": ["TRAINING"]})
    updated, _ = Rebalancer.rebalance_interval(grid, 0, 2)
    # QUALITY and PROJECT come first in the default priority.
    assert updated.at[0, "A"] == "QUEUE"
    assert updated.at[0, "B"] == "QUEUE"
    assert updated.at[0, "C"] == "CORRESPONDENCE"
    assert updated.at[0, "D"] == "TRAINING"


def test_a_custom_taxonomy_is_honoured():
    codes = DutyCodes(
        primary_on_queue="CALL_TAKER",
        on_queue=frozenset({"CALL_TAKER", "RADIO"}),
        preemptable=("TRAINING_OFFLINE", "ADMIN_TASK"),
    )
    grid = pd.DataFrame(
        {"AGT_01": ["TRAINING_OFFLINE"], "AGT_02": ["ADMIN_TASK"], "AGT_03": ["MEAL"]}
    )
    updated, logs = Rebalancer.rebalance_interval(grid, 0, 2, duty_codes=codes)
    assert updated.at[0, "AGT_01"] == "CALL_TAKER"
    assert updated.at[0, "AGT_02"] == "CALL_TAKER"
    assert updated.at[0, "AGT_03"] == "MEAL"
    assert len(logs) == 2


def test_drafted_agents_always_get_the_primary_code():
    """Guards the frozenset-ordering trap: never 'OVERTIME' by accident."""
    codes = DutyCodes()
    grid = pd.DataFrame({"A": ["QUALITY"], "B": ["PROJECT"]})
    updated, _ = Rebalancer.rebalance_interval(grid, 0, 2, duty_codes=codes)
    assert set(updated.iloc[0]) == {codes.primary_on_queue}


def test_non_default_index_is_handled_positionally():
    grid = pd.DataFrame({"A": ["QUALITY", "QUEUE"], "B": ["TRAINING", "QUALITY"]},
                        index=["09:00", "09:15"])
    updated, _ = Rebalancer.rebalance_interval(grid, 1, 2)
    assert updated.at["09:15", "B"] == "QUEUE"
    assert updated.at["09:00", "A"] == "QUALITY"


def test_codes_outside_the_taxonomy_are_left_alone():
    grid = pd.DataFrame({"A": ["QUALITY"], "B": ["SICK"], "C": ["MEAL"]})
    updated, logs = Rebalancer.rebalance_interval(grid, 0, 3)
    assert updated.at[0, "B"] == "SICK"
    assert updated.at[0, "C"] == "MEAL"
    assert any("Unresolved deficit: 2" in line for line in logs)


def test_case_and_whitespace_are_tolerated():
    grid = pd.DataFrame({"A": [" queue "], "B": ["OverTime"], "C": ["quality"]})
    updated, logs = Rebalancer.rebalance_interval(grid, 0, 2)
    assert logs == ["No deficit detected; capacity is at or above requirement."]
    assert updated.at[0, "C"] == "quality"


def test_no_deficit_returns_an_unchanged_grid():
    grid = pd.DataFrame({"A": ["QUEUE"], "B": ["QUALITY"]})
    updated, logs = Rebalancer.rebalance_interval(grid, 0, 1)
    assert updated.equals(grid)
    assert logs == ["No deficit detected; capacity is at or above requirement."]


def test_presence_floor_staffs_an_interval_erlang_would_leave_empty():
    """Overnight is a floor problem, not a queueing one."""
    grid = pd.DataFrame({"A": ["QUALITY"], "B": ["PROJECT"], "C": ["MEAL"]})
    updated, _ = Rebalancer.rebalance_to_floor(grid, 0, required_headcount=0, min_presence_floor=2)
    assert sum(v == "QUEUE" for v in updated.iloc[0]) == 2


def test_floor_below_requirement_does_not_reduce_staffing():
    grid = pd.DataFrame({"A": ["QUALITY"], "B": ["PROJECT"], "C": ["TRAINING"]})
    updated, _ = Rebalancer.rebalance_to_floor(grid, 0, required_headcount=3, min_presence_floor=1)
    assert sum(v == "QUEUE" for v in updated.iloc[0]) == 3


@pytest.mark.parametrize(
    "frame,index,message",
    [
        (pd.DataFrame(), 0, "must contain intervals"),
        (pd.DataFrame({"A": ["QUEUE"]}), 5, "outside the schedule matrix"),
        (pd.DataFrame({"A": ["QUEUE"]}), -1, "outside the schedule matrix"),
    ],
)
def test_invalid_input_is_rejected(frame, index, message):
    with pytest.raises((ValueError, IndexError), match=message):
        Rebalancer.rebalance_interval(frame, index, 1)


def test_duplicate_agent_columns_are_rejected():
    grid = pd.DataFrame([["QUEUE", "QUALITY"]], columns=["A", "A"])
    with pytest.raises(ValueError, match="must be unique"):
        Rebalancer.rebalance_interval(grid, 0, 2)


def test_negative_headcount_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Rebalancer.rebalance_interval(pd.DataFrame({"A": ["QUEUE"]}), 0, -1)
