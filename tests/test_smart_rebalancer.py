import pandas as pd
import pytest

from src.smart_rebalancer import AutonomousQueueRebalancer


def test_rebalancer_uses_priority_order_without_mutating_input():
    original = pd.DataFrame(
        {"A": ["X"], "B": ["WD"], "C": ["PI"], "D": ["DV"], "E": ["QA"]}
    )
    updated, logs = AutonomousQueueRebalancer.rebalance_interval(original, 0, 3)

    assert original.iloc[0].to_dict() == {"A": "X", "B": "WD", "C": "PI", "D": "DV", "E": "QA"}
    assert updated.iloc[0].to_dict() == {"A": "X", "B": "WD", "C": "PI", "D": "X", "E": "X"}
    assert "from 'QA'" in logs[0]
    assert "from 'DV'" in logs[1]


def test_rebalancer_handles_non_default_row_labels():
    schedule = pd.DataFrame({"A": ["QA", "X"], "B": ["WD", "QA"]}, index=["09:00", "09:15"])
    updated, _ = AutonomousQueueRebalancer.rebalance_interval(schedule, 0, 1)
    assert updated.at["09:00", "A"] == "X"
    assert updated.at["09:15", "A"] == "X"


def test_rebalancer_reports_unresolved_deficit():
    schedule = pd.DataFrame({"A": ["QA"], "B": ["BREAK"]})
    updated, logs = AutonomousQueueRebalancer.rebalance_interval(schedule, 0, 3)
    assert updated.at[0, "A"] == "X"
    assert logs[-1].startswith("Unresolved deficit: 2")


def test_rebalancer_noops_when_requirement_is_met():
    schedule = pd.DataFrame({"A": ["x"], "B": ["SS"], "C": ["QA"]})
    updated, logs = AutonomousQueueRebalancer.rebalance_interval(schedule, 0, 2)
    assert updated.equals(schedule)
    assert logs == ["No deficit detected; capacity is at or above requirement."]


def test_rebalancer_validates_bounds():
    schedule = pd.DataFrame({"A": ["QA"]})
    with pytest.raises(IndexError):
        AutonomousQueueRebalancer.rebalance_interval(schedule, 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        AutonomousQueueRebalancer.rebalance_interval(schedule, 0, -1)
