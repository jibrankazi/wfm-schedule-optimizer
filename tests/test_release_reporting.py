"""Regression checks for solver status and published-result validation."""

import json

import pytest

from scripts import check_baseline
from src.optimizer_247 import _parse_cbc_log


def test_cbc_gap_termination_preserves_the_lower_bound():
    reason, bound = _parse_cbc_log(
        "Result - Optimal solution found (within gap tolerance)\n"
        "Objective value: 100.0\nLower bound: 98.0\n"
    )
    assert reason == "relative gap limit reached"
    assert bound == 98.0


@pytest.mark.parametrize("mutation", [
    {"status": "UNKNOWN"}, {"relative_gap_pct": float("nan")},
    {"objective_bound": 98.0}, {"floor_breach_intervals": None},
])
def test_result_guard_rejects_unverified_or_inconsistent_results(mutation):
    result = {
        "status": "Optimal", "objective": 100.0, "objective_bound": 100.0,
        "relative_gap_pct": 0.0, "floor_breach_intervals": 0,
        "coverage_compliance_pct": 100.0,
    }
    result.update(mutation)
    assert check_baseline._check("fixture", result, 0.0, 0)


def test_result_guard_requires_all_three_patterns_and_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr(check_baseline, "PATTERN_DIR", tmp_path)
    monkeypatch.setattr(check_baseline, "BASELINE", tmp_path / "missing.json")
    result = {
        "status": "Optimal", "objective": 0.0, "objective_bound": 0.0,
        "relative_gap_pct": 0.0, "floor_breach_intervals": 0,
        "coverage_compliance_pct": 100.0,
    }
    (tmp_path / "weekday_extended.json").write_text(
        json.dumps({"outcome": "solved", "result": result})
    )
    assert check_baseline.main() == 1
