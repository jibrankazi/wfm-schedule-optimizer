"""Fail the build if a published result regresses.

Deliberately strict about missing keys. The defect this guards against reached
a release because a check that could not measure something reported success; a
renamed or absent field must fail loudly, not default to zero.

Two things are guarded. The three operating patterns are the headline result
and each must still prove optimality. The older 24/7 baseline is kept as a
regression witness: it was solved on an instance sized at the edge of its own
ceiling, so its wide gap is expected and only a *worsening* matters.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PATTERN_DIR = Path("artifacts/patterns")
BASELINE = Path("artifacts/cbc_baseline_247.json")

MAX_PATTERN_GAP_PCT = 0.0        # patterns are expected to prove optimality
MAX_PATTERN_FLOOR_BREACHES = 0
MAX_BASELINE_GAP_PCT = 60.0      # the tuned-down legacy instance
MAX_BASELINE_FLOOR_BREACHES = 5

PATTERN_KEYS = ("weekday_extended", "seven_day_extended", "continuous_247")
REQUIRED = (
    "status", "objective", "objective_bound", "relative_gap_pct",
    "floor_breach_intervals", "coverage_compliance_pct",
)


def _check(name: str, result: dict, max_gap: float, max_breaches: int) -> list[str]:
    missing = [f for f in REQUIRED if f not in result]
    if missing:
        return [f"{name}: missing required fields {missing}"]
    problems = []
    for field in REQUIRED[1:]:
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"{name}: {field} must be a finite number")
    if problems:
        return problems
    if result["status"] not in {"Optimal", "Feasible"}:
        problems.append(f"{name}: invalid solve status {result['status']!r}")
    if max_gap == 0 and result["status"] != "Optimal":
        problems.append(f"{name}: optimality has not been established")
    objective, bound = result["objective"], result["objective_bound"]
    expected_gap = max(0.0, 100 * (objective - bound) / max(abs(objective), 1e-12))
    if bound > objective + 1e-6 or not math.isclose(result["relative_gap_pct"], expected_gap, abs_tol=0.001):
        problems.append(f"{name}: reported gap disagrees with objective and bound")
    if result["status"] == "Optimal" and not math.isclose(objective, bound, abs_tol=1e-6):
        problems.append(f"{name}: Optimal status has unequal objective and bound")
    if not 0 <= result["coverage_compliance_pct"] <= 100 or result["floor_breach_intervals"] < 0:
        problems.append(f"{name}: invalid coverage or floor-breach metric")
    if result["relative_gap_pct"] > max_gap:
        problems.append(
            f"{name}: gap {result['relative_gap_pct']}% exceeds {max_gap}%"
        )
    if result["floor_breach_intervals"] > max_breaches:
        problems.append(
            f"{name}: {result['floor_breach_intervals']} floor breaches "
            f"exceed {max_breaches}"
        )
    return problems


def main() -> int:
    failures: list[str] = []
    checked = 0

    if not PATTERN_DIR.exists():
        return _fail([f"pattern results missing: {PATTERN_DIR}"])
    for key in PATTERN_KEYS:
        path = PATTERN_DIR / f"{key}.json"
        if not path.exists():
            failures.append(f"pattern result missing: {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("outcome") != "solved":
            failures.append(f"{path.stem}: outcome is {record.get('outcome')!r}")
            continue
        failures += _check(
            path.stem, record.get("result", {}),
            MAX_PATTERN_GAP_PCT, MAX_PATTERN_FLOOR_BREACHES,
        )
        checked += 1

    if checked == 0:
        failures.append("no solved pattern results found")

    if not BASELINE.exists():
        failures.append(f"legacy baseline missing: {BASELINE}")
    else:
        measured = json.loads(BASELINE.read_text(encoding="utf-8")).get("measured_result")
        if measured is None:
            failures.append("cbc_baseline_247.json has no 'measured_result' block")
        else:
            failures += _check(
                "legacy_247_baseline", measured,
                MAX_BASELINE_GAP_PCT, MAX_BASELINE_FLOOR_BREACHES,
            )

    if failures:
        return _fail(failures)
    print(f"baseline OK: {checked} patterns proven optimal, legacy baseline within bounds")
    return 0


def _fail(problems: list[str]) -> int:
    sys.stderr.write("baseline regression:\n  - " + "\n  - ".join(problems) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
