"""Only synthetic fixtures here; no City roster, leave or Cisco records."""
from datetime import datetime, timezone
import pandas as pd
import pytest

from src.authorized_validation import planning_week, validate_operational_week


def inputs(tmp_path):
    roster = tmp_path / "roster.csv"
    leave = tmp_path / "leave.csv"
    cuic = tmp_path / "cuic.csv"
    assignments = tmp_path / "assignments.csv"
    pd.DataFrame([{"id": 1, "name": "fixture", "kind": "full_time",
                   "min_weekly_hours": 0, "max_weekly_hours": 40}]).to_csv(roster, index=False)
    pd.DataFrame([{"agent_id": 1, "start_at": "2026-10-13T00:00:00-04:00",
                   "end_at": "2026-10-14T00:00:00-04:00",
                   "approval_status": "PENDING"}]).to_csv(leave, index=False)
    pd.DataFrame([{"SESSIONID": "A", "SESSIONSEQ": "1",
                   "CallClock": "2026-10-12 16:03:00"},
                  {"SESSIONID": "B", "SESSIONSEQ": "1",
                   "CallClock": "2026-10-12 16:13:00"}]).to_csv(cuic, index=False)
    pd.DataFrame([{"agent_id": 1, "shift_id": "Mon_FT8_1200"}]).to_csv(assignments, index=False)
    return roster, leave, cuic, assignments


def run(files):
    roster, leave, cuic, assignments = files
    return validate_operational_week(roster_csv=roster, leave_csv=leave,
        cuic_csv=cuic, assignments_csv=assignments, week_start_local="2026-10-12",
        timestamp_column="CallClock", source_clock="UTC")


def test_authorized_four_source_shape_and_observed_calls(tmp_path):
    out = run(inputs(tmp_path))
    assert out["observed_call_units"] == 2
    assert out["employee_count"] == 1
    assert out["scheduled_shifts"] == 1
    assert out["on_phone_slots_with_zero_coverage_and_observed_calls"] == 0
    assert not out["forecast_validated"]
    assert not out["cisco_and_hr_data_provenance_independently_verified"]


def test_approved_leave_blocks_shift_pending_does_not(tmp_path):
    files = inputs(tmp_path)
    leave = files[1]
    pd.DataFrame([{"agent_id": 1, "start_at": "2026-10-12T12:00:00-04:00",
                   "end_at": "2026-10-12T13:00:00-04:00",
                   "approval_status": "APPROVED"}]).to_csv(leave, index=False)
    with pytest.raises(ValueError, match="approved leave"):
        run(files)


def test_invalid_clock_leave_and_unknown_employee_rejected(tmp_path):
    files = inputs(tmp_path)
    leave = files[1]
    pd.DataFrame([{"agent_id": 99, "start_at": "2026-10-12T12:00:00-04:00",
                   "end_at": "2026-10-12T13:00:00-04:00",
                   "approval_status": "APPROVED"}]).to_csv(leave, index=False)
    with pytest.raises(ValueError, match="absent"):
        run(files)
    pd.DataFrame([{"agent_id": 1, "start_at": "2026-10-12T12:00:00",
                   "end_at": "2026-10-12T13:00:00",
                   "approval_status": "APPROVED"}]).to_csv(leave, index=False)
    with pytest.raises(ValueError, match="UTC offset"):
        run(files)


def test_dst_transition_weeks_fail_closed():
    with pytest.raises(ValueError, match="DST"):
        planning_week("2026-10-26")
    with pytest.raises(ValueError, match="DST"):
        planning_week("2026-03-02")
    assert planning_week("2026-10-12")[0] == datetime(2026, 10, 12, 4, tzinfo=timezone.utc)
