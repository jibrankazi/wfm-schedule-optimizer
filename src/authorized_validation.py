"""Join explicitly authorized roster, approved leave, CUIC demand and shift assignments.

This is an evidence-first schedule *validation*, not proof of forecast accuracy
or Cisco origin. A 672-slot planning grid has no DST-aware variable-length day,
so any week with an offset change is rejected rather than shifted silently.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .cisco_cuic_adapter import aggregate_cuic_calls
from .continuous import (
    DEFAULT_247_GRID, WeeklyAgent, build_weekly_shift_options, eligible_for_option,
)
from .csv_adapter import load_agents_csv

TORONTO = ZoneInfo("America/Toronto")
SLOTS = DEFAULT_247_GRID.total_intervals
STEP = timedelta(minutes=15)


def planning_week(date_text: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(date_text)
    except ValueError as exc:
        raise ValueError("week-start must be YYYY-MM-DD (Monday)") from exc
    if start.tzinfo is not None or start.time() != datetime.min.time() or start.weekday() != 0:
        raise ValueError("week-start must be a Toronto-local Monday date")
    local = start.replace(tzinfo=TORONTO)
    end = local + timedelta(days=7)
    if local.utcoffset() != (end - timedelta(seconds=1)).utcoffset():
        raise ValueError("DST transition week needs a variable-length planning grid; refused")
    return local.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _qualified_datetime(value: str, field: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime with UTC offset") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field} requires an explicit UTC offset to disambiguate DST")
    return dt.astimezone(timezone.utc)


def with_approved_leave(
    agents: list[WeeklyAgent], leave_csv: str | Path, week_start: datetime, week_end: datetime
) -> tuple[list[WeeklyAgent], dict]:
    path = Path(leave_csv)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"agent_id", "start_at", "end_at", "approval_status"}
    if not required.issubset(frame.columns):
        raise ValueError(f"approved leave requires columns {sorted(required)}")
    employee_ids = {a.id for a in agents}
    if len(employee_ids) != len(agents):
        raise ValueError("Roster contains duplicate agent IDs")
    forbidden: dict[int, set[int]] = {a.id: set(a.unavailable_slots) for a in agents}
    accepted = 0
    for record in frame.to_dict("records"):
        try:
            agent_id = int(record["agent_id"])
        except ValueError as exc:
            raise ValueError("Leave agent_id must be an integer") from exc
        if agent_id not in employee_ids:
            raise ValueError("Approved leave references an agent absent from supplied roster")
        status = record["approval_status"].strip().upper()
        if status not in {"APPROVED", "PENDING", "REJECTED", "CANCELLED"}:
            raise ValueError("Unknown leave approval state (must never be assumed approved)")
        start = _qualified_datetime(record["start_at"], "start_at")
        end = _qualified_datetime(record["end_at"], "end_at")
        if start >= end:
            raise ValueError("Leave end_at must be after start_at")
        if status != "APPROVED" or end <= week_start or start >= week_end:
            continue
        accepted += 1
        for slot in range(SLOTS):
            slot_start = week_start + slot * STEP
            if slot_start < end and slot_start + STEP > start:
                forbidden[agent_id].add(slot)
    patched = [replace(a, unavailable_slots=frozenset(forbidden[a.id])) for a in agents]
    return patched, {
        "leave_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "approved_leave_rows_overlapping_week": accepted,
        "agents_with_unavailable_slots": sum(bool(a.unavailable_slots) for a in patched),
    }


def validate_operational_week(
    *, roster_csv: str | Path, leave_csv: str | Path,
    cuic_csv: str | Path, assignments_csv: str | Path,
    week_start_local: str, timestamp_column: str, source_clock: str,
) -> dict:
    start, end = planning_week(week_start_local)
    agents = load_agents_csv(roster_csv)
    agents, leave_evidence = with_approved_leave(agents, leave_csv, start, end)
    agents_by_id = {a.id: a for a in agents}
    demand, source_report = aggregate_cuic_calls(cuic_csv, timestamp_column, source_clock)
    timestamps = pd.to_datetime(demand["interval_start_utc"], utc=True)
    if ((timestamps < pd.Timestamp(start)) | (timestamps >= pd.Timestamp(end))).any():
        raise ValueError("CUIC export includes observations outside selected planning week")
    observed = [0] * SLOTS
    for stamp, calls in zip(timestamps, demand["observed_calls"]):
        slot = int((stamp.to_pydatetime() - start) / STEP)
        observed[slot] += int(calls)
    schedule_path = Path(assignments_csv)
    schedule = pd.read_csv(schedule_path, dtype=str, keep_default_na=False)
    if not {"agent_id", "shift_id"}.issubset(schedule.columns):
        raise ValueError("Assignments require agent_id and shift_id columns")
    shifts = {option.id: option for option in build_weekly_shift_options()}
    on_phone = [0] * SLOTS
    assigned = set()
    paid_hours = {agent.id: 0.0 for agent in agents}
    occupied: dict[int, set[int]] = {agent.id: set() for agent in agents}
    for record in schedule.to_dict("records"):
        try:
            agent_id = int(record["agent_id"])
        except ValueError as exc:
            raise ValueError("Assignment agent_id is not an integer") from exc
        shift_id = record["shift_id"]
        if agent_id not in agents_by_id or shift_id not in shifts:
            raise ValueError("Assignment refers to unknown roster agent or shift option")
        if (agent_id, shift_id) in assigned:
            raise ValueError("Duplicate assignment row")
        assigned.add((agent_id, shift_id))
        option = shifts[shift_id]
        agent = agents_by_id[agent_id]
        if not eligible_for_option(agent, option):
            raise ValueError("Assignment violates kind, unavailability or approved leave")
        if occupied[agent_id].intersection(option.occupied_slots):
            raise ValueError("Employee has overlapping occupied shifts")
        occupied[agent_id].update(option.occupied_slots)
        paid_hours[agent_id] += option.paid_hours
        for slot in option.coverage_slots:
            on_phone[slot] += 1
    over_max = [id_ for id_, hours in paid_hours.items()
                if hours > agents_by_id[id_].max_weekly_hours + 1e-9]
    if over_max:
        raise ValueError("Assignment exceeds one or more roster max weekly hours")
    # Observed call volume and staffing are separate facts; 368s AHT alone is
    # insufficient to infer proven Erlang service-level outcomes or forecast skill.
    report = {
        "week_start_local": week_start_local,
        "planning_grid": "672 x 15-minute slots; DST transition weeks refused",
        "roster_sha256": hashlib.sha256(Path(roster_csv).read_bytes()).hexdigest(),
        "assignments_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "employee_count": len(agents),
        "scheduled_shifts": len(assigned),
        "observed_cuic_intervals": int(str(source_report["actual_intervals_with_source_records"])),
        "observed_call_units": int(sum(observed)),
        "on_phone_agent_slot_hours": round(sum(on_phone) * 0.25, 2),
        "on_phone_slots_with_zero_coverage_and_observed_calls":
            sum(1 for calls, staffed in zip(observed, on_phone) if calls > 0 and staffed == 0),
        "no_unauthorized_leave_or_availability_conflict": True,
        "no_overlapping_shifts": True,
        "source_evidence": source_report,
        "leave_evidence": leave_evidence,
        "official_service_level_80_percent_within_60_seconds_verified": False,
        "forecast_validated": False,
        "cisco_and_hr_data_provenance_independently_verified": False,
        "limitation": "Validates input structure and planned assignments only. No proof of real Cisco origin, approved HR truth, forecast skill, 80/60 SLA, or statutory policy compliance.",
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("roster-csv", "leave-csv", "cuic-csv", "assignments-csv",
                 "week-start-local", "timestamp-column", "source-clock"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    inputs = [Path(getattr(args, key.replace("-", "_"))) for key in
              ("roster-csv", "leave-csv", "cuic-csv", "assignments-csv")]
    if output.resolve() in {file.resolve() for file in inputs}:
        raise ValueError("Refusing to overwrite an input")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing validation audit")
    result = validate_operational_week(
        roster_csv=args.roster_csv, leave_csv=args.leave_csv,
        cuic_csv=args.cuic_csv, assignments_csv=args.assignments_csv,
        week_start_local=args.week_start_local, timestamp_column=args.timestamp_column,
        source_clock=args.source_clock,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
