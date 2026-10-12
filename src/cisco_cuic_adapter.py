"""Strict caller-supplied Cisco CUIC CSV -> observed 15-minute CALL volume.

No Cisco authentication, no assumed timestamp clock, no employee roster and
no promise of schedule optimization. Source timestamps require an explicit
published clock interpretation. Original session identities are never saved
in output; only counts and an input checksum are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_IDS = ("SESSIONID", "SESSIONSEQ")
ALLOWED_CLOCKS = ("UTC", "America/Toronto")


def aggregate_cuic_calls(
    path: str | Path,
    timestamp_column: str,
    source_clock: str,
    call_count_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fail closed on ambiguous phone-call timestamps and duplicate event keys."""
    if source_clock not in ALLOWED_CLOCKS:
        raise ValueError("Source clock must be explicitly UTC or America/Toronto")
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    mandatory = set(REQUIRED_IDS) | {timestamp_column}
    if call_count_column is not None:
        mandatory.add(call_count_column)
    missing = mandatory - set(frame.columns)
    if missing:
        raise ValueError(f"CUIC CSV missing explicitly required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("CUIC CSV contains no observed call rows")
    for identifier in REQUIRED_IDS:
        if frame[identifier].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing CUIC {identifier}")
    if frame[list(REQUIRED_IDS)].duplicated().any():
        raise ValueError("Duplicate SESSIONID+SESSIONSEQ; cannot silently double-count calls")
    if frame[timestamp_column].astype(str).str.strip().eq("").any():
        raise ValueError("Blank call timestamp")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    if timestamps.dt.tz is not None:
        raise ValueError("Timestamp already has an offset; provide a naive source-local clock column")
    # Toronto's fall-back hour is ambiguous; refuse rather than guessing.
    localized = timestamps.dt.tz_localize(
        source_clock, ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")
    if call_count_column is None:
        counts = pd.Series(1, index=frame.index, dtype="int64")
        definition = "One presented-call event per unique SESSIONID+SESSIONSEQ row"
    else:
        numeric = pd.to_numeric(frame[call_count_column], errors="raise")
        if numeric.isna().any() or numeric.lt(0).any() or numeric.mod(1).ne(0).any():
            raise ValueError("Call-count field must contain nonnegative whole numbers")
        if numeric.gt(100000).any():
            raise ValueError("Implausibly large call-count value")
        counts = numeric.astype("int64")
        definition = f"Sum of caller-declared {call_count_column} field (not per-event default)"
    intervals = localized.dt.floor("15min")
    grouped = (
        pd.DataFrame({"start": intervals, "calls": counts})
        .groupby("start", sort=True)
        .agg(observed_calls=("calls", "sum"), source_records=("calls", "size"))
        .reset_index()
    )
    grouped["interval_start_utc"] = grouped["start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    grouped["interval_start_toronto"] = grouped["start"].dt.tz_convert(
        "America/Toronto"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    output = grouped[[
        "interval_start_utc", "interval_start_toronto",
        "observed_calls", "source_records",
    ]].copy()
    if int(output.observed_calls.sum()) != int(counts.sum()):
        raise AssertionError("Observed call count was lost during interval aggregation")
    report: dict[str, object] = {
        "original_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "caller_declared_source_clock": source_clock,
        "timestamp_column": timestamp_column,
        "call_count_definition": definition,
        "observed_unique_call_keys": int(len(frame)),
        "observed_call_units": int(counts.sum()),
        "actual_intervals_with_source_records": int(len(output)),
        "output_contains_agent_or_call_session_identifiers": False,
        "cisco_origin_and_clock_independently_authenticated": False,
        "staff_schedule_or_service_level_verified": False,
        "note": "Actual call-record authenticity and timestamp clock must be verified against an authorized Cisco CUIC export; no gaps are backfilled, no staff requirements are inferred.",
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local authorized Cisco CUIC export")
    parser.add_argument("--timestamp-column", required=True,
                        help="Exact timestamp header from the CUIC export")
    parser.add_argument("--source-clock", required=True, choices=ALLOWED_CLOCKS,
                        help="Confirm from Cisco CUIC, never guess whether clock is UTC/local")
    parser.add_argument("--count-column", default=None,
                        help="Optional explicit aggregate call-count header")
    parser.add_argument("--output", default="results/cisco_cuic_observed_intervals.csv")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.resolve() == Path(args.input).resolve():
        raise ValueError("Refusing to overwrite the original Cisco CUIC report")
    if destination.exists():
        raise FileExistsError("Output already exists; choose a new evidence path")
    intervals, report = aggregate_cuic_calls(
        args.input, args.timestamp_column, args.source_clock, args.count_column
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(destination, index=False)
    summary = destination.with_suffix(".audit.json")
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
