"""Software-only CUIC fixture records; no employee or City Cisco data."""
from pathlib import Path

import pandas as pd
import pytest

from src.cisco_cuic_adapter import aggregate_cuic_calls


def make_csv(tmp_path: Path, timestamps: list[str]) -> Path:
    rows = [{
        "SESSIONID": "CALL-" + str(i),
        "SESSIONSEQ": "0001",
        "CallClock": t,
        "Calls Presented": "1",
    } for i, t in enumerate(timestamps)]
    file = tmp_path / "fake_cuic.csv"
    pd.DataFrame(rows).to_csv(file, index=False)
    return file


def test_realistic_clock_needs_explicit_zone_and_keeps_utc_and_local(tmp_path: Path) -> None:
    path = make_csv(tmp_path, ["2026-03-09 12:00:00", "2026-03-09 12:07:00",
                              "2026-03-09 12:15:00"])
    out, report = aggregate_cuic_calls(path, "CallClock", "UTC")
    assert out.observed_calls.tolist() == [2, 1]
    assert out.interval_start_utc.iloc[0] == "2026-03-09T12:00:00Z"
    assert out.interval_start_toronto.iloc[0] == "2026-03-09T08:00:00-0400"
    assert report["observed_unique_call_keys"] == 3
    assert report["cisco_origin_and_clock_independently_authenticated"] is False


def test_local_clock_not_assumed_utc(tmp_path: Path) -> None:
    path = make_csv(tmp_path, ["2026-01-08 10:00:00"])
    out, _ = aggregate_cuic_calls(path, "CallClock", "America/Toronto")
    assert out.interval_start_utc.iloc[0] == "2026-01-08T15:00:00Z"
    assert out.interval_start_toronto.iloc[0].endswith("-0500")


def test_dst_repeated_hour_and_nonexistent_spring_hour_are_refused(tmp_path: Path) -> None:
    for instant in ("2026-11-01 01:30:00", "2026-03-08 02:30:00"):
        path = make_csv(tmp_path, [instant])
        with pytest.raises(Exception, match="Ambiguous|NonExistent|ambiguous|nonexistent"):
            aggregate_cuic_calls(path, "CallClock", "America/Toronto")


def test_duplicate_session_pair_does_not_inflate_calls(tmp_path: Path) -> None:
    path = make_csv(tmp_path, ["2026-02-02 10:00:00", "2026-02-02 10:02:00"])
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, "SESSIONID"] = frame.loc[0, "SESSIONID"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate SESSIONID"):
        aggregate_cuic_calls(path, "CallClock", "UTC")


def test_explicit_count_field_is_not_automatically_assumed(tmp_path: Path) -> None:
    path = make_csv(tmp_path, ["2026-02-02 10:00:00", "2026-02-02 10:05:00"])
    frame = pd.read_csv(path)
    frame["Calls Presented"] = [4, 2]
    frame.to_csv(path, index=False)
    one_row_counts, _ = aggregate_cuic_calls(path, "CallClock", "UTC")
    assert one_row_counts.observed_calls.iloc[0] == 2
    supplied_count, report = aggregate_cuic_calls(
        path, "CallClock", "UTC", call_count_column="Calls Presented")
    assert supplied_count.observed_calls.iloc[0] == 6
    assert report["observed_call_units"] == 6


def test_invalid_or_unknown_source_clock_and_missing_columns_fail(tmp_path: Path) -> None:
    path = make_csv(tmp_path, ["2026-02-02 10:00:00"])
    with pytest.raises(ValueError, match="Source clock"):
        aggregate_cuic_calls(path, "CallClock", "EST")
    with pytest.raises(ValueError, match="missing explicitly"):
        aggregate_cuic_calls(path, "NotAnActualHeader", "UTC")
    frame = pd.read_csv(path)
    frame.loc[0, "Calls Presented"] = 1.5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="whole numbers"):
        aggregate_cuic_calls(path, "CallClock", "UTC", "Calls Presented")
