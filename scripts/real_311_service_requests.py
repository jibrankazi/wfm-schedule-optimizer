"""Validate real 2025 Toronto 311 service-request arrival patterns.

Not a phone-call workload, no AHT, no employee availability: therefore *not*
a staffing requirement, optimizer performance, or deployable workforce plan.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

RAW_2025 = "https://raw.githubusercontent.com/jibrankazi/data-analytics-portfolio/main/toronto-311-analysis/data/SR2025.csv"
CITY_CATALOGUE = "https://open.toronto.ca/dataset/311-service-requests-customer-initiated/"


def analyze(path=RAW_2025):
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    if "Creation Date" not in df or "Service Request Type" not in df:
        raise ValueError("Missing official Toronto 311 service-request fields")
    dt = pd.to_datetime(df["Creation Date"], errors="coerce")
    if dt.isna().any():
        raise ValueError("Invalid/missing 311 created-at timestamps")
    valid = dt.loc[dt.dt.year.eq(2025)]
    if len(valid) < 10000:
        raise ValueError("Insufficient real 2025 service requests")
    intervals = valid.dt.floor("15min").value_counts().sort_index()
    monthly = valid.groupby(valid.dt.month).size()
    weekdays = valid.groupby(valid.dt.dayofweek).size()
    return {
        "original_source": CITY_CATALOGUE,
        "actual_input": str(path),
        "measurement": "All-channel 311 service-request creation timestamps, not telephone calls",
        "requests_in_2025": int(len(valid)),
        "peak_month": int(monthly.idxmax()),
        "peak_month_requests": int(monthly.max()),
        "unique_active_15min_intervals": int(len(intervals)),
        "highest_15min_requests": int(intervals.max()),
        "highest_15min_interval": str(intervals.idxmax()),
        "weekday_counts_mon_through_sun": [int(weekdays.get(i,0)) for i in range(7)],
        "cannot_infer_staffing": "No channel-level telephone arrival data, handling times, service goals, agent schedules or availability.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default=RAW_2025)
    p.add_argument("--output",default="results/real_311_demand_proxy.json")
    args=p.parse_args()
    result=analyze(args.input)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
