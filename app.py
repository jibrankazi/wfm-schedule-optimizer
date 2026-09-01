"""Multi-persona Streamlit console for the WFM optimization engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.erlang_advanced import calculate_erlang_a_metrics, vectorized_erlang_c_headcount
from src.generator import generate_demand
from src.smart_rebalancer import AutonomousQueueRebalancer


st.set_page_config(page_title="WFM Operations Console", layout="wide")
st.title("WFM Operations Console")
st.caption("Synthetic planning data only — no employee or production call data is included.")

frontline, supervisor, executive = st.tabs(
    ["Frontline / Agent", "Watch Commander / Supervisor", "Executive Analytics"]
)

with frontline:
    st.subheader("Interval demand view")
    demand = generate_demand(seed=42)
    selected_day = st.selectbox("Day", sorted(demand["day_name"].unique()))
    day_demand = demand.loc[
        demand["day_name"] == selected_day,
        ["time", "calls", "aht_sec", "required_agents"],
    ].set_index("time")
    st.line_chart(day_demand[["calls", "required_agents"]])
    st.dataframe(day_demand, use_container_width=True)

with supervisor:
    st.subheader("Intraday queue rebalancer")
    sample = pd.DataFrame(
        {
            "AGT-001": ["X", "QA", "X"],
            "AGT-002": ["DV", "DV", "PI"],
            "AGT-003": ["WD", "X", "QA"],
            "AGT-004": ["PI", "WD", "X"],
        },
        index=["09:00", "09:15", "09:30"],
    )
    interval = st.selectbox("Interval", list(sample.index))
    required = st.number_input("Required queue headcount", min_value=0, value=3, step=1)
    st.dataframe(sample, use_container_width=True)
    if st.button("Rebalance selected interval"):
        position = list(sample.index).index(interval)
        updated, logs = AutonomousQueueRebalancer.rebalance_interval(
            sample, position, int(required)
        )
        st.dataframe(updated, use_container_width=True)
        for entry in logs:
            st.write(entry)

with executive:
    st.subheader("Queueing scenarios")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        calls = st.number_input("Calls per 15 minutes", min_value=0.0, value=45.0)
    with col2:
        aht = st.number_input("Mean handle time (seconds)", min_value=1.0, value=320.0)
    with col3:
        patience = st.number_input("Mean patience (seconds)", min_value=1.0, value=120.0)
    with col4:
        servers = st.number_input("Staffed agents", min_value=1, value=20, step=1)

    erlang_c_agents = int(
        vectorized_erlang_c_headcount(np.asarray([calls]), aht)[0]
    )
    erlang_a = calculate_erlang_a_metrics(calls / 900.0, aht, patience, int(servers))
    st.metric("Erlang C required agents", erlang_c_agents)
    st.json(erlang_a)

st.divider()
st.caption(
    "The audit hash chain is tamper-evident, not immutable storage or a compliance certification."
)
