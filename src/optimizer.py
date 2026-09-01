import pulp
import numpy as np
import pandas as pd

def build_shift_templates(start_intervals=None):
    if start_intervals is None:
        start_intervals = [0, 2, 4, 6, 8, 10, 12, 14]
    
    templates = []
    for s_idx, start in enumerate(start_intervals):
        coverage = [0] * 49
        end = min(start + 34, 49)
        for t in range(start, end):
            if t in range(start + 16, start + 18):  # 30-min unpaid meal break
                coverage[t] = 0
            else:
                coverage[t] = 1

        templates.append({
            "id": s_idx,
            "name": f"Shift_{7 + (start*15)//60:02d}_{(start*15)%60:02d}",
            "paid_hours": 7.5,
            "coverage": coverage
        })
    return templates

def solve_schedule(agents, shift_templates, workload_df):
    prob = pulp.LpProblem("WFM_Schedule_Optimizer", pulp.LpMinimize)
    agent_ids = [a["id"] for a in agents]
    days = list(range(5))
    template_ids = [s["id"] for s in shift_templates]
    intervals = list(range(49))

    x = pulp.LpVariable.dicts("x", (agent_ids, days, template_ids), cat=pulp.LpBinary)
    understaff = pulp.LpVariable.dicts("under", (days, intervals), lowBound=0)
    overstaff = pulp.LpVariable.dicts("over", (days, intervals), lowBound=0)

    prob += pulp.lpSum(100 * understaff[d][t] + 1 * overstaff[d][t] for d in days for t in intervals)

    for a in agent_ids:
        for d in days:
            prob += pulp.lpSum(x[a][d][s] for s in template_ids) <= 1

    for agent in agents:
        a_id = agent["id"]
        prob += pulp.lpSum(x[a_id][d][s] * shift_templates[s]["paid_hours"] for d in days for s in template_ids) <= agent["max_hours"]
        prob += pulp.lpSum(x[a_id][d][s] * shift_templates[s]["paid_hours"] for d in days for s in template_ids) >= agent["min_hours"]

    for d in days:
        for t in intervals:
            req = workload_df.loc[(workload_df["day"] == d) & (workload_df["interval"] == t), "required_agents"].values[0]
