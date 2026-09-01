import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from pathlib import Path

# Create output directories
Path("assets").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

np.random.seed(42)

# --- 1. Erlang C Functions ---
def erlang_c_pw(N: int, A: float) -> float:
    if N <= A:
        return 1.0
    sum_terms = sum((A**k) / math.factorial(k) for k in range(N))
    last_term = ((A**N) / math.factorial(N)) * (N / (N - A))
    return last_term / (sum_terms + last_term)

def calculate_sl(N: int, arrival_rate: float, aht_sec: float, target_sec: float = 20.0) -> float:
    A = (arrival_rate * aht_sec) / 900.0
    if N <= A:
        return 0.0
    pw = erlang_c_pw(N, A)
    return 1.0 - (pw * math.exp(-(N - A) * (target_sec / aht_sec)))

def calculate_required_agents(arrival_rate: float, aht_sec: float, target_sl: float = 0.80, max_occ: float = 0.85) -> int:
    A = (arrival_rate * aht_sec) / 900.0
    if A <= 0:
        return 0
    N = math.ceil(A) + 1
    while True:
        sl = calculate_sl(N, arrival_rate, aht_sec)
        occ = A / N
        if sl >= target_sl and occ <= max_occ:
            return N
        N += 1

# --- 2. Workload & Agent Roster ---
def generate_workload(days=5, intervals_per_day=49):
    records = []
    for day in range(days):
        t = np.linspace(0, np.pi, intervals_per_day)
        base_calls = 42 * np.sin(t) + 14 * np.sin(2 * t) + 8
        noise = np.random.normal(0, 1.5, intervals_per_day)
        calls = np.clip(base_calls + noise, 0, None).round()

        for interval in range(intervals_per_day):
            vol = int(calls[interval])
            aht = int(np.random.normal(315, 20))
            req = calculate_required_agents(vol, aht, target_sl=0.80)
            records.append({"day": day, "interval": interval, "calls": vol, "aht": aht, "required_agents": req})
    return pd.DataFrame(records)

workload_df = generate_workload()
workload_df.to_csv("data/synthetic_demand.csv", index=False)

agents = []
for i in range(60):
    is_ft = (i < 45)
    agents.append({
        "id": f"AGT_{i+1:02d}",
        "full_time": is_ft,
        "min_hours": 37.5 if is_ft else 20.0,
        "max_hours": 40.0 if is_ft else 25.0
    })

# --- 3. Shift Templates ---
shift_templates = []
start_intervals = [0, 2, 4, 6, 8, 10, 12, 14]

for s_idx, start in enumerate(start_intervals):
    coverage = [0] * 49
    end = min(start + 34, 49)
    for t in range(start, end):
        if t in range(start + 16, start + 18):  # 30-min lunch
            coverage[t] = 0
        else:
            coverage[t] = 1

    shift_templates.append({
        "id": s_idx,
        "name": f"Shift_{7 + (start*15)//60:02d}_{(start*15)%60:02d}",
        "paid_hours": 7.5,
        "coverage": coverage
    })

# --- 4. PuLP MILP Solve ---
print("[OPTIMIZING] Formulating Mixed-Integer Linear Program...")
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
        prob += pulp.lpSum(x[a][d][s] * shift_templates[s]["coverage"][t] for a in agent_ids for s in template_ids) + understaff[d][t] - overstaff[d][t] == req

status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"[STATUS] Solve Complete: {pulp.LpStatus[status]}")

# --- 5. Export Hero Visual ---
solved_coverage = np.zeros((5, 49))
for d in days:
    for t in intervals:
        solved_coverage[d, t] = sum(pulp.value(x[a][d][s]) * shift_templates[s]["coverage"][t] for a in agent_ids for s in template_ids)

day_0 = workload_df[workload_df["day"] == 0]
time_labels = [f"{7 + (i * 15)//60:02d}:{(i * 15)%60:02d}" for i in range(49)]

plt.figure(figsize=(13, 5), dpi=300)
plt.plot(time_labels, day_0["required_agents"], label="Required Agents (Erlang C 80/20)", color="#D9534F", linewidth=2.5)
plt.bar(time_labels, solved_coverage[0], label="Optimized Scheduled Headcount (MILP)", color="#0275D8", alpha=0.6, width=0.75)
plt.xticks(time_labels[::4], rotation=45, ha="right", fontsize=9)
plt.xlabel("Operating Interval (15-min increments)", fontweight="bold")
plt.ylabel("Concurrent Agents", fontweight="bold")
plt.title("Workforce Scheduling Optimization: Erlang C Required vs. MILP Scheduled Staffing (Day 1)", fontweight="bold", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("assets/coverage_vs_requirement.png")
print("[OUTPUT] Hero visual exported to assets/coverage_vs_requirement.png")
