import numpy as np
import pandas as pd
from src.erlang import calculate_required_agents

def generate_workload(days: int = 5, intervals_per_day: int = 49, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
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
            records.append({'day': day, 'interval': interval, 'calls': vol, 'aht': aht, 'required_agents': req})
    return pd.DataFrame(records)

def generate_roster(num_agents: int = 60, ft_ratio: float = 0.75):
    agents = []
    num_ft = int(num_agents * ft_ratio)
    for i in range(num_agents):
        is_ft = (i < num_ft)
        agents.append({
            'id': f'AGT_{i+1:02d}',
            'full_time': is_ft,
            'min_hours': 37.5 if is_ft else 20.0,
            'max_hours': 40.0 if is_ft else 25.0
        })
    return agents
