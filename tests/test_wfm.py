import pytest
from src.erlang import calculate_required_agents, calculate_sl
from src.generator import generate_workload, generate_roster
from src.optimizer import build_shift_templates, solve_schedule

def test_erlang_c_scaling():
    assert calculate_required_agents(0, 300) == 0
    low_agents = calculate_required_agents(20, 300, target_sl=0.80)
    high_agents = calculate_required_agents(50, 300, target_sl=0.80)
    assert high_agents > low_agents

def test_roster_composition():
    roster = generate_roster(num_agents=60, ft_ratio=0.75)
    assert len(roster) == 60
    ft_count = sum(1 for a in roster if a['full_time'])
    assert ft_count == 45

def test_optimization_solve():
    workload = generate_workload(days=1, intervals_per_day=49)
    roster = generate_roster(num_agents=30, ft_ratio=1.0)
    for a in roster:
        a['min_hours'] = 0
        a['max_hours'] = 7.5
    templates = build_shift_templates()
    status, _, _ = solve_schedule(roster, templates, workload)
    assert status == 1
