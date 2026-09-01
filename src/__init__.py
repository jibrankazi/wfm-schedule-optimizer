"""Contact centre workforce scheduling and intraday operations."""

from .audit_vault import AuditBlock, SovereignAuditVault
from .erlang import ServiceTarget, erlang_c, required_agents, service_level
from .erlang_advanced import calculate_erlang_a_metrics, vectorized_erlang_c_headcount
from .generator import (
    Agent,
    ShiftTemplate,
    build_shift_templates,
    capacity_report,
    generate_demand,
    generate_roster,
)
from .optimizer import Schedule, solve_schedule
from .smart_rebalancer import AutonomousQueueRebalancer

__version__ = "0.2.0"
__all__ = [
    "Agent",
    "AuditBlock",
    "AutonomousQueueRebalancer",
    "Schedule",
    "ServiceTarget",
    "ShiftTemplate",
    "build_shift_templates",
    "capacity_report",
    "calculate_erlang_a_metrics",
    "erlang_c",
    "generate_demand",
    "generate_roster",
    "required_agents",
    "service_level",
    "solve_schedule",
    "SovereignAuditVault",
    "vectorized_erlang_c_headcount",
]
