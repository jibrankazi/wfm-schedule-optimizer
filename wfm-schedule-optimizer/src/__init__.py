"""Contact centre workforce scheduling: Erlang C sizing plus MILP assignment."""

from .erlang import ServiceTarget, erlang_c, required_agents, service_level
from .generator import (
    Agent,
    ShiftTemplate,
    build_shift_templates,
    capacity_report,
    generate_demand,
    generate_roster,
)
from .optimizer import Schedule, solve_schedule

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "Schedule",
    "ServiceTarget",
    "ShiftTemplate",
    "build_shift_templates",
    "capacity_report",
    "erlang_c",
    "generate_demand",
    "generate_roster",
    "required_agents",
    "service_level",
    "solve_schedule",
]
