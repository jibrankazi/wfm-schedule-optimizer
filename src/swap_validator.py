"""Deterministic validation of one-for-one shift trades.

Closed-form constraint arithmetic. No solver is invoked, so a supervisor gets
an answer in microseconds rather than waiting on a re-solve - which is the
point: a full weekly solve takes minutes and cannot be run to approve a swap.

**Interval coverage is invariant under a genuine one-for-one trade.** Both
shift envelopes remain staffed; only the names change. So no swap can breach a
presence floor or alter interval headcount, and this module deliberately does
not check for it. What a swap *can* break is per-agent: contract eligibility,
the one-shift-per-day rule, the weekly hour ceiling, and the turnaround rest
window between consecutive shifts. Those four are the whole surface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

INTERVALS_PER_DAY = 96
DEFAULT_WEEKLY_MAX_HOURS = 40.0
DEFAULT_MIN_REST_INTERVALS = 40  # 10 hours at 15-minute resolution


@dataclass(frozen=True)
class ShiftAssignment:
    """One agent's shift on one day, in whole 15-minute intervals."""

    agent_id: str
    day: int
    start_interval: int
    duration_intervals: int
    paid_hours: float
    contract_kind: str

    def __post_init__(self) -> None:
        if not 0 <= self.day <= 6:
            raise ValueError("day must be 0-6")
        if not 0 <= self.start_interval < INTERVALS_PER_DAY:
            raise ValueError("start_interval must be within the day")
        if self.duration_intervals <= 0:
            raise ValueError("duration_intervals must be positive")
        if self.paid_hours <= 0:
            raise ValueError("paid_hours must be positive")
        if self.contract_kind not in {"FT", "PT"}:
            raise ValueError("contract_kind must be 'FT' or 'PT'")

    @property
    def global_start(self) -> int:
        return self.day * INTERVALS_PER_DAY + self.start_interval

    @property
    def global_end(self) -> int:
        return self.global_start + self.duration_intervals


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: list[str]


def _check_agent_schedule(
    agent_id: str,
    assignments: Sequence[ShiftAssignment],
    weekly_max_hours: float,
    min_rest_intervals: int,
) -> list[str]:
    """Every rule a single agent's proposed week must satisfy."""
    reasons: list[str] = []

    for day, count in sorted(Counter(s.day for s in assignments).items()):
        if count > 1:
            reasons.append(
                f"Agent {agent_id} assigned multiple shifts on day {day} ({count} shifts)."
            )

    total_hours = sum(s.paid_hours for s in assignments)
    if total_hours > weekly_max_hours + 1e-9:
        reasons.append(
            f"Agent {agent_id} would exceed weekly cap: "
            f"{total_hours:.1f}h > {weekly_max_hours:.1f}h."
        )

    ordered = sorted(assignments, key=lambda s: s.global_start)
    pairs = list(zip(ordered, ordered[1:]))
    if ordered:
        pairs.append((ordered[-1], ordered[0]))
    for index, (current, following) in enumerate(pairs):
        next_start = following.global_start
        if index == len(pairs) - 1:
            next_start += 7 * INTERVALS_PER_DAY
        gap = next_start - current.global_end
        if gap < 0:
            reasons.append(
                f"Agent {agent_id} has overlapping shifts: day {current.day} "
                f"and day {following.day} overlap by {-gap * 15} min."
            )
        elif gap < min_rest_intervals:
            reasons.append(
                f"Agent {agent_id} violates turnaround rest between day {current.day} "
                f"and day {following.day}: {(gap * 15) / 60.0:.2f}h provided, "
                f"{(min_rest_intervals * 15) / 60.0:.1f}h required."
            )

    return reasons


def _swapped_in(
    roster: Sequence[ShiftAssignment],
    give_up: ShiftAssignment,
    take_on: ShiftAssignment,
) -> list[ShiftAssignment]:
    """The roster with `give_up` replaced by `take_on`'s time envelope.

    The agent keeps their own id and contract; only the shift's placement and
    paid hours move across.
    """
    replacement = ShiftAssignment(
        agent_id=give_up.agent_id,
        day=take_on.day,
        start_interval=take_on.start_interval,
        duration_intervals=take_on.duration_intervals,
        paid_hours=take_on.paid_hours,
        contract_kind=give_up.contract_kind,
    )
    return [replacement if shift == give_up else shift for shift in roster]


def validate_shift_swap(
    agent_a_roster: Sequence[ShiftAssignment],
    agent_b_roster: Sequence[ShiftAssignment],
    shift_a: ShiftAssignment,
    shift_b: ShiftAssignment,
    weekly_max_hours: float = DEFAULT_WEEKLY_MAX_HOURS,
    min_rest_intervals: int = DEFAULT_MIN_REST_INTERVALS,
) -> ValidationResult:
    """Decide whether agent A's `shift_a` may be traded for agent B's `shift_b`.

    Every rejection carries a reason naming the agent, the day and the
    quantity, so a supervisor is told what to change rather than only that the
    answer was no.
    """
    if shift_a.agent_id == shift_b.agent_id:
        return ValidationResult(
            False,
            [f"Self-swap rejected: both shifts belong to Agent {shift_a.agent_id}."],
        )
    if shift_a not in agent_a_roster:
        return ValidationResult(
            False,
            [f"Shift on day {shift_a.day} is not assigned to Agent {shift_a.agent_id}."],
        )
    if shift_b not in agent_b_roster:
        return ValidationResult(
            False,
            [f"Shift on day {shift_b.day} is not assigned to Agent {shift_b.agent_id}."],
        )

    for roster, selected in ((agent_a_roster, shift_a), (agent_b_roster, shift_b)):
        if any(s.agent_id != selected.agent_id for s in roster):
            return ValidationResult(False, [f"Roster contains shifts for another agent than {selected.agent_id}."])
        if len(set(roster)) != len(roster):
            return ValidationResult(False, [f"Roster for Agent {selected.agent_id} contains duplicate shifts."])

    reasons: list[str] = []
    if shift_a.contract_kind != shift_b.contract_kind:
        reasons.append(
            f"Contract mismatch: cannot swap {shift_a.contract_kind} shift with "
            f"{shift_b.contract_kind} shift."
        )

    reasons.extend(
        _check_agent_schedule(
            shift_a.agent_id,
            _swapped_in(agent_a_roster, shift_a, shift_b),
            weekly_max_hours,
            min_rest_intervals,
        )
    )
    reasons.extend(
        _check_agent_schedule(
            shift_b.agent_id,
            _swapped_in(agent_b_roster, shift_b, shift_a),
            weekly_max_hours,
            min_rest_intervals,
        )
    )

    return ValidationResult(is_valid=not reasons, reasons=reasons)
