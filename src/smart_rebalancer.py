"""Deterministic intraday reassignment of offline duties to queue coverage.

The duty taxonomy is injected rather than hard-coded, so the same engine runs
against any operation's scheduling codes.
"""

from __future__ import annotations

import pandas as pd

from .profiles import DutyCodes

DEFAULT_DUTY_CODES = DutyCodes()


class AutonomousQueueRebalancer:
    """Pre-empt eligible offline work until an interval's deficit is closed.

    Pre-emption follows the order given in ``DutyCodes.preemptable``: the work
    an operation is most willing to interrupt comes first. Within a duty,
    agents are taken in column order, which keeps the result reproducible.
    """

    @classmethod
    def rebalance_interval(
        cls,
        schedule_matrix: pd.DataFrame,
        interval_idx: int,
        required_headcount: int,
        duty_codes: DutyCodes | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return a copied schedule and an itemized pre-emption log.

        ``interval_idx`` is positional. The row label is resolved before any
        ``DataFrame.at`` access so a non-default index behaves correctly.
        """
        codes = duty_codes or DEFAULT_DUTY_CODES

        if not isinstance(schedule_matrix, pd.DataFrame):
            raise TypeError("schedule_matrix must be a pandas DataFrame")
        if schedule_matrix.empty or schedule_matrix.shape[1] == 0:
            raise ValueError("schedule_matrix must contain intervals and agents")
        if not schedule_matrix.index.is_unique or not schedule_matrix.columns.is_unique:
            raise ValueError("schedule_matrix index and agent columns must be unique")
        if not 0 <= interval_idx < len(schedule_matrix):
            raise IndexError("interval_idx is outside the schedule matrix")
        if required_headcount < 0:
            raise ValueError("required_headcount must be non-negative")

        updated_grid = schedule_matrix.copy(deep=True)
        row_label = updated_grid.index[interval_idx]
        current_slot = updated_grid.iloc[interval_idx]

        on_queue = sum(codes.is_on_queue(value) for value in current_slot)
        deficit = required_headcount - on_queue
        logs: list[str] = []

        if deficit <= 0:
            return updated_grid, ["No deficit detected; capacity is at or above requirement."]

        for duty in codes.preemptable:
            if deficit <= 0:
                break
            eligible_agents = [
                agent
                for agent in updated_grid.columns
                if codes.normalise(updated_grid.at[row_label, agent]) == duty
            ]

            for agent in eligible_agents:
                if deficit <= 0:
                    break
                updated_grid.at[row_label, agent] = codes.primary_on_queue
                deficit -= 1
                logs.append(
                    f"Rebalance [interval {interval_idx}]: reallocated {agent} "
                    f"from '{duty}' to '{codes.primary_on_queue}' "
                    f"(deficit remaining: {deficit})"
                )

        if deficit > 0:
            logs.append(
                f"Unresolved deficit: {deficit} additional queue-qualified "
                "agent(s) are still required."
            )

        return updated_grid, logs

    @classmethod
    def rebalance_to_floor(
        cls,
        schedule_matrix: pd.DataFrame,
        interval_idx: int,
        required_headcount: int,
        min_presence_floor: int,
        duty_codes: DutyCodes | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Rebalance against the greater of the Erlang requirement and the floor.

        Overnight and low-volume intervals are the case this exists for: the
        queue may need nobody on arithmetic and still need somebody present.
        """
        if min_presence_floor < 0:
            raise ValueError("min_presence_floor must be non-negative")
        return cls.rebalance_interval(
            schedule_matrix,
            interval_idx,
            max(required_headcount, min_presence_floor),
            duty_codes,
        )
