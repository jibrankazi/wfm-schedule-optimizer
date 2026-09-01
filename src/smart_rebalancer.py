"""Deterministic intraday reassignment of offline duties to queue coverage."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd


class AutonomousQueueRebalancer:
    """Pre-empt eligible offline work until an interval's deficit is closed."""

    PREEMPTION_PRIORITY: ClassVar[tuple[str, ...]] = ("QA", "DV", "PI", "WD")
    ON_QUEUE_CODES: ClassVar[frozenset[str]] = frozenset({"X", "SS", "OT"})

    @staticmethod
    def _code(value: object) -> str:
        return str(value).strip().upper()

    @classmethod
    def rebalance_interval(
        cls,
        schedule_matrix: pd.DataFrame,
        interval_idx: int,
        required_headcount: int,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return a copied schedule and an itemized pre-emption log.

        ``interval_idx`` is positional.  The implementation resolves the row
        label before using ``DataFrame.at`` so non-default indexes are handled
        correctly.
        """
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
        on_queue = sum(cls._code(value) in cls.ON_QUEUE_CODES for value in current_slot)
        deficit = required_headcount - on_queue
        logs: list[str] = []

        if deficit <= 0:
            return updated_grid, ["No deficit detected; capacity is at or above requirement."]

        for channel in cls.PREEMPTION_PRIORITY:
            if deficit <= 0:
                break
            eligible_agents = [
                agent
                for agent in updated_grid.columns
                if cls._code(updated_grid.at[row_label, agent]) == channel
            ]

            for agent in eligible_agents:
                if deficit <= 0:
                    break
                updated_grid.at[row_label, agent] = "X"
                deficit -= 1
                logs.append(
                    f"Autonomous Rebalance [Slot {interval_idx}]: Drafted {agent} "
                    f"from '{channel}' to 'ACTIVE_QUEUE' (deficit remaining: {deficit})"
                )

        if deficit > 0:
            logs.append(
                f"Unresolved deficit: {deficit} additional queue-qualified "
                "agent(s) are still required."
            )

        return updated_grid, logs
