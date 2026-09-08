"""FastAPI gateway for queue sizing, rebalancing, and audit verification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.audit_vault import SovereignAuditVault
from src.erlang_advanced import (
    calculate_erlang_a_metrics,
    erlang_a_service_level,
    required_agents_erlang_a,
    vectorized_erlang_c_headcount,
)
from src.profiles import DutyCodes
from src.swap_validator import ShiftAssignment, validate_shift_swap
from src.smart_rebalancer import AutonomousQueueRebalancer


class ErlangCRequest(BaseModel):
    arrival_rates: list[float] = Field(min_length=1)
    aht_seconds: float = Field(gt=0)
    interval_seconds: int = Field(default=900, gt=0)
    target_service_level: float = Field(default=0.80, ge=0, le=1)
    target_seconds: float = Field(default=20.0, ge=0)
    max_occupancy: float = Field(default=0.85, gt=0, le=1)


class ErlangARequest(BaseModel):
    arrival_rate_per_second: float = Field(ge=0)
    mean_service_time_seconds: float = Field(gt=0)
    mean_patience_seconds: float = Field(gt=0)
    num_servers: int = Field(gt=0)


class ErlangAHeadcountRequest(BaseModel):
    contacts_per_interval: float = Field(ge=0)
    aht_seconds: float = Field(gt=0)
    mean_patience_seconds: float = Field(gt=0)
    interval_seconds: int = Field(default=900, gt=0)
    target_service_level: float = Field(default=0.80, ge=0, le=1)
    target_seconds: float = Field(default=20.0, ge=0)
    max_abandonment_rate: float = Field(default=0.05, ge=0, le=1)
    max_occupancy: float = Field(default=0.85, gt=0, le=1)
    min_presence_floor: int = Field(default=1, ge=0)
    ceiling: int = Field(default=500, gt=0)


class DutyCodesRequest(BaseModel):
    """Optional per-request duty taxonomy; omitted means the generic default."""

    primary_on_queue: str = Field(min_length=1)
    on_queue: list[str] = Field(min_length=1)
    preemptable: list[str] = Field(min_length=1)

    def to_domain(self) -> DutyCodes:
        return DutyCodes(
            primary_on_queue=self.primary_on_queue.strip().upper(),
            on_queue=frozenset(code.strip().upper() for code in self.on_queue),
            preemptable=tuple(code.strip().upper() for code in self.preemptable),
        )


class RebalanceRequest(BaseModel):
    schedule_matrix: dict[str, list[str]] = Field(min_length=1)
    interval_idx: int = Field(ge=0)
    required_headcount: int = Field(ge=0)
    min_presence_floor: int = Field(default=0, ge=0)
    duty_codes: DutyCodesRequest | None = None
    actor: str = Field(default="API_SUPERVISOR", min_length=1)

    @model_validator(mode="after")
    def validate_schedule_shape(self) -> "RebalanceRequest":
        lengths = {len(values) for values in self.schedule_matrix.values()}
        if not lengths or 0 in lengths or len(lengths) != 1:
            raise ValueError("all schedule columns must have the same non-zero length")
        if self.interval_idx >= next(iter(lengths)):
            raise ValueError("interval_idx is outside the schedule matrix")
        return self


class AuditMutationRequest(BaseModel):
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target_agent: str = Field(min_length=1)
    delta: dict[str, Any]


def _build_vault() -> SovereignAuditVault:
    configured_path = os.getenv("AUDIT_VAULT_PATH")
    return SovereignAuditVault(Path(configured_path)) if configured_path else SovereignAuditVault()



class ShiftAssignmentSchema(BaseModel):
    """One agent's shift on one day, in whole 15-minute intervals."""

    agent_id: str = Field(min_length=1)
    day: int = Field(ge=0, le=6)
    start_interval: int = Field(ge=0, le=95)
    duration_intervals: int = Field(gt=0)
    paid_hours: float = Field(gt=0.0)
    contract_kind: str = Field(pattern="^(FT|PT)$")

    def to_domain(self) -> ShiftAssignment:
        return ShiftAssignment(
            agent_id=self.agent_id,
            day=self.day,
            start_interval=self.start_interval,
            duration_intervals=self.duration_intervals,
            paid_hours=self.paid_hours,
            contract_kind=self.contract_kind,
        )


class SwapValidationRequest(BaseModel):
    agent_a_roster: list[ShiftAssignmentSchema] = Field(min_length=1)
    agent_b_roster: list[ShiftAssignmentSchema] = Field(min_length=1)
    shift_a: ShiftAssignmentSchema
    shift_b: ShiftAssignmentSchema
    weekly_max_hours: float = Field(default=40.0, gt=0.0)
    min_rest_intervals: int = Field(default=40, ge=0)


class SwapValidationResponse(BaseModel):
    is_valid: bool
    reasons: list[str]


app = FastAPI(
    title="WFM Schedule Optimizer API",
    version="0.3.0",
    description="Queue sizing, intraday rebalancing, and tamper-evident audit endpoints.",
)
audit_vault = _build_vault()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "audit_chain_valid": audit_vault.verify_chain_integrity(),
        "audit_blocks": len(audit_vault.chain),
    }


@app.post("/v1/queueing/erlang-c/headcount")
def erlang_c_headcount(request: ErlangCRequest) -> dict[str, list[int]]:
    try:
        headcounts = vectorized_erlang_c_headcount(
            np.asarray(request.arrival_rates, dtype=float),
            request.aht_seconds,
            interval_sec=request.interval_seconds,
            target_sl=request.target_service_level,
            target_sec=request.target_seconds,
            max_occ=request.max_occupancy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"required_headcounts": headcounts.tolist()}


@app.post("/v1/queueing/erlang-a/metrics")
def erlang_a_metrics(request: ErlangARequest) -> dict[str, float]:
    try:
        return calculate_erlang_a_metrics(
            request.arrival_rate_per_second,
            request.mean_service_time_seconds,
            request.mean_patience_seconds,
            request.num_servers,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/queueing/erlang-a/headcount")
def erlang_a_headcount(request: ErlangAHeadcountRequest) -> dict[str, Any]:
    try:
        headcount = required_agents_erlang_a(
            arrival_rate=request.contacts_per_interval,
            aht_sec=request.aht_seconds,
            mean_patience_sec=request.mean_patience_seconds,
            interval_sec=request.interval_seconds,
            target_sl=request.target_service_level,
            target_sec=request.target_seconds,
            max_abandon_rate=request.max_abandonment_rate,
            max_occupancy=request.max_occupancy,
            min_presence_floor=request.min_presence_floor,
            ceiling=request.ceiling,
        )
        calls_per_second = request.contacts_per_interval / request.interval_seconds
        if request.contacts_per_interval == 0:
            metrics = {
                "service_level": 1.0,
                "abandonment_rate": 0.0,
                "occupancy": 0.0,
            }
        else:
            queue_metrics = calculate_erlang_a_metrics(
                calls_per_second,
                request.aht_seconds,
                request.mean_patience_seconds,
                max(1, headcount),
            )
            metrics = {
                "service_level": round(
                    erlang_a_service_level(
                        calls_per_second,
                        request.aht_seconds,
                        request.mean_patience_seconds,
                        max(1, headcount),
                        request.target_seconds,
                    ),
                    6,
                ),
                "abandonment_rate": queue_metrics["abandonment_rate"],
                "occupancy": queue_metrics["occupancy"],
            }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"required_headcount": headcount, **metrics}


@app.post("/v1/rebalance")
def rebalance(request: RebalanceRequest) -> dict[str, Any]:
    schedule = pd.DataFrame(request.schedule_matrix)
    try:
        codes = request.duty_codes.to_domain() if request.duty_codes else None
        updated, logs = AutonomousQueueRebalancer.rebalance_to_floor(
            schedule,
            request.interval_idx,
            request.required_headcount,
            request.min_presence_floor,
            duty_codes=codes,
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    changed_agents = [
        agent
        for agent in schedule.columns
        if schedule.iloc[request.interval_idx][agent]
        != updated.iloc[request.interval_idx][agent]
    ]
    audit_hash: str | None = None
    if changed_agents:
        block = audit_vault.record_mutation(
            actor=request.actor,
            action="INTRADAY_QUEUE_REBALANCE",
            target_agent=",".join(changed_agents),
            delta={
                "interval_idx": request.interval_idx,
                "required_headcount": request.required_headcount,
                "changes": {
                    agent: {
                        "from": str(schedule.iloc[request.interval_idx][agent]),
                        "to": str(updated.iloc[request.interval_idx][agent]),
                    }
                    for agent in changed_agents
                },
            },
        )
        audit_hash = block.block_hash

    return {
        "schedule_matrix": updated.to_dict(orient="list"),
        "logs": logs,
        "changed_agents": changed_agents,
        "audit_hash": audit_hash,
    }


@app.post("/v1/audit/mutations", status_code=201)
def record_audit_mutation(request: AuditMutationRequest) -> dict[str, Any]:
    try:
        block = audit_vault.record_mutation(
            actor=request.actor,
            action=request.action,
            target_agent=request.target_agent,
            delta=request.delta,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return block.to_dict()


@app.get("/v1/audit/integrity")
def audit_integrity() -> dict[str, Any]:
    return {
        "valid": audit_vault.verify_chain_integrity(),
        "blocks": len(audit_vault.chain),
        "latest_hash": audit_vault.chain[-1].block_hash,
    }


@app.post("/v1/shifts/validate-swap", response_model=SwapValidationResponse)
def validate_swap(request: SwapValidationRequest) -> SwapValidationResponse:
    """Decide whether two agents may trade shifts.

    Pure constraint arithmetic - no solver runs, so the answer is immediate.
    A full weekly re-solve takes minutes and cannot be used to approve a swap.
    """
    try:
        result = validate_shift_swap(
            agent_a_roster=[s.to_domain() for s in request.agent_a_roster],
            agent_b_roster=[s.to_domain() for s in request.agent_b_roster],
            shift_a=request.shift_a.to_domain(),
            shift_b=request.shift_b.to_domain(),
            weekly_max_hours=request.weekly_max_hours,
            min_rest_intervals=request.min_rest_intervals,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SwapValidationResponse(is_valid=result.is_valid, reasons=result.reasons)
