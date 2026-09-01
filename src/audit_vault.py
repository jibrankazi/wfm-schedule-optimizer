"""Append-only, SHA-256 chained audit records.

The chain is tamper-evident: modifying a stored block breaks verification.
It is not, by itself, immutable storage or proof of regulatory compliance.
Production deployments should place the JSONL file on access-controlled,
append-only storage and externally anchor the latest hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast


GENESIS_PREVIOUS_HASH = "0" * 64


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class AuditIntegrityError(RuntimeError):
    """Raised when persisted audit records fail hash-chain verification."""


@dataclass(frozen=True, slots=True)
class AuditBlock:
    index: int
    timestamp_utc: float
    actor_identity: str
    action_type: str
    target_agent: str
    state_delta_json: str
    previous_hash: str
    block_hash: str

    @classmethod
    def create(
        cls,
        *,
        index: int,
        timestamp_utc: float,
        actor_identity: str,
        action_type: str,
        target_agent: str,
        state_delta: Mapping[str, Any],
        previous_hash: str,
    ) -> "AuditBlock":
        delta_json = _canonical_json(dict(state_delta))
        block = cls(
            index=index,
            timestamp_utc=timestamp_utc,
            actor_identity=actor_identity,
            action_type=action_type,
            target_agent=target_agent,
            state_delta_json=delta_json,
            previous_hash=previous_hash,
            block_hash="",
        )
        return cls(
            index=block.index,
            timestamp_utc=block.timestamp_utc,
            actor_identity=block.actor_identity,
            action_type=block.action_type,
            target_agent=block.target_agent,
            state_delta_json=block.state_delta_json,
            previous_hash=block.previous_hash,
            block_hash=block.compute_hash(),
        )

    @property
    def state_delta(self) -> dict[str, Any]:
        """Return a fresh copy so callers cannot mutate the hashed payload."""
        return cast(dict[str, Any], json.loads(self.state_delta_json))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp_utc,
            "actor": self.actor_identity,
            "action": self.action_type,
            "target": self.target_agent,
            "delta": self.state_delta,
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        raw = _canonical_json(self._hash_payload()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "block_hash": self.block_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuditBlock":
        return cls(
            index=int(value["index"]),
            timestamp_utc=float(value["timestamp"]),
            actor_identity=str(value["actor"]),
            action_type=str(value["action"]),
            target_agent=str(value["target"]),
            state_delta_json=_canonical_json(value["delta"]),
            previous_hash=str(value["previous_hash"]),
            block_hash=str(value["block_hash"]),
        )


class SovereignAuditVault:
    """Thread-safe append API for an in-memory or JSONL-backed hash chain."""

    def __init__(
        self,
        path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._lock = threading.Lock()
        self._chain: list[AuditBlock] = []

        if self._path is not None and self._path.exists() and self._path.stat().st_size:
            self._load()
            if not self.verify_chain_integrity():
                raise AuditIntegrityError(f"Audit chain failed verification: {self._path}")
        else:
            self._initialize_genesis_block()

    @property
    def chain(self) -> tuple[AuditBlock, ...]:
        return tuple(self._chain)

    def _initialize_genesis_block(self) -> None:
        genesis = AuditBlock.create(
            index=0,
            timestamp_utc=self._clock(),
            actor_identity="SYSTEM_GENESIS",
            action_type="INITIALIZE_VAULT",
            target_agent="ALL",
            state_delta={"status": "INITIALIZED"},
            previous_hash=GENESIS_PREVIOUS_HASH,
        )
        self._chain.append(genesis)
        self._persist(genesis)

    def _load(self) -> None:
        assert self._path is not None
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    self._chain.append(AuditBlock.from_dict(value))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise AuditIntegrityError(
                        f"Invalid audit record at line {line_number}: {self._path}"
                    ) from exc

    def _persist(self, block: AuditBlock) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(block.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_mutation(
        self,
        actor: str,
        action: str,
        target_agent: str,
        delta: Mapping[str, Any],
    ) -> AuditBlock:
        if not actor.strip() or not action.strip() or not target_agent.strip():
            raise ValueError("actor, action, and target_agent must be non-empty")

        with self._lock:
            previous = self._chain[-1]
            block = AuditBlock.create(
                index=len(self._chain),
                timestamp_utc=self._clock(),
                actor_identity=actor,
                action_type=action,
                target_agent=target_agent,
                state_delta=delta,
                previous_hash=previous.block_hash,
            )
            self._persist(block)
            self._chain.append(block)
            return block

    def verify_chain_integrity(self) -> bool:
        if not self._chain:
            return False

        for index, block in enumerate(self._chain):
            expected_previous = (
                GENESIS_PREVIOUS_HASH if index == 0 else self._chain[index - 1].block_hash
            )
            if block.index != index:
                return False
            if block.previous_hash != expected_previous:
                return False
            if block.block_hash != block.compute_hash():
                return False
        return True
