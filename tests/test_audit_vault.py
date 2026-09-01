import json

import pytest

from src.audit_vault import AuditIntegrityError, SovereignAuditVault


def test_mutations_are_hash_chained_and_verified():
    vault = SovereignAuditVault(clock=lambda: 1_700_000_000.0)
    first = vault.record_mutation("supervisor", "SWAP", "AGT-001", {"day": 2})
    second = vault.record_mutation("supervisor", "SICK", "AGT-002", {"hours": 8})

    assert first.previous_hash == vault.chain[0].block_hash
    assert second.previous_hash == first.block_hash
    assert vault.verify_chain_integrity()


def test_caller_cannot_mutate_a_hashed_delta():
    delta = {"roles": ["QA"]}
    vault = SovereignAuditVault()
    block = vault.record_mutation("system", "REBALANCE", "AGT-001", delta)
    delta["roles"].append("X")
    returned = block.state_delta
    returned["roles"].append("WD")

    assert block.state_delta == {"roles": ["QA"]}
    assert vault.verify_chain_integrity()


def test_jsonl_chain_survives_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    vault = SovereignAuditVault(path, clock=lambda: 1_700_000_000.0)
    vault.record_mutation("api", "OVERRIDE", "AGT-003", {"from": "QA", "to": "X"})

    reloaded = SovereignAuditVault(path)
    assert len(reloaded.chain) == 2
    assert reloaded.chain[-1].state_delta == {"from": "QA", "to": "X"}
    assert reloaded.verify_chain_integrity()


def test_tampered_persisted_block_is_rejected(tmp_path):
    path = tmp_path / "audit.jsonl"
    vault = SovereignAuditVault(path)
    vault.record_mutation("api", "OVERRIDE", "AGT-003", {"from": "QA", "to": "X"})

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["delta"]["to"] = "WD"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError):
        SovereignAuditVault(path)
