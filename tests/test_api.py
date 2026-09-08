import asyncio

import httpx

from api.main import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_endpoint():
    response = request("GET", "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["audit_chain_valid"] is True


def test_erlang_c_endpoint_sizes_all_intervals():
    response = request(
        "POST",
        "/v1/queueing/erlang-c/headcount",
        json={"arrival_rates": [0, 20, 45], "aht_seconds": 320},
    )
    assert response.status_code == 200
    headcounts = response.json()["required_headcounts"]
    assert headcounts[0] == 0
    assert headcounts == sorted(headcounts)


def test_erlang_a_endpoint_returns_bounded_metrics():
    response = request(
        "POST",
        "/v1/queueing/erlang-a/metrics",
        json={
            "arrival_rate_per_second": 45 / 900,
            "mean_service_time_seconds": 320,
            "mean_patience_seconds": 120,
            "num_servers": 20,
        },
    )
    assert response.status_code == 200
    assert 0 <= response.json()["abandonment_rate"] <= 1


def test_erlang_a_headcount_endpoint_inverts_all_targets():
    response = request(
        "POST",
        "/v1/queueing/erlang-a/headcount",
        json={
            "contacts_per_interval": 45,
            "aht_seconds": 320,
            "mean_patience_seconds": 120,
            "target_service_level": 0.80,
            "target_seconds": 20,
            "max_abandonment_rate": 0.05,
            "max_occupancy": 0.85,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["required_headcount"] == 19
    assert body["service_level"] >= 0.80
    assert body["abandonment_rate"] <= 0.05
    assert body["occupancy"] <= 0.85


def test_rebalance_endpoint_records_changed_agents():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"AGT-001": ["QUEUE"], "AGT-002": ["QUALITY"]},
            "interval_idx": 0,
            "required_headcount": 2,
            "actor": "test-supervisor",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["schedule_matrix"]["AGT-002"] == ["QUEUE"]
    assert body["changed_agents"] == ["AGT-002"]
    assert len(body["audit_hash"]) == 64


def test_rebalance_endpoint_rejects_ragged_matrix():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"AGT-001": ["QUEUE"], "AGT-002": ["QUALITY", "QUEUE"]},
            "interval_idx": 0,
            "required_headcount": 2,
        },
    )
    assert response.status_code == 422


def test_rebalance_accepts_a_caller_supplied_taxonomy():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"D-01": ["ADMIN"], "D-02": ["RADIO"]},
            "interval_idx": 0,
            "required_headcount": 2,
            "duty_codes": {
                "primary_on_queue": "CALL_TAKER",
                "on_queue": ["CALL_TAKER", "RADIO"],
                "preemptable": ["ADMIN"],
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["schedule_matrix"]["D-01"] == ["CALL_TAKER"]


def test_rebalance_applies_a_minimum_presence_floor():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"A": ["QUALITY"], "B": ["PROJECT"]},
            "interval_idx": 0,
            "required_headcount": 0,
            "min_presence_floor": 2,
        },
    )
    assert response.status_code == 200
    assert list(response.json()["schedule_matrix"].values()) == [["QUEUE"], ["QUEUE"]]


def test_invalid_taxonomy_is_rejected_with_422():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"A": ["QUEUE"]},
            "interval_idx": 0,
            "required_headcount": 1,
            "duty_codes": {
                "primary_on_queue": "PHONE",
                "on_queue": ["QUEUE"],
                "preemptable": ["ADMIN"],
            },
        },
    )
    assert response.status_code == 422


# --- shift swap validation -------------------------------------------------
def _shift(agent, day, start=32, duration=34, hours=8.0, kind="FT"):
    return {
        "agent_id": agent, "day": day, "start_interval": start,
        "duration_intervals": duration, "paid_hours": hours, "contract_kind": kind,
    }


def test_validate_swap_accepts_a_clean_trade():
    a, b = _shift("A", 0), _shift("B", 2)
    response = request("POST", "/v1/shifts/validate-swap", json={
        "agent_a_roster": [a], "agent_b_roster": [b], "shift_a": a, "shift_b": b,
    })
    assert response.status_code == 200
    assert response.json() == {"is_valid": True, "reasons": []}


def test_validate_swap_reports_a_rest_violation():
    roster_a = [_shift("A", 0), _shift("A", 1)]
    roster_b = [_shift("B", 0, start=56), _shift("B", 1, start=56)]
    response = request("POST", "/v1/shifts/validate-swap", json={
        "agent_a_roster": roster_a, "agent_b_roster": roster_b,
        "shift_a": roster_a[1], "shift_b": roster_b[1],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is False
    assert any("turnaround rest" in reason for reason in body["reasons"])


def test_validate_swap_rejects_a_malformed_shift_with_422():
    bad = _shift("A", 0)
    bad["contract_kind"] = "CASUAL"
    response = request("POST", "/v1/shifts/validate-swap", json={
        "agent_a_roster": [bad], "agent_b_roster": [_shift("B", 2)],
        "shift_a": bad, "shift_b": _shift("B", 2),
    })
    assert response.status_code == 422
