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


def test_rebalance_endpoint_records_changed_agents():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"AGT-001": ["X"], "AGT-002": ["QA"]},
            "interval_idx": 0,
            "required_headcount": 2,
            "actor": "test-supervisor",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["schedule_matrix"]["AGT-002"] == ["X"]
    assert body["changed_agents"] == ["AGT-002"]
    assert len(body["audit_hash"]) == 64


def test_rebalance_endpoint_rejects_ragged_matrix():
    response = request(
        "POST",
        "/v1/rebalance",
        json={
            "schedule_matrix": {"AGT-001": ["X"], "AGT-002": ["QA", "X"]},
            "interval_idx": 0,
            "required_headcount": 2,
        },
    )
    assert response.status_code == 422
