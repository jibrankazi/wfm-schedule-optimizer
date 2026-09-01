import math

import numpy as np
import pytest

from src.erlang import required_agents
from src.erlang_advanced import (
    calculate_erlang_a_metrics,
    vectorized_erlang_c_headcount,
)


def test_vector_headcount_matches_scalar_erlang_c():
    calls = np.array([0.0, 5.0, 20.0, 45.0, 70.0, 110.0])
    actual = vectorized_erlang_c_headcount(calls, 320.0)
    expected = np.array([required_agents(value, 320.0) for value in calls])
    assert np.array_equal(actual, expected)


def test_vector_headcount_preserves_shape():
    calls = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert vectorized_erlang_c_headcount(calls, 300.0).shape == calls.shape


@pytest.mark.parametrize(
    "calls,aht",
    [([-1.0], 300.0), ([math.nan], 300.0), ([1.0], 0.0)],
)
def test_vector_headcount_rejects_invalid_inputs(calls, aht):
    with pytest.raises(ValueError):
        vectorized_erlang_c_headcount(np.asarray(calls), aht)


def test_erlang_a_flow_probabilities_balance():
    metrics = calculate_erlang_a_metrics(45 / 900, 320, 120, 20)
    assert metrics["answer_probability"] + metrics["abandonment_rate"] == pytest.approx(
        1.0, abs=2e-6
    )
    assert 0.0 <= metrics["probability_of_queueing"] <= 1.0
    assert 0.0 <= metrics["occupancy"] <= 1.0
    assert metrics["average_speed_of_answer_seconds"] >= 0.0


def test_erlang_a_matches_poisson_special_case():
    # With one server and equal service/abandonment rates, total departure
    # rate from state n is n*mu, so the state distribution is Poisson(1).
    metrics = calculate_erlang_a_metrics(1 / 300, 300, 300, 1)
    assert metrics["probability_of_queueing"] == pytest.approx(1 - math.exp(-1), abs=1e-6)
    assert metrics["occupancy"] == pytest.approx(1 - math.exp(-1), abs=1e-6)
    assert metrics["abandonment_rate"] == pytest.approx(math.exp(-1), abs=1e-6)


def test_more_servers_reduce_abandonment():
    low_capacity = calculate_erlang_a_metrics(80 / 900, 320, 90, 20)
    high_capacity = calculate_erlang_a_metrics(80 / 900, 320, 90, 35)
    assert high_capacity["abandonment_rate"] < low_capacity["abandonment_rate"]


def test_erlang_a_zero_arrivals_and_validation():
    assert calculate_erlang_a_metrics(0, 300, 120, 5)["answer_probability"] == 1.0
    with pytest.raises(ValueError, match="num_servers"):
        calculate_erlang_a_metrics(1.0, 300, 120, 0)
