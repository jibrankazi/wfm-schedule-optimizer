"""Advanced queueing models for interval staffing and caller abandonment.

``vectorized_erlang_c_headcount`` sizes a complete demand vector with the
same Erlang C service and occupancy constraints used by :mod:`src.erlang`.

``calculate_erlang_a_metrics`` evaluates the stationary birth-death process
for an M/M/c+M (Erlang A) queue.  Arrival rates are calls per second; service
and patience inputs are mean durations in seconds.
"""

from __future__ import annotations

import math

import numpy as np


def vectorized_erlang_c_headcount(
    arrival_rates: np.ndarray,
    aht_seconds: float,
    interval_sec: int = 900,
    target_sl: float = 0.80,
    target_sec: float = 20.0,
    max_occ: float = 0.85,
    ceiling: int = 5_000,
) -> np.ndarray:
    """Return minimum Erlang C headcount for every demand interval.

    ``arrival_rates`` contains calls per interval, not calls per second.  The
    returned array has the same shape as the input.
    """
    rates = np.asarray(arrival_rates, dtype=float)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("arrival_rates must contain finite, non-negative values")
    if not math.isfinite(aht_seconds) or aht_seconds <= 0:
        raise ValueError("aht_seconds must be positive")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    if not 0.0 <= target_sl <= 1.0:
        raise ValueError("target_sl must be between 0 and 1")
    if target_sec < 0:
        raise ValueError("target_sec must be non-negative")
    if not 0.0 < max_occ <= 1.0:
        raise ValueError("max_occ must be greater than 0 and at most 1")
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")

    intensities = rates * aht_seconds / float(interval_sec)
    headcounts = np.zeros(rates.shape, dtype=int)

    for index in np.ndindex(intensities.shape):
        intensity = float(intensities[index])
        if intensity <= 0:
            continue

        agents = max(1, math.ceil(intensity))
        while agents <= ceiling:
            blocking = 1.0
            for count in range(1, agents + 1):
                blocking = (intensity * blocking) / (count + intensity * blocking)

            utilisation = intensity / agents
            if utilisation >= 1.0:
                probability_wait = 1.0
            else:
                probability_wait = blocking / (
                    utilisation * blocking + (1.0 - utilisation)
                )
            service_level = 1.0 - probability_wait * math.exp(
                -(agents - intensity) * (target_sec / aht_seconds)
            )

            if service_level >= target_sl and utilisation <= max_occ:
                headcounts[index] = agents
                break
            agents += 1
        else:
            raise ValueError(
                f"No headcount up to {ceiling} satisfies the target for "
                f"{rates[index]:.2f} calls per interval"
            )

    return headcounts


def _stationary_distribution(
    arrival_rate: float,
    service_rate: float,
    patience_rate: float,
    servers: int,
    tail_tolerance: float = 1e-13,
    max_states: int = 100_000,
) -> np.ndarray:
    """Return the normalized Erlang A state probabilities.

    Log weights avoid overflow when the offered load is far above capacity.
    The tail stops only after the birth/death ratio is below one and a
    geometric upper bound is smaller than ``tail_tolerance``.
    """
    log_weights = [0.0]
    max_log_weight = 0.0

    for state in range(1, max_states + 1):
        departures = min(state, servers) * service_rate
        abandonments = max(state - servers, 0) * patience_rate
        ratio = arrival_rate / (departures + abandonments)
        log_weight = log_weights[-1] + math.log(ratio)
        log_weights.append(log_weight)
        max_log_weight = max(max_log_weight, log_weight)

        if state > servers and ratio < 1.0:
            relative_weight = math.exp(log_weight - max_log_weight)
            tail_bound = relative_weight * ratio / (1.0 - ratio)
            if tail_bound < tail_tolerance:
                break
    else:
        raise RuntimeError("Erlang A state distribution did not converge")

    weights = np.exp(np.asarray(log_weights) - max_log_weight)
    return weights / weights.sum()


def calculate_erlang_a_metrics(
    arrival_rate_lambda: float,
    mean_service_time_mu: float,
    mean_patience_theta: float,
    num_servers: int,
) -> dict[str, float]:
    """Compute steady-state Erlang A metrics for an M/M/c+M queue.

    Parameters use seconds: ``arrival_rate_lambda`` is calls per second while
    the two ``mean_*`` values are durations.  ``abandonment_rate`` is the
    fraction of arrivals that abandon, not a count per second.  ASA is the
    mean queue wait among calls that are eventually answered.
    """
    values = (arrival_rate_lambda, mean_service_time_mu, mean_patience_theta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("queue inputs must be finite")
    if arrival_rate_lambda < 0:
        raise ValueError("arrival_rate_lambda must be non-negative")
    if mean_service_time_mu <= 0:
        raise ValueError("mean_service_time_mu must be positive")
    if mean_patience_theta <= 0:
        raise ValueError("mean_patience_theta must be positive")
    if num_servers <= 0:
        raise ValueError("num_servers must be positive")

    if arrival_rate_lambda == 0:
        return {
            "probability_of_queueing": 0.0,
            "abandonment_rate": 0.0,
            "answer_probability": 1.0,
            "occupancy": 0.0,
            "average_speed_of_answer_seconds": 0.0,
            "expected_queue_length": 0.0,
        }

    service_rate = 1.0 / mean_service_time_mu
    patience_rate = 1.0 / mean_patience_theta
    probabilities = _stationary_distribution(
        arrival_rate_lambda, service_rate, patience_rate, num_servers
    )
    states = np.arange(probabilities.size, dtype=float)
    busy_servers = np.minimum(states, num_servers)
    queue_lengths = np.maximum(states - num_servers, 0.0)

    expected_busy = float(np.dot(probabilities, busy_servers))
    expected_queue = float(np.dot(probabilities, queue_lengths))
    probability_queueing = float(probabilities[num_servers:].sum())
    abandonment_probability = patience_rate * expected_queue / arrival_rate_lambda
    abandonment_probability = min(1.0, max(0.0, abandonment_probability))
    answer_probability = 1.0 - abandonment_probability

    # PASTA lets an arrival see the stationary state distribution.  For an
    # arrival with q callers ahead, each stage ends in service/advance or the
    # tagged caller's abandonment.  Event time and event type are independent,
    # which makes the conditional waiting-time sum straightforward.
    served_wait_total = 0.0
    served_probability = float(probabilities[:num_servers].sum())
    conditional_service_probability = 1.0
    conditional_wait = 0.0
    service_capacity = num_servers * service_rate
    for queue_ahead, state in enumerate(range(num_servers, probabilities.size)):
        relevant_rate = service_capacity + (queue_ahead + 1) * patience_rate
        advance_probability = (
            service_capacity + queue_ahead * patience_rate
        ) / relevant_rate
        conditional_service_probability *= advance_probability
        conditional_wait += 1.0 / relevant_rate
        weight = float(probabilities[state]) * conditional_service_probability
        served_probability += weight
        served_wait_total += weight * conditional_wait

    asa_seconds = served_wait_total / served_probability if served_probability else 0.0

    return {
        "probability_of_queueing": round(probability_queueing, 6),
        "abandonment_rate": round(abandonment_probability, 6),
        "answer_probability": round(answer_probability, 6),
        "occupancy": round(expected_busy / num_servers, 6),
        "average_speed_of_answer_seconds": round(asa_seconds, 3),
        "expected_queue_length": round(expected_queue, 6),
    }
