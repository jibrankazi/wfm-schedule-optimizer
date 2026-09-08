"""Advanced queueing models for interval staffing and caller abandonment.

``vectorized_erlang_c_headcount`` sizes a complete demand vector with the
same Erlang C service and occupancy constraints used by :mod:`src.erlang`.

``calculate_erlang_a_metrics`` evaluates the stationary birth-death process
for an M/M/c+M (Erlang A) queue. ``erlang_a_service_level`` evaluates the
probability that an offered contact starts service within the target time,
counting abandonment as a service-level failure. ``required_agents_erlang_a``
inverts those metrics by testing integer server counts in ascending order.

Arrival rates are calls per second; service and patience inputs are mean
durations in seconds. The inversion function is the exception: its
``arrival_rate`` input is contacts per interval, matching :mod:`src.erlang`.
"""

from __future__ import annotations

import math

import numpy as np


def _validate_erlang_a_inputs(
    arrival_rate: float,
    mean_service_time: float,
    mean_patience: float,
    servers: int,
) -> None:
    values = (arrival_rate, mean_service_time, mean_patience)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("queue inputs must be finite")
    if arrival_rate < 0:
        raise ValueError("arrival_rate_lambda must be non-negative")
    if mean_service_time <= 0:
        raise ValueError("mean_service_time_mu must be positive")
    if mean_patience <= 0:
        raise ValueError("mean_patience_theta must be positive")
    if servers <= 0:
        raise ValueError("num_servers must be positive")


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
    _validate_erlang_a_inputs(
        arrival_rate_lambda,
        mean_service_time_mu,
        mean_patience_theta,
        num_servers,
    )

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


def _tagged_customer_step(
    probabilities: np.ndarray,
    advance_rates: np.ndarray,
    total_rates: np.ndarray,
    uniform_rate: float,
) -> np.ndarray:
    """Apply one transition of the uniformized tagged-customer chain."""
    queue_states = advance_rates.size
    served_index = queue_states
    abandoned_index = queue_states + 1
    updated = np.zeros_like(probabilities)

    # Absorbing states remain absorbing.
    updated[served_index:] = probabilities[served_index:]

    for queue_ahead in range(queue_states):
        mass = probabilities[queue_ahead]
        if mass == 0.0:
            continue
        updated[queue_ahead] += mass * (
            1.0 - total_rates[queue_ahead] / uniform_rate
        )
        destination = served_index if queue_ahead == 0 else queue_ahead - 1
        updated[destination] += mass * (
            advance_rates[queue_ahead] / uniform_rate
        )
        updated[abandoned_index] += mass * (
            (total_rates[queue_ahead] - advance_rates[queue_ahead])
            / uniform_rate
        )
    return updated


def _uniformized_evolution(
    probabilities: np.ndarray,
    advance_rates: np.ndarray,
    total_rates: np.ndarray,
    duration_sec: float,
    tail_tolerance: float = 1e-13,
) -> np.ndarray:
    """Evolve a small CTMC using stable, time-sliced uniformization."""
    if duration_sec <= 0.0:
        return probabilities.copy()

    uniform_rate = float(total_rates.max())
    # Keeping each Poisson mean modest avoids exp(-mean) underflow and makes
    # the tail test reliable even for unusually long service thresholds.
    slices = max(1, math.ceil(uniform_rate * duration_sec / 40.0))
    slice_duration = duration_sec / slices
    poisson_mean = uniform_rate * slice_duration
    evolved = probabilities.copy()

    for _ in range(slices):
        state = evolved.copy()
        weight = math.exp(-poisson_mean)
        cumulative = weight
        result = weight * state
        step = 0

        while 1.0 - cumulative > tail_tolerance:
            step += 1
            state = _tagged_customer_step(
                state, advance_rates, total_rates, uniform_rate
            )
            weight *= poisson_mean / step
            result += weight * state
            cumulative += weight
            if step > 10_000:  # pragma: no cover - defensive convergence guard
                raise RuntimeError("uniformization did not converge")

        # The omitted Poisson tail is below tolerance; normalizing prevents
        # tiny probability loss from accumulating across time slices.
        evolved = result / cumulative

    return evolved


def erlang_a_service_level(
    arrival_rate_lambda: float,
    mean_service_time_mu: float,
    mean_patience_theta: float,
    num_servers: int,
    target_time_sec: float,
) -> float:
    """Return the fraction of offered contacts served within the target time.

    An arrival sees the stationary M/M/c+M state distribution (PASTA). Calls
    that see an idle server are answered immediately. Queued calls are then
    evolved through a tagged-customer CTMC: service completions and abandonments
    ahead move the caller forward, while the caller's own patience expiry is an
    absorbing failure. New arrivals join behind and therefore do not affect the
    tagged caller's wait.

    This is not the Erlang C waiting-time tail with a different queue-length
    distribution. Reusing that formula would be incorrect once abandonment is
    active.
    """
    _validate_erlang_a_inputs(
        arrival_rate_lambda,
        mean_service_time_mu,
        mean_patience_theta,
        num_servers,
    )
    if not math.isfinite(target_time_sec) or target_time_sec < 0:
        raise ValueError("target_time_sec must be finite and non-negative")
    if arrival_rate_lambda == 0:
        return 1.0

    service_rate = 1.0 / mean_service_time_mu
    patience_rate = 1.0 / mean_patience_theta
    stationary = _stationary_distribution(
        arrival_rate_lambda, service_rate, patience_rate, num_servers
    )

    queued_probabilities = stationary[num_servers:]
    queue_states = queued_probabilities.size
    served_index = queue_states
    initial = np.zeros(queue_states + 2, dtype=float)
    initial[:queue_states] = queued_probabilities
    initial[served_index] = float(stationary[:num_servers].sum())

    queue_ahead = np.arange(queue_states, dtype=float)
    service_capacity = num_servers * service_rate
    advance_rates = service_capacity + queue_ahead * patience_rate
    total_rates = advance_rates + patience_rate
    evolved = _uniformized_evolution(
        initial, advance_rates, total_rates, target_time_sec
    )
    return min(1.0, max(0.0, float(evolved[served_index])))


def required_agents_erlang_a(
    arrival_rate: float,
    aht_sec: float,
    mean_patience_sec: float,
    interval_sec: int = 900,
    target_sl: float = 0.80,
    target_sec: float = 20.0,
    max_abandon_rate: float = 0.05,
    max_occupancy: float = 0.85,
    min_presence_floor: int = 1,
    ceiling: int = 500,
) -> int:
    """Find the minimum Erlang A server count satisfying every target.

    ``arrival_rate`` is contacts per interval. The search deliberately does
    not use the Erlang C answer as an upper bound: a strict abandonment cap can
    require more servers than the corresponding Erlang C service-level answer.
    Failure to find a compliant count raises instead of silently returning a
    non-compliant schedule requirement.
    """
    numeric = (
        arrival_rate,
        aht_sec,
        mean_patience_sec,
        target_sl,
        target_sec,
        max_abandon_rate,
        max_occupancy,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("queue inputs must be finite")
    if arrival_rate < 0:
        raise ValueError("arrival_rate must be non-negative")
    if aht_sec <= 0:
        raise ValueError("aht_sec must be positive")
    if mean_patience_sec <= 0:
        raise ValueError("mean_patience_sec must be positive")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    if not 0.0 <= target_sl <= 1.0:
        raise ValueError("target_sl must be between 0 and 1")
    if target_sec < 0:
        raise ValueError("target_sec must be non-negative")
    if not 0.0 <= max_abandon_rate <= 1.0:
        raise ValueError("max_abandon_rate must be between 0 and 1")
    if not 0.0 < max_occupancy <= 1.0:
        raise ValueError("max_occupancy must be greater than 0 and at most 1")
    if min_presence_floor < 0:
        raise ValueError("min_presence_floor must be non-negative")
    if ceiling < max(1, min_presence_floor):
        raise ValueError("ceiling must be at least the presence floor")

    if arrival_rate == 0:
        return min_presence_floor

    calls_per_second = arrival_rate / float(interval_sec)
    offered_load = calls_per_second * aht_sec
    # Flow conservation gives a safe lower bound.  Meeting the abandonment
    # cap means at least (1-max_abandon_rate) of offered work is answered; the
    # occupancy ceiling therefore requires this many servers at minimum.
    occupancy_floor = math.ceil(
        offered_load * (1.0 - max_abandon_rate) / max_occupancy - 1e-12
    )
    first_server_count = max(1, min_presence_floor, occupancy_floor)
    for servers in range(first_server_count, ceiling + 1):
        metrics = calculate_erlang_a_metrics(
            calls_per_second, aht_sec, mean_patience_sec, servers
        )
        achieved_sl = erlang_a_service_level(
            calls_per_second,
            aht_sec,
            mean_patience_sec,
            servers,
            target_sec,
        )
        if (
            achieved_sl >= target_sl
            and metrics["abandonment_rate"] <= max_abandon_rate
            and metrics["occupancy"] <= max_occupancy
        ):
            return servers

    raise ValueError(
        f"No headcount up to {ceiling} satisfies the Erlang A targets for "
        f"{arrival_rate:.2f} calls per {interval_sec}s interval"
    )
