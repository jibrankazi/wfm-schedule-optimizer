import math

def erlang_c_pw(N: int, A: float) -> float:
    if N <= A:
        return 1.0
    sum_terms = sum((A**k) / math.factorial(k) for k in range(N))
    last_term = ((A**N) / math.factorial(N)) * (N / (N - A))
    return last_term / (sum_terms + last_term)

def calculate_sl(N: int, arrival_rate: float, aht_sec: float, target_sec: float = 20.0) -> float:
    A = (arrival_rate * aht_sec) / 900.0
    if N <= A:
        return 0.0
    pw = erlang_c_pw(N, A)
    return 1.0 - (pw * math.exp(-(N - A) * (target_sec / aht_sec)))

def calculate_required_agents(arrival_rate: float, aht_sec: float, target_sl: float = 0.80, max_occ: float = 0.85) -> int:
    A = (arrival_rate * aht_sec) / 900.0
    if A <= 0:
        return 0
    N = math.ceil(A) + 1
    while True:
        sl = calculate_sl(N, arrival_rate, aht_sec)
        occ = A / N
        if sl >= target_sl and occ <= max_occ:
            return N
        N += 1
