import random

from scripts import tune_resonance


def test_unique_random_generates_within_bounds():
    rng = random.Random(0)
    values = [0.01, 0.05]
    candidate = tune_resonance._unique_random(rng, 0.0, 0.1, values)
    assert 0.0 <= candidate <= 0.1
    assert all(abs(candidate - val) > 1e-4 for val in values)


def test_bayes_next_candidate_returns_value(monkeypatch):
    rng = random.Random(1)
    results = [
        {"k_res": 0.02, "objective": 0.10},
        {"k_res": 0.08, "objective": 0.30},
        {"k_res": 0.05, "objective": 0.20},
    ]
    used = [entry["k_res"] for entry in results]
    candidate = tune_resonance._bayes_next_candidate(
        results,
        used,
        k_min=0.0,
        k_max=0.12,
        candidate_points=25,
        xi=0.01,
        rng=rng,
    )
    assert 0.0 <= candidate <= 0.12
