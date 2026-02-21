from __future__ import annotations

from rag.assoc_safety import calc_saturation_stats, clamp_score, sanitize_weights


def test_sanitize_weights_clamps_negative_and_normalizes():
    out = sanitize_weights({"semantic": -1.0, "temporal": 2.0}, normalize=True, fallback_key="semantic")
    assert out["semantic"] == 0.0
    assert out["temporal"] == 1.0


def test_sanitize_weights_fallback_when_all_zero():
    out = sanitize_weights({"semantic": -1.0, "temporal": -2.0}, normalize=True, fallback_key="semantic")
    assert out["semantic"] == 1.0
    assert out["temporal"] == 0.0


def test_clamp_score_swapped_bounds():
    assert clamp_score(2.5, 1.0, 0.0) == 1.0


def test_calc_saturation_stats_counts_bound_hits():
    sat_min, sat_max, n = calc_saturation_stats([0.0, 0.2, 1.0, 1.0], 0.0, 1.0)
    assert n == 4
    assert sat_min == 1
    assert sat_max == 2
