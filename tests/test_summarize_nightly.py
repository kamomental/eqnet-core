import json

from scripts.summarize_nightly import generate_summary, should_emit_summary


def _base_report():
    return {
        "alerts": [],
        "alerts_detail": [],
        "culture_stats": {
            "default": {
                "count": 42,
                "mean_valence": 0.12,
                "mean_rho": 0.44,
            }
        },
        "policy_feedback": {"enabled": False},
    }


def test_should_emit_summary_requires_alerts_or_delta():
    report = _base_report()
    # No alerts, disabled feedback -> should skip.
    assert not should_emit_summary(report, delta_threshold=0.01)

    report["alerts_detail"] = [{"kind": "culture.high_abs_valence", "value": 0.7, "threshold": 0.6}]
    assert should_emit_summary(report, delta_threshold=0.01)


def test_should_emit_summary_on_significant_delta():
    report = _base_report()
    report["policy_feedback"] = {
        "enabled": True,
        "politeness_before": 0.50,
        "politeness_after": 0.54,
        "delta": 0.04,
        "reason": "resonance_high",
    }
    assert should_emit_summary(report, delta_threshold=0.02)

    body = generate_summary(report, top_k=2)
    assert "politeness 0.500 -> 0.540" in body
