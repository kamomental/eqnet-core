import json
from pathlib import Path

import jsonschema

from ops import nightly


def test_nightly_json_matches_schema(tmp_path) -> None:
    """Validate that the JSON payload produced by nightly matches the schema."""
    report = {
        "config_snapshot": {
            "ignition": {"theta_on": 0.62, "theta_off": 0.48, "dwell_steps": 8},
            "telemetry": {"log_path": "telemetry/ignition-%Y%m%d.jsonl"},
        },
        "field_state": {
            "S_mean": 0.42,
            "H_mean": 0.51,
            "rho_mean": 0.48,
            "Ignition_mean": 0.46,
            "valence_mean": -0.03,
            "arousal_mean": 0.12,
            "rho_I_corr": 0.27,
            "S_I_corr": -0.13,
            "H_I_corr": 0.05,
            "valence_I_corr": 0.18,
            "arousal_I_corr": 0.21,
        },
        "plots": {
            "ignition_timeseries": "reports/plots/ignition_timeseries.png",
            "rho_vs_I_scatter": "reports/plots/rho_vs_I_scatter.png",
            "memory_graph": "reports/plots/memory_graph.png",
            "affective_map": "reports/plots/affective_map.png",
            "culture_resonance": "reports/plots/culture_resonance.png",
            "culture_trend": "reports/plots/culture_trend.png",
            "resonance_objective": "reports/plots/resonance_objective.png",
            "resonance_bayes_trace": "reports/plots/resonance_bayes_trace.png",
            "vision_counts": "reports/plots/vision_counts.png",
            "vision_pose": "reports/plots/vision_pose.png",
        },
        "tuning_suggestion": {
            "reason": "rho/I correlation low (0.270)",
            "theta_on": {"current": 0.62, "suggested": 0.6},
            "theta_off": {"current": 0.48, "suggested": 0.46},
        },
        "warnings": ["demo warning"],
        "affective_stats": {
            "valence": {"mean": 0.1, "std": 0.02, "q25": 0.05, "q50": 0.1, "q75": 0.15, "count": 200},
            "arousal": {"mean": 0.2, "std": 0.03, "q25": 0.1, "q50": 0.2, "q75": 0.25, "count": 200}
        },
        "vision_snapshot": {
            "events": 24,
            "detections_total": 96,
            "counts_by_kind": {"handshake": 12, "person": 50},
            "mean_valence": 0.08,
            "mean_arousal": 0.22,
            "pose_mean": {"yaw_mean": -1.2, "pitch_mean": 0.4, "roll_mean": 0.1},
            "fps_mean": 29.8,
            "ts_last": 1_730_000_000_000,
        },
        "resonance": {
            "pairs": [
                {
                    "agents": ["A", "B"],
                    "rho_corr": 0.45,
                    "rho_corr_ci95": [0.2, 0.7],
                    "rho_cross_corr_peak": 0.5,
                    "rho_cross_corr_lag": 2,
                    "rho_cross_corr_lag_refined": 1.8,
                    "energy": 0.03,
                    "n_eff": 120,
                    "objective": 0.41,
                    "partial_corr": 0.4,
                    "granger": {"lag": 1, "a_to_b_f": 2.1, "b_to_a_f": 0.9},
                }
            ],
            "partial_corr": 0.4,
            "granger": {"lag": 1, "a_to_b_f": 2.1, "b_to_a_f": 0.9}
        },
        "culture_stats": {
            "JP_basic": {
                "count": 120.0,
                "mean_valence": 0.12,
                "mean_arousal": 0.35,
                "mean_rho": 0.48,
                "mean_politeness": 0.7,
                "mean_intimacy": 0.3,
            }
        },
        "resonance_history_path": "reports/resonance_history.jsonl",
        "resonance_bayes_trace_path": "reports/resonance_bayes_trace.jsonl",
        "culture_history_path": "reports/culture_history.jsonl",
        "policy_feedback": {
            "enabled": True,
            "politeness_before": 0.5,
            "politeness_after": 0.52,
            "delta": 0.02,
            "intimacy_before": 0.48,
            "intimacy_after": 0.49,
            "intimacy_delta": 0.01,
            "corr": 0.68,
            "corr_source": "resonance_summary",
            "reason": "resonance_high",
            "reference_tag": "JP_basic",
            "reference_rho": 0.52,
            "reference_valence": 0.12,
            "alerts": ["demo_alert"],
            "vision_adjustment": {
                "politeness_delta": 0.01,
                "intimacy_delta": 0.005,
                "details": [
                    {
                        "kind": "handshake",
                        "count": 4,
                        "effective": 0.2,
                        "normalize": "events",
                        "politeness_delta": 0.01,
                        "intimacy_delta": 0.005,
                    }
                ],
                "counts_by_kind": {"handshake": 4},
                "normalisers": {"events": 24, "detections": 96},
            },
        },
        "policy_feedback_history_path": "reports/policy_feedback_history.jsonl",
        "alerts": ["demo_alert"],
        "alerts_detail": [
            {"kind": "culture.high_abs_valence", "tag": "JP_basic", "value": 0.12, "threshold": 0.6}
        ],
        "run_seed": 20251027,
    }
    out_dir = tmp_path / "reports"
    json_path = nightly._write_json_summary(report, out_dir=str(out_dir))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(Path("schema/nightly.v1.json").read_text(encoding="utf-8"))

    jsonschema.validate(instance=payload, schema=schema)
    assert payload["schema"] == "nightly.v1"
    assert payload["plots"]["memory_graph"].endswith("memory_graph.png")
    assert payload["plots"]["affective_map"].endswith("affective_map.png")
    assert payload["plots"]["culture_resonance"].endswith("culture_resonance.png")
    assert payload["plots"]["culture_trend"].endswith("culture_trend.png")
    assert "valence" in payload.get("affective_stats", {})
    assert "vision_snapshot" in payload
    assert "resonance" in payload
    assert payload.get("resonance_history_path")
    assert payload.get("resonance_bayes_trace_path")
    assert "culture_stats" in payload
    assert payload.get("culture_history_path")
    assert payload.get("alerts_detail")
    assert payload.get("policy_feedback")
    assert payload.get("policy_feedback_history_path")
    assert "vision_adjustment" in payload["policy_feedback"]
    normalisers = payload["policy_feedback"]["vision_adjustment"].get("normalisers", {})
    assert "events" in normalisers
