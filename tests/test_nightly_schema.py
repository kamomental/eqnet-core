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
        "alerts": ["demo_alert"],
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
    assert "valence" in payload.get("affective_stats", {})
