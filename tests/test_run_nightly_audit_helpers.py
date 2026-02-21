from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.run_nightly_audit import (
    _extract_nightly_metrics,
    _load_previous_report,
    _recommended_action_code,
    _snapshot_changed_keys,
    _weekly_metric_summary,
)


def test_extract_nightly_metrics_reads_nested_values() -> None:
    payload = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.71,
            "uncertainty_confidence": {"low_ratio": 0.22},
        }
    }
    metrics = _extract_nightly_metrics(payload)
    assert metrics["sat_p95"] == 0.71
    assert metrics["low_ratio"] == 0.22


def test_load_previous_report_reads_previous_day(tmp_path: Path) -> None:
    out_json = tmp_path / "nightly_audit_20251215.json"
    prev = tmp_path / "nightly_audit_20251214.json"
    prev.write_text(
        json.dumps({"nightly_audit": {"lazy_rag_sat_ratio_p95": 0.5}}),
        encoding="utf-8",
    )
    loaded = _load_previous_report(out_json, date(2025, 12, 15))
    assert loaded.get("nightly_audit", {}).get("lazy_rag_sat_ratio_p95") == 0.5


def test_weekly_metric_summary_aggregates_existing_week_files(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    for stamp, sat, low in [
        ("20251215", 0.50, 0.10),
        ("20251216", 0.60, 0.20),
    ]:
        (tmp_path / f"nightly_audit_{stamp}.json").write_text(
            json.dumps(
                {
                    "nightly_audit": {
                        "lazy_rag_sat_ratio_p95": sat,
                        "uncertainty_confidence": {"low_ratio": low},
                    }
                }
            ),
            encoding="utf-8",
        )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.70,
            "uncertainty_confidence": {"low_ratio": 0.30},
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current)
    assert summary["count"] == 3.0
    assert summary["sat_p95_avg"] == 0.6
    assert summary["sat_p95_max"] == 0.7
    assert summary["low_ratio_avg"] == 0.2
    assert summary["low_ratio_max"] == 0.3


def test_snapshot_changed_keys_detects_nested_changes() -> None:
    prev = {
        "assoc_temporal_tau_sec": 86400.0,
        "assoc_weights": {"semantic": 1.0, "temporal": 0.1},
    }
    cur = {
        "assoc_temporal_tau_sec": 43200.0,
        "assoc_weights": {"semantic": 1.0, "temporal": 0.2},
    }
    changed = _snapshot_changed_keys(cur, prev)
    assert "assoc_temporal_tau_sec" in changed
    assert "assoc_weights.temporal" in changed


def test_recommended_action_code_for_saturation_high() -> None:
    weekly = {"sat_p95_avg": 0.65, "sat_p95_max": 0.8, "low_ratio_max": 0.1}
    payload = {"nightly_audit": {"uncertainty_reason_top3": []}}
    code = _recommended_action_code(weekly, payload)
    assert code == "saturation_high"


def test_recommended_action_code_none_when_no_samples() -> None:
    weekly = {"count": 0.0, "sat_p95_avg": 0.0, "sat_p95_max": 0.0, "low_ratio_max": 0.0}
    payload = {"nightly_audit": {"uncertainty_reason_top3": []}}
    code = _recommended_action_code(weekly, payload)
    assert code == "none"
