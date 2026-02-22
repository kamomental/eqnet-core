from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.run_nightly_audit import (
    _build_change_proposal_from_mecpe_alert,
    _extract_nightly_metrics,
    _load_mecpe_alert_thresholds,
    _load_mecpe_proposal_rules,
    _load_previous_report,
    _proposal_allowed_by_rules,
    _recommended_action_code,
    _snapshot_changed_keys,
    _weekly_metric_summary,
)


def test_extract_nightly_metrics_reads_nested_values() -> None:
    payload = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.71,
            "uncertainty_confidence": {"low_ratio": 0.22},
            "health": {"status": "YELLOW"},
            "mecpe_audit": {
                "total_records": 10,
                "mecpe_lines_total": 12,
                "hash_integrity": {"ok_rate": 0.9},
                "future_cause_conflict": {"conflict_rate": 0.1},
                "evidence_staleness": {"missing_ratio": 0.3},
                "contract_errors": {"total": 2, "ratio": 0.167, "top_type": "json_decode"},
            },
        }
    }
    metrics = _extract_nightly_metrics(payload)
    assert metrics["sat_p95"] == 0.71
    assert metrics["low_ratio"] == 0.22
    assert metrics["mecpe_hash_ok_rate"] == 0.9
    assert metrics["mecpe_conflict_rate"] == 0.1
    assert metrics["mecpe_staleness_ratio"] == 0.3
    assert metrics["mecpe_lines_total"] == 12.0
    assert metrics["mecpe_contract_error_total"] == 2.0
    assert metrics["mecpe_contract_error_ratio"] == 0.167
    assert metrics["mecpe_contract_error_ratio_legacy"] == 0.2
    assert metrics["mecpe_contract_error_top_type"] == "json_decode"
    assert metrics["health_status"] == "YELLOW"


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
    mecpe = summary.get("mecpe_audit") or {}
    assert "hash_integrity" in mecpe
    assert "future_cause_conflict" in mecpe
    assert "evidence_staleness" in mecpe
    assert "contract_errors" in mecpe
    assert "lines_total_sum" in (mecpe.get("contract_errors") or {})
    assert "health_flags" in summary
    assert "mecpe_alert" in summary
    alert = summary.get("mecpe_alert") or {}
    assert "level" in alert
    assert "reasons" in alert
    assert "summary" in alert


def test_weekly_metric_summary_sets_alert_level_from_ratio_and_type(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    rows = [
        ("20251215", 0.50, 0.10, "GREEN", 0, 0.0, ""),
        ("20251216", 0.60, 0.20, "YELLOW", 2, 0.02, "json_decode"),
    ]
    for stamp, sat, low, status, errors, ratio, top_type in rows:
        (tmp_path / f"nightly_audit_{stamp}.json").write_text(
            json.dumps(
                {
                    "nightly_audit": {
                        "lazy_rag_sat_ratio_p95": sat,
                        "uncertainty_confidence": {"low_ratio": low},
                        "health": {"status": status},
                        "mecpe_audit": {
                            "total_records": 10,
                            "mecpe_lines_total": 10,
                            "hash_integrity": {"ok_rate": 1.0},
                            "future_cause_conflict": {"conflict_rate": 0.0},
                            "evidence_staleness": {"missing_ratio": 0.0},
                            "contract_errors": {"total": errors, "ratio": ratio, "top_type": top_type},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.70,
            "uncertainty_confidence": {"low_ratio": 0.30},
            "health": {"status": "YELLOW"},
            "mecpe_audit": {
                "total_records": 10,
                "mecpe_lines_total": 10,
                "hash_integrity": {"ok_rate": 1.0},
                "future_cause_conflict": {"conflict_rate": 0.0},
                "evidence_staleness": {"missing_ratio": 0.0},
                "contract_errors": {"total": 1, "ratio": 0.01, "top_type": "missing_required_key"},
            },
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current)
    alert = summary.get("mecpe_alert") or {}
    assert alert.get("level") == "ALERT"
    reasons = alert.get("reasons") or []
    assert any("ratio_max>=0.01" == reason for reason in reasons)
    assert any("high_priority_type_detected" == reason for reason in reasons)


def test_weekly_metric_summary_includes_shadow_eval_aggregate(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "change_decisions-20251217.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "change_decision.v0",
                        "decision_id": "d-1",
                        "timestamp_ms": 1734393600000,
                        "proposal_id": "p-1",
                        "decision": "ACCEPT_SHADOW",
                        "actor": "human",
                        "reason": "approved",
                        "source_week": "2025-W51",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "change_decision.v0",
                        "decision_id": "d-2",
                        "timestamp_ms": 1734393600001,
                        "proposal_id": "p-2",
                        "decision": "ACCEPT_SHADOW",
                        "actor": "human",
                        "reason": "approved",
                        "source_week": "2025-W51",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "change_decision.v0",
                        "decision_id": "d-3",
                        "timestamp_ms": 1734393600002,
                        "proposal_id": "p-3",
                        "decision": "ACCEPT_SHADOW",
                        "actor": "human",
                        "reason": "approved",
                        "source_week": "2025-W51",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry_dir / "proposal_links-20251217.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "proposal_link.v0",
                        "link_id": "l-1",
                        "timestamp_ms": 1734393600000,
                        "proposal_id": "p-1",
                        "eval_report_id": "e-1",
                        "link_type": "shadow_eval",
                        "source_week": "2025-W51",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "proposal_link.v0",
                        "link_id": "l-2",
                        "timestamp_ms": 1734393600001,
                        "proposal_id": "p-2",
                        "eval_report_id": "e-2",
                        "link_type": "shadow_eval",
                        "source_week": "2025-W51",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry_dir / "eval_reports-20251217.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "eval_report.v0",
                        "eval_report_id": "e-1",
                        "timestamp_ms": 1734393600000,
                        "proposal_id": "p-1",
                        "method": "replay_eval",
                        "verdict": "PASS",
                        "metrics_before": {"contract_errors_ratio": 0.1, "hash_integrity_ok_rate": 0.8},
                        "metrics_after": {"contract_errors_ratio": 0.05, "hash_integrity_ok_rate": 0.9},
                        "source_week": "2025-W51",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "eval_report.v0",
                        "eval_report_id": "e-2",
                        "timestamp_ms": 1734393600001,
                        "proposal_id": "p-2",
                        "method": "replay_eval",
                        "verdict": "FAIL",
                        "metrics_before": {"contract_errors_ratio": 0.02, "hash_integrity_ok_rate": 0.9},
                        "metrics_after": {"contract_errors_ratio": 0.03, "hash_integrity_ok_rate": 0.88},
                        "source_week": "2025-W51",
                        "delta": {"contract_errors_ratio": 0.01, "hash_integrity_ok_rate": -0.02},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.70,
            "uncertainty_confidence": {"low_ratio": 0.30},
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current, telemetry_dir=telemetry_dir)
    shadow = summary.get("mecpe_shadow_eval") or {}
    counts = shadow.get("counts") or {}
    assert counts.get("pass") == 1
    assert counts.get("fail") == 1
    assert counts.get("approved_shadow_count") == 3
    assert counts.get("approved_but_no_eval_count") == 1
    ratios = shadow.get("by_verdict_ratio") or {}
    assert ratios.get("pass_ratio") == 0.5
    assert ratios.get("fail_ratio") == 0.5
    delta_summary = shadow.get("delta_summary") or {}
    contract_delta = delta_summary.get("contract_errors_ratio") or {}
    assert contract_delta.get("min") == -0.05
    assert contract_delta.get("max") == 0.01
    pending = shadow.get("pending_proposals") or []
    assert pending == ["p-3"]
    assert shadow.get("approval_to_eval_latency_ms_p50") == 0
    assert shadow.get("approval_to_eval_latency_ms_p95") == 0
    assert shadow.get("oldest_pending_age_ms") > 0


def test_weekly_metric_summary_includes_canary_eval_aggregate(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "change_decisions-20251217.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "change_decision.v0",
                        "decision_id": "d-10",
                        "timestamp_ms": 1734393600000,
                        "proposal_id": "p-10",
                        "decision": "ACCEPT_CANARY",
                        "actor": "human",
                        "reason": "approved",
                        "source_week": "2025-W51",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry_dir / "proposal_links-20251217.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "proposal_link.v0",
                "link_id": "l-10",
                "timestamp_ms": 1734393600000,
                "proposal_id": "p-10",
                "eval_report_id": "e-10",
                "link_type": "canary_eval",
                "source_week": "2025-W51",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry_dir / "eval_reports-20251217.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "eval_report.v0",
                "eval_report_id": "e-10",
                "timestamp_ms": 1734393600000,
                "proposal_id": "p-10",
                "method": "canary_eval",
                "verdict": "PASS",
                "metrics_before": {"contract_errors_ratio": 0.08, "hash_integrity_ok_rate": 0.9},
                "metrics_after": {"contract_errors_ratio": 0.04, "hash_integrity_ok_rate": 0.93},
                "source_week": "2025-W51",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.70,
            "uncertainty_confidence": {"low_ratio": 0.30},
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current, telemetry_dir=telemetry_dir)
    canary = summary.get("mecpe_canary_eval") or {}
    counts = canary.get("counts") or {}
    assert counts.get("pass") == 1
    assert counts.get("approved_shadow_count") == 1
    assert counts.get("approved_but_no_eval_count") == 0
    ratios = canary.get("by_verdict_ratio") or {}
    assert ratios.get("pass_ratio") == 1.0
    delta_summary = canary.get("delta_summary") or {}
    contract_delta = delta_summary.get("contract_errors_ratio") or {}
    assert contract_delta.get("min") == -0.04
    assert contract_delta.get("max") == -0.04


def test_weekly_metric_summary_sets_warn_when_shadow_pending_ratio_high(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    decision_rows = []
    for idx in range(5):
        decision_rows.append(
            json.dumps(
                {
                    "schema_version": "change_decision.v0",
                    "decision_id": f"d-s-{idx}",
                    "timestamp_ms": 1734393600000 + idx,
                    "proposal_id": f"ps-{idx}",
                    "decision": "ACCEPT_SHADOW",
                    "actor": "human",
                    "reason": "approved",
                    "source_week": "2025-W51",
                }
            )
        )
    (telemetry_dir / "change_decisions-20251217.jsonl").write_text(
        "\n".join(decision_rows) + "\n",
        encoding="utf-8",
    )
    link_rows = []
    for idx in range(3):
        link_rows.append(
            json.dumps(
                {
                    "schema_version": "proposal_link.v0",
                    "link_id": f"l-s-{idx}",
                    "timestamp_ms": 1734393600000 + idx,
                    "proposal_id": f"ps-{idx}",
                    "eval_report_id": f"e-s-{idx}",
                    "link_type": "shadow_eval",
                    "source_week": "2025-W51",
                }
            )
        )
    (telemetry_dir / "proposal_links-20251217.jsonl").write_text(
        "\n".join(link_rows) + "\n",
        encoding="utf-8",
    )
    eval_rows = []
    for idx in range(3):
        eval_rows.append(
            json.dumps(
                {
                    "schema_version": "eval_report.v0",
                    "eval_report_id": f"e-s-{idx}",
                    "timestamp_ms": 1734393600000 + idx,
                    "proposal_id": f"ps-{idx}",
                    "method": "replay_eval",
                    "verdict": "PASS",
                    "metrics_before": {"contract_errors_ratio": 0.05},
                    "metrics_after": {"contract_errors_ratio": 0.04},
                    "source_week": "2025-W51",
                }
            )
        )
    (telemetry_dir / "eval_reports-20251217.jsonl").write_text(
        "\n".join(eval_rows) + "\n",
        encoding="utf-8",
    )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.5,
            "uncertainty_confidence": {"low_ratio": 0.1},
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current, telemetry_dir=telemetry_dir)
    flow = (summary.get("mecpe_eval_flow") or {}).get("shadow") or {}
    assert flow.get("approved_count") == 5
    assert flow.get("pending_count") == 2
    assert flow.get("pending_ratio") == 0.4
    alert = summary.get("mecpe_alert") or {}
    assert alert.get("level") in {"WARN", "ALERT"}
    reasons = alert.get("reasons") or []
    assert "shadow_pending_ratio>=0.2" in reasons
    flow_reasons = alert.get("flow_reasons") or {}
    assert "shadow_pending_ratio>=0.2" in (flow_reasons.get("shadow") or [])
    actions = alert.get("recommended_flow_actions") or []
    assert "shadow:run_eval_queue" in actions
    thresholds = alert.get("thresholds_snapshot") or {}
    assert thresholds.get("approved_count_min") == 5
    assert thresholds.get("pending_ratio_warn") == 0.2


def test_weekly_metric_summary_sets_alert_when_canary_pending_too_old(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    old_ts = 1764547200000  # 2025-12-01T00:00:00Z
    decision_rows = []
    for idx in range(5):
        decision_rows.append(
            json.dumps(
                {
                    "schema_version": "change_decision.v0",
                    "decision_id": f"d-c-{idx}",
                    "timestamp_ms": old_ts + idx,
                    "proposal_id": f"pc-{idx}",
                    "decision": "ACCEPT_CANARY",
                    "actor": "human",
                    "reason": "approved",
                    "source_week": "2025-W51",
                }
            )
        )
    (telemetry_dir / "change_decisions-20251217.jsonl").write_text(
        "\n".join(decision_rows) + "\n",
        encoding="utf-8",
    )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.5,
            "uncertainty_confidence": {"low_ratio": 0.1},
        }
    }
    summary = _weekly_metric_summary(tmp_path, day, current, telemetry_dir=telemetry_dir)
    flow = (summary.get("mecpe_eval_flow") or {}).get("canary") or {}
    assert flow.get("approved_count") == 5
    assert flow.get("pending_count") == 5
    assert flow.get("oldest_pending_age_ms", 0) >= 14 * 24 * 60 * 60 * 1000
    alert = summary.get("mecpe_alert") or {}
    assert alert.get("level") == "ALERT"
    reasons = alert.get("reasons") or []
    assert "canary_oldest_pending_age>=14d" in reasons
    flow_reasons = alert.get("flow_reasons") or {}
    assert "canary_oldest_pending_age>=14d" in (flow_reasons.get("canary") or [])
    actions = alert.get("recommended_flow_actions") or []
    assert "canary:review_gate_decisions_urgent" in actions
    thresholds = alert.get("thresholds_snapshot") or {}
    assert thresholds.get("oldest_pending_age_warn_days") == 7
    assert thresholds.get("oldest_pending_age_alert_days") == 14


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


def test_build_change_proposal_from_mecpe_alert_emits_one_for_warn() -> None:
    weekly_metrics = {
        "mecpe_alert": {
            "level": "WARN",
            "reasons": ["yellow_days>=3"],
            "summary": {"yellow_days": 3, "ratio_max": 0.012},
        }
    }
    proposal = _build_change_proposal_from_mecpe_alert(
        weekly_metrics=weekly_metrics,
        timestamp_ms=1735689600000,
    )
    assert proposal is not None
    assert proposal["risk_level"] in ("LOW", "MED")
    assert proposal["requires_gate"] == "shadow"
    assert "trigger" in proposal and "suggested_change" in proposal


def test_build_change_proposal_from_mecpe_alert_suppressed_on_contract_break() -> None:
    rules = {
        "default_policy": {"action": "allow"},
        "risk_order": ["LOW", "MED", "HIGH"],
        "rules": [
            {
                "rule_id": "suppress_on_contract_break",
                "when": {
                    "mecpe_alert_level_in": ["WARN", "ALERT"],
                    "contract_top_type_in": ["missing_required_key", "invalid_hash_len"],
                },
                "then": {"action": "suppress", "reason": "contract_break_top_type"},
            }
        ],
    }
    weekly_metrics = {
        "mecpe_alert": {
            "level": "ALERT",
            "reasons": ["ratio_max>=0.05"],
            "summary": {"ratio_max": 0.1},
        }
    }
    proposal = _build_change_proposal_from_mecpe_alert(
        weekly_metrics=weekly_metrics,
        timestamp_ms=1735689600000,
        rules=rules,
        contract_top_type="missing_required_key",
    )
    assert proposal is None


def test_build_change_proposal_from_mecpe_alert_allowed_for_warn_shadow() -> None:
    rules = {
        "default_policy": {"action": "suppress"},
        "risk_order": ["LOW", "MED", "HIGH"],
        "rules": [
            {
                "rule_id": "allow_shadow_eval_on_warn_alert",
                "when": {
                    "mecpe_alert_level_in": ["WARN", "ALERT"],
                    "contract_top_type_not_in": ["missing_required_key", "invalid_hash_len"],
                },
                "then": {
                    "action": "allow",
                    "requires_gate": "shadow",
                    "max_risk_level": "MED",
                    "reason": "safe_shadow_eval",
                },
            }
        ],
    }
    weekly_metrics = {
        "mecpe_alert": {
            "level": "WARN",
            "reasons": ["ratio_max>=0.01"],
            "summary": {"ratio_max": 0.02},
        }
    }
    proposal = _build_change_proposal_from_mecpe_alert(
        weekly_metrics=weekly_metrics,
        timestamp_ms=1735689600000,
        rules=rules,
        contract_top_type="json_decode",
    )
    assert proposal is not None
    assert proposal["requires_gate"] == "shadow"


def test_load_mecpe_proposal_rules_missing_file_returns_safe_suppress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EQNET_MECPE_PROPOSAL_RULES", str(tmp_path / "missing.yaml"))
    rules = _load_mecpe_proposal_rules(tmp_path / "fallback.yaml")
    assert (rules.get("default_policy") or {}).get("action") == "suppress"


def test_load_mecpe_alert_thresholds_missing_file_returns_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EQNET_MECPE_ALERT_THRESHOLDS", str(tmp_path / "missing.yaml"))
    payload = _load_mecpe_alert_thresholds(tmp_path / "fallback.yaml")
    assert payload.get("approved_count_min") == 5
    assert payload.get("pending_ratio_warn") == 0.2
    assert payload.get("pending_ratio_alert") == 0.5
    assert payload.get("oldest_pending_age_warn_days") == 7
    assert payload.get("oldest_pending_age_alert_days") == 14


def test_weekly_metric_summary_applies_custom_flow_thresholds(tmp_path: Path) -> None:
    day = date(2025, 12, 17)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    # 1 approved, 1 pending -> pending_ratio=1.0
    (telemetry_dir / "change_decisions-20251217.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "change_decision.v0",
                "decision_id": "d-z-1",
                "timestamp_ms": 1734393600000,
                "proposal_id": "pz-1",
                "decision": "ACCEPT_SHADOW",
                "actor": "human",
                "reason": "approved",
                "source_week": "2025-W51",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current = {
        "nightly_audit": {
            "lazy_rag_sat_ratio_p95": 0.5,
            "uncertainty_confidence": {"low_ratio": 0.1},
        }
    }
    summary = _weekly_metric_summary(
        tmp_path,
        day,
        current,
        telemetry_dir=telemetry_dir,
        flow_thresholds={
            "approved_count_min": 1,
            "pending_ratio_warn": 0.9,
            "pending_ratio_alert": 0.95,
            "oldest_pending_age_warn_days": 30,
            "oldest_pending_age_alert_days": 60,
        },
    )
    alert = summary.get("mecpe_alert") or {}
    reasons = alert.get("reasons") or []
    assert "shadow_pending_ratio>=0.9" in reasons
    assert "shadow_pending_ratio>=0.95" in reasons
    thresholds = alert.get("thresholds_snapshot") or {}
    assert thresholds.get("approved_count_min") == 1
    assert thresholds.get("pending_ratio_warn") == 0.9
    assert thresholds.get("pending_ratio_alert") == 0.95
    assert thresholds.get("oldest_pending_age_warn_days") == 30
    assert thresholds.get("oldest_pending_age_alert_days") == 60


def test_proposal_allowed_by_rules_suppresses_risk_too_high() -> None:
    rules = {
        "default_policy": {"action": "allow"},
        "risk_order": ["LOW", "MED", "HIGH"],
        "rules": [
            {
                "rule_id": "allow_shadow_eval",
                "when": {"mecpe_alert_level_in": ["WARN", "ALERT"]},
                "then": {"action": "allow", "requires_gate": "shadow", "max_risk_level": "MED", "reason": "ok"},
            }
        ],
    }
    allowed, reason = _proposal_allowed_by_rules(
        rules=rules,
        mecpe_alert_level="ALERT",
        contract_top_type=None,
        requires_gate="shadow",
        risk_level="HIGH",
    )
    assert allowed is False
    assert "risk_too_high" in reason
