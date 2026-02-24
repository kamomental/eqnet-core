from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.run_nightly_audit import (
    _adaptive_fsm_snapshot,
    _build_change_proposal_from_mecpe_alert,
    _evaluate_preventive_proposals,
    _extract_nightly_metrics,
    _fsm_feature_events_from_trace,
    _forecast_lite,
    _load_forecast_effect_rules,
    _load_mecpe_alert_thresholds,
    _load_mecpe_proposal_rules,
    _load_previous_report,
    _proposal_allowed_by_rules,
    _recommended_action_code,
    _snapshot_changed_keys,
    _compute_perception_quality,
    _summarize_perception_quality,
    _summarize_perception_quality_breakdown,
    _green_priority_patch_summary,
    _load_green_bridge_policy,
    _compute_green_bridge_snapshot,
    _load_behavior_change_policy,
    _compute_behavior_change_snapshot,
    _sync_realtime_summary,
    _weekly_metric_summary,
)
from eqnet.runtime.nightly.preventive_summary import evaluate_preventive_proposals
from eqnet.runtime.nightly.realtime_forecast_summary import build_forecast_lite_bundle
from eqnet.runtime.nightly.perception_quality_summary import (
    compute_perception_quality,
    summarize_perception_quality,
    summarize_perception_quality_breakdown,
)
from eqnet.runtime.nightly.green_bridge import (
    load_green_bridge_policy,
    compute_green_bridge_snapshot,
)
from eqnet.runtime.nightly.behavior_change import (
    load_behavior_change_policy,
    compute_behavior_change_snapshot,
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


def test_forecast_lite_emits_preventive_proposal() -> None:
    forecast = _forecast_lite(
        current_metrics={
            "sat_p95": 0.7,
            "low_ratio": 0.3,
            "mecpe_contract_error_ratio": 0.02,
        },
        prev_metrics={
            "sat_p95": 0.5,
            "low_ratio": 0.1,
            "mecpe_contract_error_ratio": 0.0,
        },
        nightly_payload={},
        fsm_mode="DRIFTING",
        companion_meta={
            "policy_fingerprint": "cp-fp",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
    )
    assert isinstance(forecast["forecast_lite_score"], float)
    assert 0.0 <= float(forecast["forecast_lite_score"]) <= 1.0
    assert forecast["preventive_proposal_count"] == 1
    proposals = forecast.get("preventive_proposals") or []
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.get("kind") == "PREVENTIVE_PROPOSAL"
    assert proposal.get("requires_approval") is True
    assert proposal.get("expected_effect") == "NEXT_DAY_DETERIORATION_DOWN"
    assert proposal.get("origin_channel") == "dialogue"
    companion = ((proposal.get("policy_meta") or {}).get("companion_policy") or {})
    assert companion.get("kind") == "companion_policy"
    assert companion.get("policy_fingerprint") == "cp-fp"
    realtime = forecast.get("realtime_forecast_proposals") or []
    assert len(realtime) == 1
    assert realtime[0].get("origin_channel") == "dialogue"


def test_fsm_feature_events_from_trace_extracts_inner_loop_signals() -> None:
    rows = [
        {
            "timestamp_ms": 1000,
            "policy": {
                "observations": {
                    "hub": {
                        "delegation_mode": "shadow",
                        "delegate_status": "ok",
                        "idempotency_status": "done",
                        "memory_entropy_delta": 0.2,
                        "repair_state_after": "NON_BLAME",
                        "repair_event": "ACK",
                        "repair_reason_codes": ["USER_DISTRESS"],
                        "budget_throttle_applied": True,
                        "output_control_profile": "repair",
                        "day_key": "2025-12-17",
                        "episode_id": "ep-1",
                    }
                }
            },
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "qfp"}}},
        }
    ]
    events = _fsm_feature_events_from_trace(rows)
    assert len(events) == 1
    event = events[0]
    assert event["shadow_pass_rate"] == 1.0
    assert event["memory_entropy_delta"] == 0.2
    assert event["repair_active"] == 1.0
    assert event["budget_throttle_applied"] == 1.0
    assert event["output_control_cautious"] == 1.0
    assert event["output_control_fingerprint"] == "qfp"


def test_adaptive_fsm_snapshot_from_trace_contains_policy_metadata() -> None:
    rows = [
        {
            "timestamp_ms": 1000,
            "policy": {"observations": {"hub": {"memory_entropy_delta": 0.0}}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        }
    ]
    snapshot = _adaptive_fsm_snapshot(rows)
    assert snapshot["mode"] in {"STABLE", "DRIFTING", "DEGRADED", "RECOVERING"}
    assert isinstance(snapshot["policy_fingerprint"], str)
    assert snapshot["policy_fingerprint"]
    assert snapshot["policy_version"] == "fsm_policy_v0"
    assert "fsm_policy_v0.yaml" in snapshot["policy_source"]


def test_evaluate_preventive_proposals_marks_helped_for_next_day_improvement() -> None:
    prev_report = {
        "preventive_proposals": [
            {
                "proposal_id": "pp-1",
                "expected_effect": "NEXT_DAY_DETERIORATION_DOWN",
                "baseline_day_key": "2025-12-16",
                "baseline_snapshot": {"deterioration_score": 0.7},
                "policy_meta": {
                    "policy_fingerprint": "fp-a",
                    "policy_version": "fsm_policy_v0",
                    "policy_source": "configs/fsm_policy_v0.yaml",
                },
            }
        ]
    }
    current_metrics = {"sat_p95": 0.2, "low_ratio": 0.2, "mecpe_contract_error_ratio": 0.0}
    today_fsm_meta = {
        "mode": "RECOVERING",
        "policy_fingerprint": "fp-a",
        "policy_version": "fsm_policy_v0",
        "policy_source": "configs/fsm_policy_v0.yaml",
    }
    out = _evaluate_preventive_proposals(
        prev_report=prev_report,
        current_metrics=current_metrics,
        today_fsm_meta=today_fsm_meta,
        companion_meta={
            "policy_fingerprint": "cp-fp",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
        day_key="2025-12-17",
        rules=_load_forecast_effect_rules(Path("configs/forecast_effect_rules_v0.yaml")),
    )
    outcomes = out.get("outcomes") or []
    assert len(outcomes) == 1
    assert outcomes[0]["effect_result"] == "HELPED"
    assert "EFFECT_HELPED_DETERIORATION_DOWN" in outcomes[0]["reason_codes"]
    assert out["helped_count"] == 1


def test_evaluate_preventive_proposals_marks_unknown_on_policy_mismatch() -> None:
    prev_report = {
        "preventive_proposals": [
            {
                "proposal_id": "pp-2",
                "expected_effect": "NEXT_DAY_DETERIORATION_DOWN",
                "baseline_day_key": "2025-12-16",
                "baseline_snapshot": {"deterioration_score": 0.4},
                "policy_meta": {
                    "policy_fingerprint": "fp-old",
                    "policy_version": "fsm_policy_v0",
                    "policy_source": "configs/fsm_policy_v0.yaml",
                },
            }
        ]
    }
    current_metrics = {"sat_p95": 0.5, "low_ratio": 0.3, "mecpe_contract_error_ratio": 0.01}
    today_fsm_meta = {
        "mode": "DRIFTING",
        "policy_fingerprint": "fp-new",
        "policy_version": "fsm_policy_v0",
        "policy_source": "configs/fsm_policy_v0.yaml",
    }
    out = _evaluate_preventive_proposals(
        prev_report=prev_report,
        current_metrics=current_metrics,
        today_fsm_meta=today_fsm_meta,
        companion_meta={
            "policy_fingerprint": "cp-fp-new",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
        day_key="2025-12-17",
        rules=_load_forecast_effect_rules(Path("configs/forecast_effect_rules_v0.yaml")),
    )
    outcomes = out.get("outcomes") or []
    assert len(outcomes) == 1
    assert outcomes[0]["effect_result"] == "UNKNOWN"
    assert "EFFECT_UNKNOWN_POLICY_MISMATCH" in outcomes[0]["reason_codes"]
    assert out["unknown_count"] == 1


def test_evaluate_preventive_proposals_wrapper_matches_core_module() -> None:
    prev_report = {
        "preventive_proposals": [
            {
                "proposal_id": "pp-3",
                "expected_effect": "NEXT_DAY_DETERIORATION_DOWN",
                "baseline_day_key": "2025-12-16",
                "baseline_snapshot": {"deterioration_score": 0.4},
                "policy_meta": {
                    "policy_fingerprint": "fp-a",
                    "policy_version": "fsm_policy_v0",
                    "policy_source": "configs/fsm_policy_v0.yaml",
                },
            }
        ]
    }
    current_metrics = {"sat_p95": 0.5, "low_ratio": 0.3, "mecpe_contract_error_ratio": 0.01}
    today_fsm_meta = {
        "mode": "DRIFTING",
        "policy_fingerprint": "fp-a",
        "policy_version": "fsm_policy_v0",
        "policy_source": "configs/fsm_policy_v0.yaml",
    }
    companion_meta = {
        "policy_fingerprint": "cp-fp",
        "policy_version": "lifelong_companion_policy_v0",
        "policy_source": "configs/lifelong_companion_policy_v0.yaml",
    }
    rules = _load_forecast_effect_rules(Path("configs/forecast_effect_rules_v0.yaml"))
    wrapper = _evaluate_preventive_proposals(
        prev_report=prev_report,
        current_metrics=current_metrics,
        today_fsm_meta=today_fsm_meta,
        companion_meta=companion_meta,
        day_key="2025-12-17",
        rules=rules,
    )
    core = evaluate_preventive_proposals(
        prev_report=prev_report,
        current_metrics=current_metrics,
        today_fsm_meta=today_fsm_meta,
        companion_meta=companion_meta,
        day_key="2025-12-17",
        rules=rules,
    )
    assert wrapper == core


def test_forecast_lite_wrapper_matches_core_module() -> None:
    now_utc = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    kwargs = {
        "current_metrics": {
            "sat_p95": 0.7,
            "low_ratio": 0.3,
            "mecpe_contract_error_ratio": 0.02,
        },
        "prev_metrics": {
            "sat_p95": 0.5,
            "low_ratio": 0.1,
            "mecpe_contract_error_ratio": 0.0,
        },
        "nightly_payload": {},
        "fsm_mode": "DRIFTING",
        "day_key": "2025-12-17",
        "fsm_policy_meta": {
            "policy_fingerprint": "fp-a",
            "policy_version": "fsm_policy_v0",
            "policy_source": "configs/fsm_policy_v0.yaml",
        },
        "companion_meta": {
            "policy_fingerprint": "cp-fp",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
        "now_utc": now_utc,
    }
    wrapper = _forecast_lite(**kwargs)
    core = build_forecast_lite_bundle(**kwargs)
    assert wrapper == core


def test_compute_perception_quality_wrapper_matches_core_module() -> None:
    now_ts_ms = int(datetime(2026, 2, 1, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
    rules = {
        "channel_defaults": {"dialogue": {"ttl_sec": 300, "confidence": 0.5, "noise": 0.5}},
        "priority_weights": {"freshness": 0.5, "confidence": 0.35, "noise": 0.15},
        "thresholds": {"low_freshness_ratio": 0.25, "low_confidence": 0.35, "high_noise": 0.75},
        "reason_codes": {
            "origin_unknown": "ORIGIN_UNKNOWN",
            "low_freshness": "LOW_FRESHNESS",
            "low_confidence": "LOW_CONFIDENCE",
            "high_noise": "HIGH_NOISE",
        },
    }
    event = {
        "origin_channel": "dialogue",
        "ts_utc": "2026-02-01T23:58:00+00:00",
        "kind": "REALTIME_FORECAST_PROPOSAL",
    }
    wrapper = _compute_perception_quality(event, now_ts_ms=now_ts_ms, rules=rules)
    core = compute_perception_quality(event, now_ts_ms=now_ts_ms, rules=rules)
    assert wrapper == core


def test_summarize_perception_quality_wrapper_matches_core_module() -> None:
    now_ts_ms = int(datetime(2026, 2, 1, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
    rules = {
        "channel_defaults": {"dialogue": {"ttl_sec": 300, "confidence": 0.5, "noise": 0.5}},
        "priority_weights": {"freshness": 0.5, "confidence": 0.35, "noise": 0.15},
        "thresholds": {"low_freshness_ratio": 0.25, "low_confidence": 0.35, "high_noise": 0.75},
        "reason_codes": {
            "origin_unknown": "ORIGIN_UNKNOWN",
            "low_freshness": "LOW_FRESHNESS",
            "low_confidence": "LOW_CONFIDENCE",
            "high_noise": "HIGH_NOISE",
        },
    }
    events = [
        {"origin_channel": "dialogue", "ts_utc": "2026-02-01T23:58:00+00:00", "kind": "REALTIME_FORECAST_PROPOSAL"},
        {"origin_channel": "dialogue", "ts_utc": "2026-02-01T23:50:00+00:00", "kind": "REALTIME_FORECAST_PROPOSAL"},
    ]
    wrapper_summary = _summarize_perception_quality(events, now_ts_ms=now_ts_ms, rules=rules)
    core_summary = summarize_perception_quality(events, now_ts_ms=now_ts_ms, rules=rules)
    assert wrapper_summary == core_summary
    wrapper_breakdown = _summarize_perception_quality_breakdown(events, now_ts_ms=now_ts_ms, rules=rules)
    core_breakdown = summarize_perception_quality_breakdown(events, now_ts_ms=now_ts_ms, rules=rules)
    assert wrapper_breakdown == core_breakdown


def test_green_priority_patch_summary_aggregates_counts_and_stats() -> None:
    proposals = [
        {"green_priority_patch": {"applied": True, "delta": 0.05, "reason_codes": ["GREEN_PRIORITY_PATCH_APPLIED"]}},
        {"green_priority_patch": {"applied": False, "delta": 0.0, "reason_codes": ["GREEN_PRIORITY_PATCH_SKIPPED_STATE_GUARD"]}},
        {"green_priority_patch": {"applied": True, "delta": 0.08, "reason_codes": ["GREEN_PRIORITY_PATCH_APPLIED"]}},
    ]
    out = _green_priority_patch_summary(proposals)
    assert out["green_priority_patch_applied_count"] == 2
    assert out["green_priority_patch_skipped_by_guard_count"] == 1
    stats = out.get("green_priority_patch_delta_stats") or {}
    assert stats.get("median") == 0.05
    assert stats.get("p95") == 0.08


def test_green_bridge_wrappers_match_core_module(tmp_path: Path) -> None:
    cfg = tmp_path / "green_bridge_policy_v0.yaml"
    cfg.write_text(
        "\n".join(
            [
                "schema_version: green_bridge_policy_v0",
                "policy_version: green_bridge_policy_v0",
                "policy_source: test",
                "enabled: true",
                "culture_resonance: 0.3",
                "tau_rate: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper_policy = _load_green_bridge_policy(cfg)
    core_policy = load_green_bridge_policy(cfg)
    assert wrapper_policy == core_policy

    kwargs = {
        "current_metrics": {"sat_p95": 0.6, "low_ratio": 0.2, "mecpe_contract_error_ratio": 0.01},
        "fsm_mode": "DRIFTING",
        "companion_policy_valid": True,
        "policy": wrapper_policy,
    }
    wrapper_snapshot = _compute_green_bridge_snapshot(**kwargs)
    core_snapshot = compute_green_bridge_snapshot(**kwargs)
    assert wrapper_snapshot == core_snapshot


def test_behavior_change_wrappers_match_core_module(tmp_path: Path) -> None:
    policy_path = tmp_path / "behavior_change_v0.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "schema_version: behavior_change_v1",
                "enabled: true",
                "window:",
                "  baseline_days: 1",
                "  compare_days: 1",
                "diff_gate:",
                "  thresholds:",
                "    harmed_rate_delta:",
                "      warn: 0.01",
                "      fail: 0.03",
                "    reject_rate_delta:",
                "      warn: 0.10",
                "      fail: 0.20",
                "  min_support:",
                "    per_signature_events: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "change_decisions-20260201.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "change_decision.v0",
                "decision_id": "d-1",
                "timestamp_ms": 1769990400000,
                "proposal_id": "p-1",
                "decision": "REJECT",
                "actor": "human",
                "reason": "test",
                "source_week": "2026-W05",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper_policy = _load_behavior_change_policy(policy_path)
    core_policy = load_behavior_change_policy(policy_path)
    assert wrapper_policy == core_policy
    kwargs = {
        "current_payload": {
            "priority_score_stats": {"median": 0.7},
            "sync_micro_harmed_rate": 0.2,
        },
        "prev_payload": {
            "priority_score_stats": {"median": 0.5},
            "sync_micro_harmed_rate": 0.1,
        },
        "telemetry_dir": telemetry_dir,
        "day": date(2026, 2, 1),
        "policy": wrapper_policy,
    }
    wrapper_snapshot = _compute_behavior_change_snapshot(**kwargs)
    core_snapshot = compute_behavior_change_snapshot(**kwargs)
    assert wrapper_snapshot == core_snapshot


def test_behavior_change_tolerance_relaxes_warn_only() -> None:
    policy = {
        "schema_version": "behavior_change_v1",
        "enabled": True,
        "window": {"baseline_days": 1, "compare_days": 1},
        "diff_gate": {
            "thresholds": {
                "harmed_rate_delta": {"warn": 0.05, "fail": 0.20},
                "reject_rate_delta": {"warn": 0.10, "fail": 0.30},
            },
            "min_support": {"per_signature_events": 0},
        },
        "decision": {
            "reject_values": ["REJECT"],
            "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
        },
        "tolerance": {
            "enabled": True,
            "apply_to": "warn_only",
            "margin_cap": 0.05,
            "recovery_alpha": 1.0,
            "harm_beta": 0.0,
            "unknown_gamma": 0.0,
            "epsilon_to_fail": 0.001,
            "initial_trust_budget": 0.5,
        },
    }
    out = _compute_behavior_change_snapshot(
        current_payload={
            "priority_score_stats": {"median": 0.7},
            "sync_micro_harmed_rate": 0.06,
            "sync_micro_helped_rate": 1.0,
            "sync_micro_unknown_rate": 0.0,
        },
        prev_payload={
            "priority_score_stats": {"median": 0.6},
            "sync_micro_harmed_rate": 0.0,
        },
        telemetry_dir=Path("telemetry"),
        day=date(2026, 2, 1),
        policy=policy,
    )
    tol = out.get("tolerance") or {}
    assert float(tol.get("margin") or 0.0) > 0.0
    eff = (tol.get("effective_warn") or {}).get("harmed_rate_delta")
    base = (tol.get("base_warn") or {}).get("harmed_rate_delta")
    assert float(eff or 0.0) > float(base or 0.0)
    assert out["gate"]["status"] in {"PASS", "WARN"}
    assert "BC_HARMED_RATE_DELTA_FAIL" not in set(out["gate"]["reason_codes"])


def test_behavior_change_tolerance_never_crosses_fail_boundary() -> None:
    policy = {
        "schema_version": "behavior_change_v1",
        "enabled": True,
        "window": {"baseline_days": 1, "compare_days": 1},
        "diff_gate": {
            "thresholds": {
                "harmed_rate_delta": {"warn": 0.01, "fail": 0.03},
                "reject_rate_delta": {"warn": 0.10, "fail": 0.20},
            },
            "min_support": {"per_signature_events": 0},
        },
        "decision": {
            "reject_values": ["REJECT"],
            "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
        },
        "tolerance": {
            "enabled": True,
            "apply_to": "warn_only",
            "margin_cap": 0.50,
            "recovery_alpha": 1.0,
            "harm_beta": 0.0,
            "unknown_gamma": 0.0,
            "epsilon_to_fail": 0.001,
            "initial_trust_budget": 1.0,
        },
    }
    out = _compute_behavior_change_snapshot(
        current_payload={
            "priority_score_stats": {"median": 0.7},
            "sync_micro_harmed_rate": 0.04,
            "sync_micro_helped_rate": 1.0,
            "sync_micro_unknown_rate": 0.0,
        },
        prev_payload={
            "priority_score_stats": {"median": 0.6},
            "sync_micro_harmed_rate": 0.0,
        },
        telemetry_dir=Path("telemetry"),
        day=date(2026, 2, 1),
        policy=policy,
    )
    tol = out.get("tolerance") or {}
    eff_harmed_warn = float(((tol.get("effective_warn") or {}).get("harmed_rate_delta")) or 0.0)
    fail_harmed = float(((tol.get("base_fail") or {}).get("harmed_rate_delta")) or 0.0)
    assert eff_harmed_warn <= fail_harmed - 0.001 + 1e-9
    assert out["gate"]["status"] == "FAIL"
    assert "BC_HARMED_RATE_DELTA_FAIL" in set(out["gate"]["reason_codes"])


def test_behavior_change_tolerance_uses_active_preset_for_warn_parameters() -> None:
    policy = {
        "schema_version": "behavior_change_v1",
        "enabled": True,
        "window": {"baseline_days": 1, "compare_days": 1},
        "diff_gate": {
            "thresholds": {
                "harmed_rate_delta": {"warn": 0.05, "fail": 0.20},
                "reject_rate_delta": {"warn": 0.10, "fail": 0.30},
            },
            "min_support": {"per_signature_events": 0},
        },
        "decision": {
            "reject_values": ["REJECT"],
            "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
        },
        "tolerance": {
            "enabled": True,
            "apply_to": "warn_only",
            "active_preset": "love",
            "preset_source": "manual",
            "margin_cap": 0.01,
            "recovery_alpha": 0.2,
            "mix_weight_sig": 0.2,
            "harm_beta": 0.0,
            "unknown_gamma": 0.0,
            "epsilon_to_fail": 0.001,
            "initial_trust_budget": 0.5,
            "presets": {
                "default": {"margin_cap": 0.01, "recovery_alpha": 0.2, "mix_weight_sig": 0.2},
                "love": {"margin_cap": 0.10, "recovery_alpha": 1.0, "mix_weight_sig": 0.9},
            },
        },
    }
    out = _compute_behavior_change_snapshot(
        current_payload={
            "priority_score_stats": {"median": 0.7},
            "sync_micro_harmed_rate": 0.06,
            "sync_micro_helped_rate": 1.0,
            "sync_micro_unknown_rate": 0.0,
        },
        prev_payload={
            "priority_score_stats": {"median": 0.6},
            "sync_micro_harmed_rate": 0.0,
        },
        telemetry_dir=Path("telemetry"),
        day=date(2026, 2, 1),
        policy=policy,
    )
    tol = out.get("tolerance") or {}
    assert str(tol.get("active_preset") or "") == "love"
    assert str(tol.get("preset_source") or "") == "manual"
    # love preset margin_cap/recovery_alpha should make margin noticeably positive.
    assert float(tol.get("margin") or 0.0) > 0.005
    eff = (tol.get("effective_warn") or {}).get("harmed_rate_delta")
    base = (tol.get("base_warn") or {}).get("harmed_rate_delta")
    assert float(eff or 0.0) > float(base or 0.0)


def test_behavior_change_signature_mix_changes_effective_warn() -> None:
    telemetry_dir = Path("telemetry_test_bc_sig")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    try:
        day = date(2026, 2, 1)
        (telemetry_dir / "change_decisions-20260201.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "schema_version": "change_decision.v0",
                            "decision_id": "d-a-1",
                            "timestamp_ms": 1769990400000,
                            "proposal_id": "p-a-1",
                            "decision": "ACCEPT_SHADOW",
                            "actor": "human",
                            "reason": "ok",
                            "source_week": "2026-W05",
                            "world_type": "alpha",
                        }
                    ),
                    json.dumps(
                        {
                            "schema_version": "change_decision.v0",
                            "decision_id": "d-b-1",
                            "timestamp_ms": 1769990400001,
                            "proposal_id": "p-b-1",
                            "decision": "REJECT",
                            "actor": "human",
                            "reason": "no",
                            "source_week": "2026-W05",
                            "world_type": "beta",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        policy = {
            "schema_version": "behavior_change_v1",
            "enabled": True,
            "window": {"baseline_days": 1, "compare_days": 1},
            "diff_gate": {
                "thresholds": {
                    "harmed_rate_delta": {"warn": 0.05, "fail": 0.20},
                    "reject_rate_delta": {"warn": 0.10, "fail": 0.30},
                },
                "min_support": {"per_signature_events": 0},
            },
            "decision": {
                "reject_values": ["REJECT"],
                "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
            },
            "signature": {"fields": ["world_type"]},
            "tolerance": {
                "enabled": True,
                "apply_to": "warn_only",
                "margin_cap": 0.05,
                "recovery_alpha": 0.4,
                "harm_beta": 0.0,
                "unknown_gamma": 0.0,
                "reject_beta": 0.4,
                "epsilon_to_fail": 0.001,
                "initial_trust_budget": 0.5,
                "mix_weight_sig": 1.0,
                "min_support_per_sig": 1,
            },
        }
        alpha = _compute_behavior_change_snapshot(
            current_payload={"world_type": "alpha", "sync_micro_harmed_rate": 0.0, "sync_micro_helped_rate": 0.0, "sync_micro_unknown_rate": 0.0},
            prev_payload={"sync_micro_harmed_rate": 0.0},
            telemetry_dir=telemetry_dir,
            day=day,
            policy=policy,
        )
        beta = _compute_behavior_change_snapshot(
            current_payload={"world_type": "beta", "sync_micro_harmed_rate": 0.0, "sync_micro_helped_rate": 0.0, "sync_micro_unknown_rate": 0.0},
            prev_payload={"sync_micro_harmed_rate": 0.0},
            telemetry_dir=telemetry_dir,
            day=day,
            policy=policy,
        )
        alpha_warn = float((((alpha.get("tolerance") or {}).get("effective_warn") or {}).get("harmed_rate_delta")) or 0.0)
        beta_warn = float((((beta.get("tolerance") or {}).get("effective_warn") or {}).get("harmed_rate_delta")) or 0.0)
        assert alpha_warn > beta_warn
        assert str(((alpha.get("tolerance") or {}).get("trust_source")) or "") == "mixed"
        assert str(((beta.get("tolerance") or {}).get("trust_source")) or "") == "mixed"
    finally:
        for path in telemetry_dir.glob("*"):
            path.unlink()
        telemetry_dir.rmdir()


def test_behavior_change_signature_falls_back_to_global_when_under_supported() -> None:
    telemetry_dir = Path("telemetry_test_bc_fallback")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    try:
        day = date(2026, 2, 1)
        (telemetry_dir / "change_decisions-20260201.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "change_decision.v0",
                    "decision_id": "d-a-1",
                    "timestamp_ms": 1769990400000,
                    "proposal_id": "p-a-1",
                    "decision": "ACCEPT_SHADOW",
                    "actor": "human",
                    "reason": "ok",
                    "source_week": "2026-W05",
                    "world_type": "alpha",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        policy = {
            "schema_version": "behavior_change_v1",
            "enabled": True,
            "window": {"baseline_days": 1, "compare_days": 1},
            "diff_gate": {
                "thresholds": {
                    "harmed_rate_delta": {"warn": 0.05, "fail": 0.20},
                    "reject_rate_delta": {"warn": 0.10, "fail": 0.30},
                },
                "min_support": {"per_signature_events": 0},
            },
            "decision": {
                "reject_values": ["REJECT"],
                "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
            },
            "signature": {"fields": ["world_type"]},
            "tolerance": {
                "enabled": True,
                "apply_to": "warn_only",
                "margin_cap": 0.05,
                "recovery_alpha": 0.0,
                "harm_beta": 0.0,
                "unknown_gamma": 0.0,
                "reject_beta": 0.4,
                "epsilon_to_fail": 0.001,
                "initial_trust_budget": 0.5,
                "mix_weight_sig": 1.0,
                "min_support_per_sig": 2,
            },
        }
        out = _compute_behavior_change_snapshot(
            current_payload={"world_type": "alpha", "sync_micro_harmed_rate": 0.0, "sync_micro_helped_rate": 0.0, "sync_micro_unknown_rate": 0.0},
            prev_payload={"sync_micro_harmed_rate": 0.0},
            telemetry_dir=telemetry_dir,
            day=day,
            policy=policy,
        )
        tol = out.get("tolerance") or {}
        assert str(tol.get("trust_source") or "") == "global_fallback_min_support"
        assert float(tol.get("effective_trust_budget") or 0.0) == float(tol.get("trust_budget_global") or 0.0)
    finally:
        for path in telemetry_dir.glob("*"):
            path.unlink()
        telemetry_dir.rmdir()


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


def test_sync_realtime_summary_aggregates_micro_and_downshift(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    day = date(2026, 2, 1)
    stamp = day.strftime("%Y%m%d")
    (telemetry_dir / f"sync_micro_outcomes-{stamp}.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"schema_version": "sync_micro_outcome.v0", "result": "HELPED"}),
                json.dumps({"schema_version": "sync_micro_outcome.v0", "result": "HARMED"}),
                json.dumps({"schema_version": "sync_micro_outcome.v0", "result": "UNKNOWN"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    day_start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    (telemetry_dir / f"sync_downshifts-{stamp}.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "sync_downshift_applied.v0",
                "timestamp_ms": day_start + 1000,
                "cooldown_until_ts_ms": day_start + 61_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = _sync_realtime_summary(telemetry_dir, day)
    assert out["sync_micro_helped_count"] == 1
    assert out["sync_micro_harmed_count"] == 1
    assert out["sync_micro_unknown_count"] == 1
    assert out["sync_downshift_applied_count"] == 1
    assert out["sync_emit_suppressed_time_ratio"] > 0.0
