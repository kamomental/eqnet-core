from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from eqnet.runtime.future_contracts import (
    compute_perception_quality,
    evaluate_realtime_outcomes,
    load_perception_quality_rules,
    load_realtime_rules,
    summarize_perception_quality_breakdown,
    validate_realtime_forecast_proposal,
)
from eqnet.runtime.companion_policy import load_lifelong_companion_policy


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_realtime_rules_contract_loads() -> None:
    rules = load_realtime_rules(Path("configs/realtime_forecast_rules_v0.yaml"))
    assert rules["schema_version"] == "realtime_forecast_rules_v0"
    assert int(rules.get("ttl_sec_default") or 0) > 0


def test_validate_realtime_forecast_requires_approval_true() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-1",
        "ts_utc": _iso(now),
        "ttl_sec": 60,
        "requires_approval": False,
        "origin_channel": "dialogue",
    }
    ok, reasons = validate_realtime_forecast_proposal(proposal, now_ts_ms=int(now.timestamp() * 1000))
    assert ok is False
    assert "REQUIRES_APPROVAL_MUST_BE_TRUE" in reasons


def test_realtime_outcome_unknown_ttl_expired() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-expired",
        "ts_utc": _iso(ts),
        "ttl_sec": 30,
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": {"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
    }
    out = evaluate_realtime_outcomes(
        [proposal],
        evaluation_day_key="2026-01-02",
        today_policy_meta={"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int((ts + timedelta(minutes=10)).timestamp() * 1000),
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "UNKNOWN"
    assert "UNKNOWN_TTL_EXPIRED" in row["reason_codes"]


def test_realtime_outcome_unknown_policy_mismatch_and_idempotent() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-pol",
        "ts_utc": _iso(ts),
        "ttl_sec": 3600,
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": {"policy_fingerprint": "fp-old", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
    }
    out = evaluate_realtime_outcomes(
        [proposal, proposal],  # duplicate append case
        evaluation_day_key="2026-01-01",
        today_policy_meta={"policy_fingerprint": "fp-new", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int(ts.timestamp() * 1000),
    )
    assert len(out["outcomes"]) == 1
    row = out["outcomes"][0]
    assert row["effect_result"] == "UNKNOWN"
    assert "UNKNOWN_POLICY_MISMATCH" in row["reason_codes"]


def test_realtime_outcome_marks_origin_unknown_for_legacy_event_without_origin_channel() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    legacy_proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-legacy",
        "ts_utc": _iso(ts),
        "ttl_sec": 3600,
        "requires_approval": True,
    }
    out = evaluate_realtime_outcomes(
        [legacy_proposal],
        evaluation_day_key="2026-01-01",
        today_policy_meta={"policy_fingerprint": "fp-new", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int(ts.timestamp() * 1000),
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "UNKNOWN"
    assert "ORIGIN_UNKNOWN" in row["reason_codes"]


def test_perception_quality_priority_is_deterministic_for_same_input() -> None:
    rules = load_perception_quality_rules(Path("configs/perception_quality_rules_v0.yaml"))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = {
        "origin_channel": "dialogue",
        "ts_utc": _iso(ts),
        "ttl_sec": 300,
        "confidence": 0.6,
        "noise": 0.4,
    }
    now_ms = int((ts + timedelta(seconds=30)).timestamp() * 1000)
    q1 = compute_perception_quality(event, now_ts_ms=now_ms, rules=rules)
    q2 = compute_perception_quality(event, now_ts_ms=now_ms, rules=rules)
    assert q1["priority_score"] == q2["priority_score"]


def test_quality_breakdown_by_origin_and_kind_is_emitted() -> None:
    rules = load_perception_quality_rules(Path("configs/perception_quality_rules_v0.yaml"))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now_ms = int((ts + timedelta(seconds=60)).timestamp() * 1000)
    events = [
        {
            "kind": "REALTIME_FORECAST_PROPOSAL",
            "origin_channel": "dialogue",
            "ts_utc": _iso(ts),
            "ttl_sec": 300,
            "confidence": 0.7,
            "noise": 0.3,
        },
        {
            "kind": "REALTIME_FORECAST_PROPOSAL",
            "origin_channel": "sensor",
            "ts_utc": _iso(ts),
            "ttl_sec": 300,
            "confidence": 0.8,
            "noise": 0.2,
        },
    ]
    out = summarize_perception_quality_breakdown(events, now_ts_ms=now_ms, rules=rules)
    by_origin = out.get("quality_by_origin") or {}
    by_kind = out.get("quality_by_kind") or {}
    assert "dialogue" in by_origin
    assert "sensor" in by_origin
    assert by_origin["dialogue"]["count"] == 1
    assert by_origin["sensor"]["count"] == 1
    assert "REALTIME_FORECAST_PROPOSAL" in by_kind
    assert by_kind["REALTIME_FORECAST_PROPOSAL"]["count"] == 2


def test_realtime_outcome_blocked_when_self_sacrifice_risk_flagged() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-self-sacrifice",
        "ts_utc": _iso(ts),
        "ttl_sec": 3600,
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": {"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        "companion_constraints": {"self_sacrifice_risk": True, "reality_anchor_present": True, "non_isolation_confirmed": True},
    }
    companion_policy = load_lifelong_companion_policy(Path("configs/lifelong_companion_policy_v0.yaml"))
    out = evaluate_realtime_outcomes(
        [proposal],
        evaluation_day_key="2026-01-01",
        today_policy_meta={"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int(ts.timestamp() * 1000),
        companion_policy=companion_policy,
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "BLOCKED"
    assert "BLOCKED_SELF_SACRIFICE_FORBIDDEN" in row["reason_codes"]


def test_realtime_outcome_blocked_when_reality_anchor_missing() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-no-anchor",
        "ts_utc": _iso(ts),
        "ttl_sec": 3600,
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": {"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        "companion_constraints": {"self_sacrifice_risk": False, "reality_anchor_present": False, "non_isolation_confirmed": True},
    }
    companion_policy = load_lifelong_companion_policy(Path("configs/lifelong_companion_policy_v0.yaml"))
    out = evaluate_realtime_outcomes(
        [proposal],
        evaluation_day_key="2026-01-01",
        today_policy_meta={"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int(ts.timestamp() * 1000),
        companion_policy=companion_policy,
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "BLOCKED"
    assert "BLOCKED_REALITY_ANCHOR_REQUIRED" in row["reason_codes"]


def test_realtime_outcome_blocked_when_non_isolation_missing() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": "p-isolation-risk",
        "ts_utc": _iso(ts),
        "ttl_sec": 3600,
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": {"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        "companion_constraints": {"self_sacrifice_risk": False, "reality_anchor_present": True, "non_isolation_confirmed": False},
    }
    companion_policy = load_lifelong_companion_policy(Path("configs/lifelong_companion_policy_v0.yaml"))
    out = evaluate_realtime_outcomes(
        [proposal],
        evaluation_day_key="2026-01-01",
        today_policy_meta={"policy_fingerprint": "fp-1", "policy_version": "fsm_policy_v0", "policy_source": "configs/fsm_policy_v0.yaml"},
        now_ts_ms=int(ts.timestamp() * 1000),
        companion_policy=companion_policy,
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "BLOCKED"
    assert "BLOCKED_NON_ISOLATION_REQUIRED" in row["reason_codes"]
