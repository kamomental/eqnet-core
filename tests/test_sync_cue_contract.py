from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from eqnet.runtime.future_contracts import (
    evaluate_sync_outcomes,
    load_sync_policy,
    load_sync_quality_rules,
    validate_sync_cue_proposal,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_sync_proposal(ts: datetime) -> dict:
    return {
        "kind": "SYNC_CUE_PROPOSAL",
        "schema_version": "sync_cue_proposal_v0",
        "proposal_id": "sync-1",
        "ts_utc": _iso(ts),
        "origin_channel": "sensor",
        "ttl_sec": 10,
        "requires_approval": True,
        "target": "breath_phase",
        "sync_order_parameter_r": 0.42,
        "sync_cue": "CUE_BREATH_INHALE_ON_BEAT",
        "reason_codes": ["LOW_SYNC_R"],
        "policy_meta": {
            "sync_policy": {
                "kind": "sync_policy",
                "policy_fingerprint": "syncfp123",
                "policy_version": "sync_policy_v0",
                "policy_source": "configs/sync_policy_v0.yaml",
            },
            "companion_policy": {
                "kind": "companion_policy",
                "policy_fingerprint": "compfp123",
                "policy_version": "lifelong_companion_policy_v0",
                "policy_source": "configs/lifelong_companion_policy_v0.yaml",
            },
        },
    }


def test_sync_policy_contract_loads() -> None:
    sync_policy = load_sync_policy(Path("configs/sync_policy_v0.yaml"))
    sync_quality = load_sync_quality_rules(Path("configs/sync_quality_rules_v0.yaml"))
    assert sync_policy["schema_version"] == "sync_policy_v0"
    assert sync_quality["schema_version"] == "sync_quality_rules_v0"


def test_validate_sync_cue_proposal_accepts_valid_payload() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(now)
    ok, reasons = validate_sync_cue_proposal(proposal, now_ts_ms=int(now.timestamp() * 1000))
    assert ok is True
    assert reasons == []


def test_validate_sync_cue_rejects_missing_ttl_and_requires_approval() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(now)
    proposal["ttl_sec"] = 0
    proposal["requires_approval"] = False
    ok, reasons = validate_sync_cue_proposal(proposal, now_ts_ms=int(now.timestamp() * 1000))
    assert ok is False
    assert "INVALID_TTL" in reasons
    assert "REQUIRES_APPROVAL_MUST_BE_TRUE" in reasons


def test_validate_sync_cue_rejects_non_sensor_origin() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(now)
    proposal["origin_channel"] = "dialogue"
    ok, reasons = validate_sync_cue_proposal(proposal, now_ts_ms=int(now.timestamp() * 1000))
    assert ok is False
    assert "ORIGIN_SENSOR_REQUIRED" in reasons


def test_validate_sync_cue_rejects_ttl_expired() -> None:
    ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(ts)
    ok, reasons = validate_sync_cue_proposal(
        proposal,
        now_ts_ms=int((ts + timedelta(seconds=30)).timestamp() * 1000),
    )
    assert ok is False
    assert "TTL_EXPIRED" in reasons


def test_sync_outcomes_idempotent_by_proposal_id_and_day() -> None:
    ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(ts)
    proposal["baseline_snapshot"] = {"r_baseline": 0.4}
    rules = load_sync_quality_rules(Path("configs/sync_quality_rules_v0.yaml"))
    out = evaluate_sync_outcomes(
        [proposal, proposal],
        evaluation_day_key="2026-02-02",
        now_ts_ms=int((ts + timedelta(seconds=1)).timestamp() * 1000),
        today_sync_policy_meta={
            "policy_fingerprint": "syncfp123",
            "policy_version": "sync_policy_v0",
            "policy_source": "configs/sync_policy_v0.yaml",
        },
        today_companion_policy_meta={
            "policy_fingerprint": "compfp123",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
        today_sync_snapshot={"sync_order_parameter_r": 0.45},
        companion_policy={},
        rules=rules,
    )
    assert len(out.get("outcomes") or []) == 1


def test_sync_outcomes_unknown_on_policy_mismatch() -> None:
    ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
    proposal = _valid_sync_proposal(ts)
    proposal["baseline_snapshot"] = {"r_baseline": 0.4}
    rules = load_sync_quality_rules(Path("configs/sync_quality_rules_v0.yaml"))
    out = evaluate_sync_outcomes(
        [proposal],
        evaluation_day_key="2026-02-02",
        now_ts_ms=int((ts + timedelta(seconds=1)).timestamp() * 1000),
        today_sync_policy_meta={
            "policy_fingerprint": "syncfp999",
            "policy_version": "sync_policy_v0",
            "policy_source": "configs/sync_policy_v0.yaml",
        },
        today_companion_policy_meta={
            "policy_fingerprint": "compfp123",
            "policy_version": "lifelong_companion_policy_v0",
            "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        },
        today_sync_snapshot={"sync_order_parameter_r": 0.45},
        companion_policy={},
        rules=rules,
    )
    row = (out.get("outcomes") or [])[0]
    assert row.get("effect_result") == "UNKNOWN"
    assert "UNKNOWN_POLICY_MISMATCH" in (row.get("reason_codes") or [])
