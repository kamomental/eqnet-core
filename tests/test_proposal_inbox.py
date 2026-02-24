from __future__ import annotations

import json
from pathlib import Path

from scripts.proposal_inbox import (
    _active_sync_suppression,
    append_decision,
    append_sync_execution,
    apply_sync_suppression,
    collect_inbox_items,
)


def test_collect_inbox_items_sorts_and_prioritizes_blocked_reasons() -> None:
    payload = {
        "realtime_forecast_proposals": [
            {
                "proposal_id": "rt-1",
                "requires_approval": True,
                "priority_score": 0.8,
                "reason_codes": ["LOW_FRESHNESS"],
                "priority_reason_codes": ["HIGH_NOISE"],
                "companion_constraints": {
                    "self_sacrifice_risk": True,
                    "reality_anchor_present": True,
                    "non_isolation_confirmed": True,
                },
                "policy_meta": {"policy_fingerprint": "abc12345"},
            }
        ],
        "preventive_proposals": [
            {
                "proposal_id": "pp-1",
                "requires_approval": True,
                "baseline_snapshot": {"forecast_lite_score": 0.5},
                "reason_codes": ["baseline_watch"],
            }
        ],
    }
    companion_policy = {
        "principles": {
            "mutualism": {"self_sacrifice_forbidden": True},
            "safety": {"reality_anchor_required": True, "non_isolation_required": True},
        }
    }
    items = collect_inbox_items(payload, companion_policy=companion_policy)
    assert len(items) == 2
    assert items[0].proposal_id == "rt-1"
    assert items[0].kind == "REALTIME_FORECAST_PROPOSAL"
    assert items[0].reason_codes[0] == "BLOCKED_SELF_SACRIFICE_FORBIDDEN"
    assert items[0].companion_reason_codes[0] == "BLOCKED_SELF_SACRIFICE_FORBIDDEN"


def test_collect_inbox_items_includes_sync_cue_proposal() -> None:
    payload = {
        "sync_cue_proposals": [
            {
                "kind": "SYNC_CUE_PROPOSAL",
                "proposal_id": "sync-1",
                "requires_approval": True,
                "origin_channel": "sensor",
                "ttl_sec": 10,
                "ts_utc": "2026-02-01T00:00:00Z",
                "priority_score": 0.6,
                "reason_codes": ["LOW_SYNC_R"],
                "priority_reason_codes": ["HIGH_NOISE"],
                "policy_meta": {"policy_fingerprint": "syncfp"},
            }
        ]
    }
    items = collect_inbox_items(payload, companion_policy=None)
    assert len(items) == 1
    assert items[0].kind == "SYNC_CUE_PROPOSAL"
    assert items[0].origin_channel == "sensor"
    assert items[0].ttl_sec == 10


def test_append_decision_writes_change_decision_with_reason_codes(tmp_path: Path) -> None:
    out = append_decision(
        telemetry_dir=tmp_path,
        proposal_id="rt-1",
        action="decline",
        reason_codes=["BLOCKED_NON_ISOLATION_REQUIRED", "LOW_FRESHNESS"],
        timestamp_ms=1767225600000,
    )
    assert out.exists()
    rows = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    payload = json.loads(rows[-1])
    assert payload["schema_version"] == "change_decision.v0"
    assert payload["proposal_id"] == "rt-1"
    assert payload["decision"] == "REJECT"
    assert payload["reason"] == "BLOCKED_NON_ISOLATION_REQUIRED"
    assert payload["inbox_action"] == "decline"
    assert "reason_codes" in payload


def test_append_sync_execution_writes_execution_ticket(tmp_path: Path) -> None:
    out = append_sync_execution(
        telemetry_dir=tmp_path,
        proposal={
            "proposal_id": "sync-1",
            "sync_cue": "CUE_BREATH_INHALE_ON_BEAT",
            "ttl_sec": 12,
            "origin_channel": "sensor",
            "sync_order_parameter_r": 0.45,
        },
        timestamp_ms=1767225600000,
    )
    rows = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    payload = json.loads(rows[-1])
    assert payload["schema_version"] == "sync_cue_execution.v0"
    assert payload["proposal_id"] == "sync-1"
    assert payload["cue_type"] == "CUE_BREATH_INHALE_ON_BEAT"
    assert payload["ttl_sec"] == 12


def test_apply_sync_suppression_marks_sync_items_only() -> None:
    payload = {
        "sync_cue_proposals": [
            {
                "kind": "SYNC_CUE_PROPOSAL",
                "proposal_id": "sync-1",
                "requires_approval": True,
                "origin_channel": "sensor",
                "ttl_sec": 10,
            }
        ],
        "preventive_proposals": [
            {
                "kind": "PREVENTIVE_PROPOSAL",
                "proposal_id": "pp-1",
                "requires_approval": True,
            }
        ],
    }
    items = collect_inbox_items(payload, companion_policy=None)
    patched = apply_sync_suppression(items, {"active": True, "reason_codes": ["DOWNSHIFT_HARM_STREAK"]})
    assert len(patched) == 2
    sync = [x for x in patched if x.kind == "SYNC_CUE_PROPOSAL"][0]
    prev = [x for x in patched if x.kind == "PREVENTIVE_PROPOSAL"][0]
    assert sync.status == "SUPPRESSED"
    assert "DOWNSHIFT_HARM_STREAK" in sync.suppression_reason_codes
    assert prev.status == "PENDING"


def test_active_sync_suppression_reads_latest_downshift(tmp_path: Path) -> None:
    path = tmp_path / "sync_downshifts-20260201.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_ms": 1000, "cooldown_until_ts_ms": 1500, "reason_codes": ["DOWNSHIFT_UNKNOWN_STREAK"]}),
                json.dumps({"timestamp_ms": 2000, "cooldown_until_ts_ms": 5000, "reason_codes": ["DOWNSHIFT_HARM_STREAK"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = _active_sync_suppression(tmp_path, now_ts_ms=3000)
    assert state["active"] is True
    assert "DOWNSHIFT_HARM_STREAK" in (state.get("reason_codes") or [])
