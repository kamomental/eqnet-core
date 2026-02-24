from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from eqnet.runtime.future_contracts import evaluate_imagery_events, load_imagery_policy, validate_imagery_event


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_imagery_policy_contract_loads() -> None:
    policy = load_imagery_policy(Path("configs/imagery_policy_v0.yaml"))
    assert policy["schema_version"] == "imagery_policy_v0"
    assert int(policy.get("ttl_sec_default") or 0) > 0


def test_validate_imagery_requires_guard_and_unverified_and_no_policy_update() -> None:
    event = {
        "kind": "IMAGERY_HYPOTHESIS",
        "schema_version": "imagery_event_v0",
        "imagery_id": "i-1",
        "origin_channel": "imagery",
        "source": "imagery",
        "factuality": "verified",
        "contamination_guard": False,
        "policy_update_allowed": True,
    }
    ok, reasons = validate_imagery_event(event)
    assert ok is False
    assert "FACTUALITY_MUST_BE_UNVERIFIED" in reasons
    assert "CONTAMINATION_GUARD_REQUIRED" in reasons
    assert "POLICY_UPDATE_FORBIDDEN" in reasons


def test_imagery_decay_after_ttl() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = {
        "kind": "IMAGERY_HYPOTHESIS",
        "schema_version": "imagery_event_v0",
        "imagery_id": "i-decay",
        "origin_channel": "imagery",
        "source": "imagery",
        "factuality": "unverified",
        "contamination_guard": True,
        "policy_update_allowed": False,
        "ts_utc": _iso(ts),
        "ttl_sec": 10,
    }
    out = evaluate_imagery_events(
        [event],
        now_ts_ms=int((ts + timedelta(minutes=1)).timestamp() * 1000),
        evaluation_day_key="2026-01-01",
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "DECAYED"
    assert "IMAGERY_DECAYED" in row["reason_codes"]


def test_imagery_legacy_event_without_origin_channel_is_marked_origin_unknown() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = {
        "kind": "IMAGERY_HYPOTHESIS",
        "schema_version": "imagery_event_v0",
        "imagery_id": "i-legacy",
        "source": "imagery",
        "factuality": "unverified",
        "contamination_guard": True,
        "policy_update_allowed": False,
        "ts_utc": _iso(ts),
        "ttl_sec": 10,
    }
    out = evaluate_imagery_events(
        [event],
        now_ts_ms=int(ts.timestamp() * 1000),
        evaluation_day_key="2026-01-01",
    )
    row = out["outcomes"][0]
    assert row["effect_result"] == "UNKNOWN"
    assert "ORIGIN_UNKNOWN" in row["reason_codes"]
