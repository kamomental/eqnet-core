import json
from pathlib import Path

from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit


def test_generate_audit(tmp_path: Path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)

    sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "scenario-demo",
        "source_loop": "hub",
        "boundary": {"score": 0.6, "reasons": {"hazard_level": 0.6}},
        "prospection": {"accepted": False, "jerk": 0.9, "temperature": 0.8},
        "policy": {"throttles": {"directiveness_cap": True}},
        "invariants": {"TRACE_001": False},
    }
    (day_dir / "hub-123.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    out_path = generate_audit(
        NightlyAuditConfig(
            trace_root=trace_root,
            out_root=tmp_path / "audit",
            date_yyyy_mm_dd=day,
            boundary_threshold=0.5,
            think_log_path=None,
            act_log_path=None,
        )
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["date"] == day
    assert data["schema_version"] == "nightly_audit_v1"
    assert data["prospection"]["reject_rate"] == 1.0
    assert data["prospection"]["accept_rate"] == 0.0
    assert data["policy"]["offer_throttle_counts"]["directiveness_cap"] == 1
    assert data["boundary"]["span_count"] == 1
    assert data["health"]["status"] in {"YELLOW", "RED"}
    assert data["evidence"]["invariants"]["warn_failures"]
    separation = data.get("separation") or {}
    assert separation.get("enabled") is False
    assert separation.get("status") == "NOT_INSTALLED"


def test_generate_audit_separation_installed_baseline_when_logs_have_guard_signals(tmp_path: Path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)

    sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "scenario-demo",
        "source_loop": "hub",
        "boundary": {"score": 0.6, "reasons": {"hazard_level": 0.6}},
        "prospection": {"accepted": False, "jerk": 0.9, "temperature": 0.8},
        "policy": {"throttles": {"directiveness_cap": True}},
        "invariants": {"TRACE_001": False},
    }
    (day_dir / "hub-123.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    think_log = tmp_path / "logs" / "think_log.jsonl"
    act_log = tmp_path / "logs" / "act_log.jsonl"
    think_log.parent.mkdir(parents=True, exist_ok=True)
    think_row = {
        "trace_id": "t-think",
        "turn_id": "turn-think",
        "memory_kind": "imagery",
        "meta": {"audit_event": "PROMOTION_GUARD_BLOCKED"},
    }
    act_row = {
        "trace_id": "t-act",
        "turn_id": "turn-act",
        "memory_kind": "experience",
        "meta": {"promotion_evidence_event_id": "obs-1"},
    }
    think_log.write_text(json.dumps(think_row) + "\n", encoding="utf-8")
    act_log.write_text(json.dumps(act_row) + "\n", encoding="utf-8")

    out_path = generate_audit(
        NightlyAuditConfig(
            trace_root=trace_root,
            out_root=tmp_path / "audit",
            date_yyyy_mm_dd=day,
            boundary_threshold=0.5,
            think_log_path=think_log,
            act_log_path=act_log,
        )
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    separation = data.get("separation") or {}
    checks = separation.get("checks") or {}
    assert checks.get("think_log_connected") is True
    assert checks.get("act_log_connected") is True
    assert checks.get("memory_kind_enum_enforced") is True
    assert checks.get("promotion_guard_enforced") is True
    src = checks.get("source_misattribution_events") or {}
    assert src.get("wired") is True
    assert src.get("active") is False
    assert src.get("active_count") == 0
    assert checks.get("source_misattribution_events_connected") is False
    assert separation.get("status") == "INSTALLED_BASELINE+SOURCE_RECHECK_WIRED"


def test_generate_audit_separation_source_misattribution_event_connected(tmp_path: Path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "scenario-demo",
        "source_loop": "hub",
        "boundary": {"score": 0.2},
        "prospection": {"accepted": True},
        "policy": {"throttles": {}},
        "invariants": {"TRACE_001": True},
    }
    (day_dir / "hub-123.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    think_log = tmp_path / "logs" / "think_log.jsonl"
    act_log = tmp_path / "logs" / "act_log.jsonl"
    think_log.parent.mkdir(parents=True, exist_ok=True)
    think_log.write_text(
        json.dumps(
            {
                "trace_id": "t-think",
                "turn_id": "turn-think",
                "memory_kind": "unknown",
                "audit_event": "SOURCE_FUZZY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    act_log.write_text(
        json.dumps(
            {
                "trace_id": "t-act",
                "turn_id": "turn-act",
                "memory_kind": "experience",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = generate_audit(
        NightlyAuditConfig(
            trace_root=trace_root,
            out_root=tmp_path / "audit",
            date_yyyy_mm_dd=day,
            think_log_path=think_log,
            act_log_path=act_log,
        )
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    checks = ((data.get("separation") or {}).get("checks") or {})
    src = checks.get("source_misattribution_events") or {}
    assert src.get("wired") is True
    assert src.get("active") is True
    assert int(src.get("active_count") or 0) >= 1
    assert checks.get("source_misattribution_events_connected") is True
    status = (data.get("separation") or {}).get("status", "")
    assert "SOURCE_RECHECK_ACTIVE" in status
