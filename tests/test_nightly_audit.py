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
