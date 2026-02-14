from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_trace(path: Path) -> None:
    records = [
        {
            "schema_version": "trace_v1",
            "timestamp_ms": 1,
            "event_type": "decision_cycle",
            "world_type": "community",
            "boundary": {
                "reasons": {
                    "risk": 0.22,
                    "uncertainty": 0.2,
                    "drive_norm": 0.72,
                }
            },
            "prospection": {"accepted": True},
            "decision_reason": "free-text should not be exposed",
        },
        {
            "schema_version": "trace_v1",
            "timestamp_ms": 2,
            "event_type": "decision_cycle",
            "world_type": "community",
            "boundary": {
                "reasons": {
                    "risk": 0.78,
                    "uncertainty": 0.66,
                    "drive_norm": 0.31,
                }
            },
            "prospection": {"accepted": False},
            "decision_reason": "another plaintext",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_make_replay_outputs_growth_expression_and_digest(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    out_file = tmp_path / "replay.json"
    _write_trace(trace_file)

    cmd = [
        sys.executable,
        "docs/replay/make_replay.py",
        "--trace_dir",
        str(trace_file),
        "--out",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "replay_payload.v1"
    events = payload["events"]
    assert len(events) == 2

    first = events[0]
    second = events[1]

    assert first["expression_diff"]["face"]["id"] in {"smile", "neutral", "surprise", "jitome", "tired"}
    assert "bond" in first["growth_state"]["axes"]
    assert "stability" in first["growth_state"]["axes"]
    assert "curiosity" in first["growth_state"]["axes"]

    assert first["decision_reason"] is None
    assert isinstance(first["decision_reason_digest"], str)
    assert first["decision_reason_digest"] != ""
    assert second["decision_reason"] is None

    assert "culture_state" in first
    assert "axes" in first["culture_state"]
    assert "trust" in first["culture_state"]["axes"]
    assert "agent_society" in first
    assert "metrics" in first["agent_society"]
    assert "cohesion" in first["agent_society"]["metrics"]
