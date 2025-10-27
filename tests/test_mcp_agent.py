import datetime as dt
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _write_sample_nightly(path: Path) -> None:
    payload = {
        "resonance": {
            "summary": {"corr": 0.62, "lag": 2, "energy": 0.12, "objective": 0.42, "n_eff": 120}
        },
        "vision_snapshot": {
            "events": 12,
            "detections_total": 48,
            "counts_by_kind": {"bow": 5, "anomaly": 1},
            "pose_mean": {"yaw_mean": -2.0, "pitch_mean": 0.3},
        },
        "policy_feedback": {
            "enabled": True,
            "politeness_before": 0.5,
            "politeness_after": 0.52,
            "delta": 0.02,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mcp_capabilities_and_resources(monkeypatch, tmp_path):
    nightly_path = tmp_path / "nightly.json"
    _write_sample_nightly(nightly_path)
    monkeypatch.setenv("EQNET_NIGHTLY_PATH", str(nightly_path))
    monkeypatch.setenv("EQNET_A2A_LOG_DIR", str(tmp_path / "logs"))
    import scripts.run_mcp_agent as mcp_agent_module

    importlib.reload(mcp_agent_module)
    app = mcp_agent_module.app

    client = TestClient(app)
    caps = client.get("/mcp/capabilities").json()
    assert any(tool["name"] == "a2a:turn.post" for tool in caps["tools"])

    resonance = client.get("/mcp/resources/resonance/summary").json()
    assert resonance["corr"] == 0.62
    vision = client.get("/mcp/resources/vision/snapshot").json()
    assert vision["counts_by_kind"]["bow"] == 5
    culture = client.get("/mcp/resources/culture/feedback").json()
    assert culture["delta"] == 0.02

    spec = {
        "session_id": "sess-test",
        "version": "2025.10",
        "intent": "delegate.demo",
        "from": "agent-A",
        "to": "agent-B",
        "scopes": ["read:resonance"],
        "guardrails": {"max_steps": 3, "tool_timeout_s": 5, "no_recursive": True},
        "expires_at": (dt.datetime.utcnow() + dt.timedelta(minutes=10)).replace(microsecond=0).isoformat() + "Z",
    }
    resp = client.post("/mcp/tools/a2a/contract.open", json=spec).json()
    assert resp["accepted"]
    assert resp["limits"]["max_steps"] == 3
    client.post(
        "/mcp/tools/a2a/turn.post",
        json={
            "session_id": "sess-test",
            "role": "planner",
            "actor": "agent-A",
            "payload": {"text": "hi"},
            "metadata": {"source_agent": "agent-A"},
        },
    )
    client.post(
        "/mcp/tools/a2a/score.report",
        json={
            "session_id": "sess-test",
            "turn_index": 0,
            "candidate_id": "opt-1",
            "scores": {"score": 1.0},
            "objective": "score",
            "objective_value": 1.0,
        },
    )
    snapshot = client.get("/mcp/a2a/session/sess-test").json()
    assert snapshot["turns"][0]["payload"]["text"] == "hi"
