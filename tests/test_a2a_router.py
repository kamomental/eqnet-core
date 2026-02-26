import datetime as dt
import json

from ops.a2a_router import A2ARouter


def test_router_contract_turn_score(tmp_path, monkeypatch):
    router = A2ARouter(log_dir=tmp_path)
    expires = (dt.datetime.utcnow() + dt.timedelta(minutes=5)).replace(microsecond=0).isoformat() + "Z"
    spec = {
        "session_id": "sess-1",
        "version": "2025.10",
        "intent": "delegate.test",
        "from": "agent-A",
        "to": "agent-B",
        "scopes": ["read:resonance"],
        "guardrails": {"max_steps": 2, "tool_timeout_s": 5, "no_recursive": True},
        "expires_at": expires,
        "cost_budget_usd": 0.01,
    }
    opened = router.open_contract(spec)
    assert opened["accepted"] is True
    assert opened["limits"]["max_steps"] == 2

    turn = router.append_turn(
        {
            "session_id": "sess-1",
            "role": "planner",
            "actor": "agent-A",
            "payload": {"text": "hello"},
            "metadata": {"source_agent": "agent-A"},
        }
    )
    assert turn["turn_index"] == 0
    assert turn["remaining_steps"] == 1

    router.record_score(
        {
            "session_id": "sess-1",
            "turn_index": 0,
            "candidate_id": "plan#1",
            "scores": {"safety": 0.9, "cost": 0.2},
            "objective": "0.7*safety-0.3*cost",
            "objective_value": 0.55,
            "actor": "agent-B",
        }
    )

    snapshot = router.session_snapshot("sess-1")
    assert snapshot["remaining_steps"] == 1
    assert snapshot["turns"][0]["payload"]["text"] == "hello"
    assert snapshot["scores"][0]["candidate_id"] == "plan#1"
    assert snapshot["scores"][0]["objective_value"] == 0.55

    closed = router.close("sess-1", reason="completed")
    assert closed["reason"] == "completed"


def test_router_blocks_disallowed_tool_call(tmp_path):
    router = A2ARouter(log_dir=tmp_path)
    expires = (dt.datetime.utcnow() + dt.timedelta(minutes=5)).replace(microsecond=0).isoformat() + "Z"
    spec = {
        "session_id": "sess-tool-1",
        "version": "2025.10",
        "intent": "delegate.test",
        "from": "agent-A",
        "to": "agent-B",
        "scopes": ["read:resonance"],
        "guardrails": {"max_steps": 2, "tool_timeout_s": 5, "no_recursive": True},
        "expires_at": expires,
        "metadata": {"tool_policy": {"disallow_tools": ["web.fetch"]}},
    }
    opened = router.open_contract(spec)
    assert opened["accepted"] is True

    out = router.append_turn(
        {
            "session_id": "sess-tool-1",
            "role": "actor",
            "actor": "agent-A",
            "payload": {"tool_name": "web.fetch", "args": {"url": "https://example.com"}},
        }
    )
    assert out["status"] == "blocked"
    assert out["blocked"] is True
    assert "ONLINE_DELTA_TOOL_BLOCKED" in out["reason_codes"]
    snapshot = router.session_snapshot("sess-tool-1")
    assert snapshot["remaining_steps"] == 2
    assert len(snapshot["turns"]) == 0


def test_router_disallow_wins_over_allow_on_tool_conflict(tmp_path):
    router = A2ARouter(log_dir=tmp_path)
    expires = (dt.datetime.utcnow() + dt.timedelta(minutes=5)).replace(microsecond=0).isoformat() + "Z"
    spec = {
        "session_id": "sess-tool-2",
        "version": "2025.10",
        "intent": "delegate.test",
        "from": "agent-A",
        "to": "agent-B",
        "scopes": ["read:resonance"],
        "guardrails": {"max_steps": 2, "tool_timeout_s": 5, "no_recursive": True},
        "expires_at": expires,
        "metadata": {"tool_policy": {"allow_tools": ["web.fetch"]}},
    }
    router.open_contract(spec)
    out = router.append_turn(
        {
            "session_id": "sess-tool-2",
            "role": "actor",
            "actor": "agent-A",
            "payload": {"tool_name": "web.fetch"},
            "metadata": {"allow_tools": ["web.fetch"], "disallow_tools": ["web.fetch"]},
        }
    )
    assert out["status"] == "blocked"
    assert out["blocked"] is True
    assert out["tool_name"] == "web.fetch"


def test_router_blocked_tool_emits_trace_v1_event(tmp_path, monkeypatch):
    trace_root = tmp_path / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))
    router = A2ARouter(log_dir=tmp_path / "a2a")
    expires = (dt.datetime.utcnow() + dt.timedelta(minutes=5)).replace(microsecond=0).isoformat() + "Z"
    router.open_contract(
        {
            "session_id": "sess-tool-trace",
            "version": "2025.10",
            "intent": "delegate.test",
            "from": "agent-A",
            "to": "agent-B",
            "scopes": ["read:resonance"],
            "guardrails": {"max_steps": 2, "tool_timeout_s": 5, "no_recursive": True},
            "expires_at": expires,
            "metadata": {"tool_policy": {"disallow_tools": ["web.fetch"]}},
        }
    )
    out = router.append_turn(
        {
            "session_id": "sess-tool-trace",
            "role": "actor",
            "actor": "agent-A",
            "payload": {"tool_name": "web.fetch"},
            "metadata": {"online_delta_ids": ["od-1"]},
        }
    )
    assert out["status"] == "blocked"
    files = list(trace_root.rglob("*.jsonl"))
    assert files
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    event = rows[-1]
    assert event.get("event_type") == "tool_call_blocked"
    assert event.get("tool_name") == "web.fetch"
    assert "ONLINE_DELTA_TOOL_BLOCKED" in (event.get("reason_codes") or [])
