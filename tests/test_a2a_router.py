import datetime as dt

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
