from fastapi.testclient import TestClient

from apps.observer.main import app


client = TestClient(app)


def test_inner_os_pre_turn_update_route() -> None:
    response = client.post(
        "/inner-os/pre-turn-update",
        json={
            "user_input": {"text": "hello"},
            "sensor_input": {"voice_level": 0.35, "body_stress_index": 0.28},
            "local_context": {"last_gate_context": {"valence": 0.1, "arousal": 0.2}},
            "current_state": {"current_energy": 0.8},
            "safety_bias": 0.1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["schema"] == "inner_os_pre_turn_result/v1"
    assert payload["result"]["state"]["stress"] >= 0.0
    assert "terrain" in payload["result"]["interaction_hints"]


def test_inner_os_memory_recall_route() -> None:
    response = client.post(
        "/inner-os/memory-recall",
        json={
            "text_cue": "harbor",
            "visual_cue": "signboard and slope",
            "current_state": {"stress": 0.2, "temporal_pressure": 0.1},
            "retrieval_summary": {"backend": "sse"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["schema"] == "inner_os_memory_recall_result/v1"
    assert payload["result"]["recall_payload_schema"] == "inner_os_recall_payload/v1"
    assert payload["result"]["memory_evidence_bundle_schema"] == "inner_os_memory_evidence_bundle/v1"
    assert payload["result"]["ignition_hints"]["recall_active"] is True


def test_inner_os_response_gate_route() -> None:
    response = client.post(
        "/inner-os/response-gate",
        json={
            "draft": {"text": "tentative reply"},
            "current_state": {"talk_mode": "watch", "route": "conscious", "mode": "reality", "stress": 0.2},
            "safety_signals": {"safety_bias": 0.25},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["schema"] == "inner_os_response_gate_result/v1"
    assert payload["result"]["allowed_surface_intensity"] > 0.0


def test_inner_os_post_turn_update_route() -> None:
    response = client.post(
        "/inner-os/post-turn-update",
        json={
            "user_input": {"text": "hello"},
            "output": {"reply_text": "soft reply"},
            "current_state": {"memory_anchor": "harbor", "route": "watch", "talk_mode": "watch"},
            "memory_write_candidates": [{"memory_anchor": "harbor", "text": "soft reply"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["schema"] == "inner_os_post_turn_result/v1"
    assert payload["result"]["audit_record"]["kind"] == "thin_audit"
