from pathlib import Path

from inner_os.integration_hooks import IntegrationHooks
from inner_os.memory_core import MemoryCore
from inner_os.service import InnerOSService


def test_inner_os_service_roundtrip(tmp_path: Path) -> None:
    service = InnerOSService(
        IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "service_memory.jsonl"))
    )

    pre = service.pre_turn_update(
        {
            "user_input": {"text": "hello"},
            "sensor_input": {"voice_level": 0.4, "body_stress_index": 0.3, "person_count": 1, "autonomic_balance": 0.48},
            "local_context": {"last_gate_context": {"valence": 0.1, "arousal": 0.2}, "relational_world": {"resource_scarcity": 0.48, "institutional_pressure": 0.52, "ritual_signal": 0.31}},
            "current_state": {"current_energy": 0.8},
            "safety_bias": 0.1,
        }
    )
    assert pre["state"]["stress"] >= 0.0
    assert "terrain" in pre["interaction_hints"]
    assert "voice_level" in pre["interaction_hints"]
    assert pre["interaction_hints"]["environment_pressure"]["summary"]
    assert pre["interaction_hints"]["relationship"]["attachment"] >= 0.0
    assert pre["interaction_hints"]["personality"]["caution_bias"] >= 0.0
    assert pre["interaction_hints"]["persistence"]["continuity_score"] >= 0.0

    recall = service.memory_recall(
        {
            "text_cue": "harbor",
            "visual_cue": "signboard and slope",
            "current_state": pre["state"],
            "retrieval_summary": {"backend": "sse"},
        }
    )
    assert recall["ignition_hints"]["recall_active"] is True
    assert recall["ignition_hints"]["environment_pressure"]["summary"]

    gate = service.response_gate(
        {
            "draft": {"text": "tentative reply"},
            "current_state": {**pre["state"], "mode": "reality"},
            "safety_signals": {"safety_bias": 0.2},
        }
    )
    assert gate["schema"] == "inner_os_response_gate_result/v1"
    assert gate["allowed_surface_intensity"] > 0.0

    post = service.post_turn_update(
        {
            "user_input": {"text": "hello"},
            "output": {"reply_text": "soft reply"},
            "current_state": {**pre["state"], "memory_anchor": "harbor"},
            "memory_write_candidates": [recall["recall_payload"]],
            "recall_payload": recall["recall_payload"],
            "transferred_lessons": [{"kind": "transferred_learning", "summary": "pause before commitment", "confidence": 0.8, "policy_hint": "pause_and_observe_under_ambiguity"}],
        }
    )
    assert post["schema"] == "inner_os_post_turn_result/v1"
    assert post["audit_record"]["kind"] == "thin_audit"
    assert post["audit_record"]["transferred_lessons_used"] == 1
    assert post["audit_record"]["reconstructed_memory_appended"] is True
    assert post["audit_record"]["environment_pressure"]["summary"]
    assert post["audit_record"]["attachment"] >= 0.0
    assert post["audit_record"]["caution_bias"] >= 0.0
    assert post["audit_record"]["continuity_score"] >= 0.0
    assert len(post["memory_appends"]) >= 1
