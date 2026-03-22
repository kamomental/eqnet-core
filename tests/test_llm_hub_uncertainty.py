from __future__ import annotations

from typing import Optional
from types import SimpleNamespace

from emot_terrain_lab.hub.llm_hub import LLMHub
from terrain import llm as terrain_llm


class _DummyRAG:
    def __init__(self, *, context: Optional[str], sat_ratio: float) -> None:
        self._context = context
        self.last_score_diag = {"sat_ratio": sat_ratio}

    def build_context(self, user_text: str) -> Optional[str]:
        return self._context


def test_llm_hub_adds_uncertainty_meta_when_low_evidence(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    monkeypatch.setattr(hub, "_get_lazy_rag", lambda: _DummyRAG(context=None, sat_ratio=0.85))

    resp = hub.generate("test", context=None, controls={})
    assert resp.confidence < 0.6
    assert "retrieval_sparse" in resp.uncertainty_reason
    assert "score_saturation_high" in resp.uncertainty_reason
    assert "推定信頼度" in resp.text
    assert "（低）" in resp.text
    assert "関連記憶が少ない" in resp.text
    assert "スコア飽和が高い" in resp.text


def test_llm_hub_uncertainty_low_when_context_provided(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    resp = hub.generate("test", context="known context", controls={})
    assert resp.confidence >= 0.7
    assert resp.uncertainty_reason == ()
    assert "（中）" in resp.text
    assert "不確実要因: 低" in resp.text


def test_llm_hub_confidence_label_fallback_on_invalid_thresholds(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    # invalid order: low >= mid
    hub._runtime_cfg.ui.uncertainty_confidence_low_max = 0.9
    hub._runtime_cfg.ui.uncertainty_confidence_mid_max = 0.7
    resp = hub.generate("test", context="known context", controls={})
    # fallback defaults (0.54, 0.79) should classify ~0.78 as "中"
    assert "（中）" in resp.text


def test_llm_hub_confidence_threshold_fallback_emits_warning(monkeypatch, caplog):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    hub._runtime_cfg.ui.uncertainty_confidence_low_max = 0.95
    hub._runtime_cfg.ui.uncertainty_confidence_mid_max = 0.70
    caplog.set_level("WARNING")

    _ = hub.generate("test", context="known context", controls={})
    assert "llm_hub.confidence_threshold_fallback" in caplog.text


def test_llm_hub_injects_inner_os_policy_prompt(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "0")
    seen = {}

    def fake_chat(system_prompt, prompt, **kwargs):
        seen["system_prompt"] = system_prompt
        seen["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(terrain_llm, "chat_text", fake_chat)
    hub = LLMHub()

    _ = hub.generate(
        "Can you stay with what is visible first?",
        context="known context",
        controls={},
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "repair_then_attune",
            "dialogue_order": [
                "open:name_overreach_and_reduce_force",
                "follow:invite_visible_state",
                "close:hold_space",
            ],
            "do_not_cross": ["overinterpret"],
            "shell_guidance": ["opening:held", "return:careful_return"],
        },
        action_posture={
            "engagement_mode": "repair",
            "outcome_goal": "restore_contact_without_pressure",
            "boundary_mode": "softened",
            "attention_target": "person:user",
            "memory_write_priority": "relation_episode",
        },
        actuation_plan={
            "execution_mode": "repair_contact",
            "primary_action": "soft_repair",
            "reply_permission": "speak_briefly",
            "wait_before_action": "measured",
            "action_queue": ["name_overreach", "reduce_force", "reopen_carefully"],
        },
        conversational_objects={
            "primary_object_id": "object:0",
            "deferred_object_ids": ["object:1"],
            "objects": [
                {"object_id": "object:0", "label": "what is visible first"},
                {"object_id": "object:1", "label": "deeper meaning"},
            ],
        },
        object_operations={
            "primary_operation_id": "operation:0",
            "question_budget": 0,
            "question_pressure": 0.72,
            "defer_dominance": 0.81,
            "guarded_topics": ["deeper meaning"],
            "operations": [
                {
                    "operation_id": "operation:0",
                    "operation_kind": "hold_without_probe",
                    "target_label": "what is visible first",
                }
            ],
        },
        interaction_effects={
            "effects": [
                {
                    "effect_id": "effect:0",
                    "effect_kind": "preserve_self_pacing",
                    "target_label": "what is visible first",
                    "intensity": 0.74,
                }
            ]
        },
        interaction_judgement_summary={
            "observed_lines": ["The other person said: Can you stay with what is visible first?"],
            "inferred_lines": ["System estimates that the other person has low room for detail right now."],
            "selected_object_lines": ["System is treating 'what is visible first' as the current object."],
            "deferred_object_lines": ["System is not opening 'deeper meaning' yet."],
            "operation_lines": ["System is holding the visible part without probing."],
            "intended_effect_lines": ["System wants the other person to keep their own pace."],
        },
        interaction_condition_report={
            "scene_lines": ["System sees this as a scene where distance should not close too quickly."],
            "relation_lines": ["System sees a repair-sensitive relation here."],
            "memory_lines": ["System is keeping the shared thread visible."],
            "integration_lines": ["So system is leaving room instead of probing deeper."],
        },
        content_sequence=[
            {"act": "acknowledge_overreach", "text": "I may have come in a little too directly."},
            {"act": "visible_anchor", "text": "Let me slow down and stay with what is actually visible first."},
        ],
        surface_profile={
            "opening_delay": "long",
            "response_length": "short",
            "certainty_style": "careful",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
        },
        utterance_stance="measured_check_in",
    )

    assert "[inner_os_policy]" in seen["prompt"]
    assert '"conversation_contract"' in seen["prompt"]
    assert '"primary_object": "what is visible first"' in seen["prompt"]
    assert '"focus_now": "what is visible first"' in seen["prompt"]
    assert '"primary_operation": "hold_without_probe"' in seen["prompt"]
    assert '"response_action_now"' in seen["prompt"]
    assert '"question_pressure": 0.72' in seen["prompt"]
    assert '"effect": "preserve_self_pacing"' in seen["prompt"]
    assert '"wanted_effect_on_other"' in seen["prompt"]
    assert '"do_not_open_yet": [' in seen["prompt"]
    assert '"condition_summary"' in seen["prompt"]
    assert '"action_posture"' in seen["prompt"]
    assert '"actuation_plan"' in seen["prompt"]
    assert '"content_sequence"' in seen["prompt"]
    assert '"utterance_stance": "measured_check_in"' in seen["prompt"]
    assert "known context" in seen["prompt"]
    assert "Can you stay with what is visible first?" in seen["prompt"]


def test_llm_hub_exposes_model_source(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "0")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(
        terrain_llm,
        "get_llm",
        lambda: SimpleNamespace(model="qwen-3.5-instruct", model_source="cache"),
    )
    hub = LLMHub()

    resp = hub.generate("test", context="known context", controls={})

    assert resp.model == "qwen-3.5-instruct"
    assert resp.model_source == "cache"
