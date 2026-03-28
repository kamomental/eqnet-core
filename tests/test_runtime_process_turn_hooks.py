from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from emot_terrain_lab.terrain import llm as terrain_llm

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, _apply_inner_os_expression_controls
from inner_os.expression import build_expression_hints_from_gate_result
from inner_os.integration_hooks import IntegrationHooks
from inner_os.headless_runtime import HeadlessInnerOSRuntime
from inner_os.memory_core import MemoryCore


@dataclass
class _SensorSnapshot:
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _SensorHarness:
    snapshot: _SensorSnapshot


@dataclass
class _SelfHarness:
    current_energy: float = 0.8


@dataclass
class _ProcessTurnHarness:
    _integration_hooks: IntegrationHooks
    _inner_os_headless_runtime: HeadlessInnerOSRuntime = field(default_factory=HeadlessInnerOSRuntime)
    _runtime_sensors: _SensorHarness = field(
        default_factory=lambda: _SensorHarness(_SensorSnapshot({"body_stress_index": 0.25, "activity_level": 0.5}))
    )
    _last_shadow_estimate: Optional[Dict[str, Any]] = None
    _last_gate_context: Dict[str, Any] = field(default_factory=dict)
    _self_model: _SelfHarness = field(default_factory=_SelfHarness)
    _last_step_context: Optional[str] = None
    _surface_user_history: Any = field(default_factory=lambda: deque(maxlen=4))
    _surface_world_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "zone_id": "market",
            "mode": "reality",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        }
    )

    process_turn = EmotionalHubRuntime.process_turn
    _normalize_perception_summary = EmotionalHubRuntime._normalize_perception_summary
    _visual_reflection_line = EmotionalHubRuntime._visual_reflection_line
    _attach_visual_reflection = EmotionalHubRuntime._attach_visual_reflection
    _inner_os_relational_context = EmotionalHubRuntime._inner_os_relational_context
    _inner_os_working_memory_seed = EmotionalHubRuntime._inner_os_working_memory_seed
    _merge_conscious_long_term_theme = EmotionalHubRuntime._merge_conscious_long_term_theme
    _latest_nightly_long_term_theme_summary = EmotionalHubRuntime._latest_nightly_long_term_theme_summary
    _latest_nightly_identity_arc_summary = EmotionalHubRuntime._latest_nightly_identity_arc_summary
    _latest_nightly_identity_arc_registry_summary = EmotionalHubRuntime._latest_nightly_identity_arc_registry_summary
    _latest_nightly_relation_arc_summary = EmotionalHubRuntime._latest_nightly_relation_arc_summary
    _latest_nightly_relation_arc_registry_summary = EmotionalHubRuntime._latest_nightly_relation_arc_registry_summary
    _latest_nightly_group_relation_arc_summary = EmotionalHubRuntime._latest_nightly_group_relation_arc_summary
    _latest_nightly_partner_relation_summary = EmotionalHubRuntime._latest_nightly_partner_relation_summary
    _latest_nightly_partner_relation_registry_summary = EmotionalHubRuntime._latest_nightly_partner_relation_registry_summary
    _apply_inner_os_surface_policy = EmotionalHubRuntime._apply_inner_os_surface_policy
    _apply_inner_os_surface_profile = EmotionalHubRuntime._apply_inner_os_surface_profile
    _build_inner_os_llm_guidance = EmotionalHubRuntime._build_inner_os_llm_guidance
    _apply_surface_context_packet_to_content_sequence = EmotionalHubRuntime._apply_surface_context_packet_to_content_sequence
    _shape_inner_os_surface_text = EmotionalHubRuntime._shape_inner_os_surface_text
    _shape_inner_os_surface_profile_text = EmotionalHubRuntime._shape_inner_os_surface_profile_text
    _shape_inner_os_content_sequence = EmotionalHubRuntime._shape_inner_os_content_sequence
    _select_short_inner_os_sequence = EmotionalHubRuntime._select_short_inner_os_sequence
    _compact_inner_os_sequence_text = EmotionalHubRuntime._compact_inner_os_sequence_text
    _inner_os_stream_state_from_hints = EmotionalHubRuntime._inner_os_stream_state_from_hints
    _inner_os_live_followup_signals = EmotionalHubRuntime._inner_os_live_followup_signals
    _inner_os_live_style_feedback = EmotionalHubRuntime._inner_os_live_style_feedback
    _inner_os_style_alignment = EmotionalHubRuntime._inner_os_style_alignment
    _inner_os_stream_converged = EmotionalHubRuntime._inner_os_stream_converged
    _run_inner_os_live_response_loop = EmotionalHubRuntime._run_inner_os_live_response_loop
    _compose_inner_os_surface_text = EmotionalHubRuntime._compose_inner_os_surface_text
    _inner_os_surface_probe = EmotionalHubRuntime._inner_os_surface_probe
    _inner_os_surface_reopening_line = EmotionalHubRuntime._inner_os_surface_reopening_line
    _inner_os_surface_contract_state = EmotionalHubRuntime._inner_os_surface_contract_state
    _inner_os_surface_prefix = EmotionalHubRuntime._inner_os_surface_prefix
    _inner_os_surface_closing = EmotionalHubRuntime._inner_os_surface_closing
    _inner_os_surface_policy_level = EmotionalHubRuntime._inner_os_surface_policy_level
    _inner_os_surface_policy_intent = EmotionalHubRuntime._inner_os_surface_policy_intent
    _recent_surface_user_history = EmotionalHubRuntime._recent_surface_user_history
    _recent_surface_response_history = EmotionalHubRuntime._recent_surface_response_history
    _recent_dialogue_thread_history = EmotionalHubRuntime._recent_dialogue_thread_history
    _build_context_payload = EmotionalHubRuntime._build_context_payload
    _serialize_response_meta = EmotionalHubRuntime._serialize_response_meta
    _emit_inner_os_distillation_record = EmotionalHubRuntime._emit_inner_os_distillation_record
    _build_inner_os_dashboard_snapshot = EmotionalHubRuntime._build_inner_os_dashboard_snapshot
    export_inner_os_transfer_package = EmotionalHubRuntime.export_inner_os_transfer_package
    build_inner_os_model_swap_bundle = EmotionalHubRuntime.build_inner_os_model_swap_bundle
    load_inner_os_transfer_package = EmotionalHubRuntime.load_inner_os_transfer_package
    warm_start_from_transfer_package = EmotionalHubRuntime.warm_start_from_transfer_package
    build_inner_os_state_seed_from_transfer_package = EmotionalHubRuntime.build_inner_os_state_seed_from_transfer_package
    apply_inner_os_transfer_package = EmotionalHubRuntime.apply_inner_os_transfer_package
    _load_inner_os_transfer_package_from_disk = EmotionalHubRuntime._load_inner_os_transfer_package_from_disk
    _persist_inner_os_transfer_package = EmotionalHubRuntime._persist_inner_os_transfer_package
    _persist_inner_os_dashboard_snapshot = EmotionalHubRuntime._persist_inner_os_dashboard_snapshot

    def _surface_safety_bias(self) -> float:
        return 0.2

    def _surface_mode(self) -> str:
        return "reality"

    def _dominant_memory_anchor(self) -> str:
        return "harbor slope"

    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am trying to stay with what is grounded. I will avoid overreading the scene.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": [{"id": "vision-1"}]},
            perception_summary={"text": "soft evening slope and signboard"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": {
                "reply": "tentative memory reply",
                "fidelity": 0.88,
                "meta": {"source_class": "self", "audit_event": "OK"},
                "candidate": {"label": "harbor slope"},
            },
        }


@dataclass
class _DegradedHarness(_ProcessTurnHarness):
    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am keeping the reading tentative.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": []},
            perception_summary={"error": "vision_error", "detail": "500 Server Error"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": None,
        }


def test_apply_inner_os_expression_controls_softens_llm_controls() -> None:
    base = {"temperature": 0.48, "top_p": 0.92, "directness": 0.22}
    adjusted = _apply_inner_os_expression_controls(
        base,
        {"tentative_bias": 0.64, "assertiveness_cap": 0.58, "question_bias": 0.35},
    )
    assert adjusted["directness"] < base["directness"]
    assert adjusted["temperature"] < base["temperature"]
    assert adjusted["top_p"] < base["top_p"]
    assert adjusted["inner_os_tentative_bias"] == 0.64
    assert adjusted["inner_os_assertiveness_cap"] == 0.58
    assert adjusted["inner_os_question_bias"] == 0.35


def test_shape_inner_os_surface_text_shortens_check_in_text() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    shaped = harness._shape_inner_os_surface_text(
        "I am trying to stay with what is grounded. I will avoid overreading the scene.",
        intent="check_in",
    )
    assert shaped == "I am trying to stay with what is grounded."


def test_inner_os_surface_closing_depends_on_question_bias() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.12) == ""
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.34) == "Then I can answer a little more cleanly."
    assert harness._inner_os_surface_closing(intent="check_in", question_bias=0.34) == "I want to stay with this gently first."


def test_apply_inner_os_surface_policy_prefixes_clarifying_line() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert updated.text.endswith("Then I can answer a little more cleanly.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"


def test_apply_inner_os_surface_policy_keeps_opening_only_when_question_bias_is_low() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.12},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert not updated.text.endswith("Then I can answer a little more cleanly.")


def test_apply_inner_os_surface_policy_uses_check_in_prefix() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check in gently before I go further.")
    assert "I will avoid overreading the scene." not in updated.text
    assert updated.text.endswith("I want to stay with this gently first.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:check_in"


def test_process_turn_uses_response_history_for_discussion_anchor() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._surface_response_history = deque(maxlen=3)
    harness._surface_response_history.append(
        "前に触れていた「港での約束」のところから、いま話せる分だけ戻れば十分です。"
    )

    result = harness.process_turn(
        user_text="前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
        intent="check_in",
    )

    inner_os = dict(result.persona_meta.get("inner_os") or {})
    packet = dict(inner_os.get("interaction_policy_packet") or {})
    recent_dialogue_state = dict(packet.get("recent_dialogue_state") or {})
    discussion_thread_state = dict(packet.get("discussion_thread_state") or {})

    assert recent_dialogue_state.get("recent_anchor") == "港での約束"
    assert discussion_thread_state.get("topic_anchor") == "港での約束"


def test_apply_inner_os_surface_policy_can_use_interaction_policy_packet() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "response_strategy": "respectful_wait",
                "opening_move": "acknowledge_and_wait",
                "closing_move": "leave_space",
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("I can leave this with room around it.")


def test_apply_inner_os_surface_policy_can_use_object_operation_and_effects_without_strategy() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "opening_move": "acknowledge_without_probe",
                "closing_move": "",
                "primary_object_operation": {
                    "operation_kind": "hold_without_probe",
                },
                "object_operation_kinds": ["hold_without_probe", "protect_unfinished_part"],
                "interaction_effect_kinds": ["avoid_forced_reopening"],
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_apply_inner_os_surface_policy_can_use_ordered_contract_fields_without_strategy() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "primary_object_operation": {
                    "operation_kind": "hold_without_probe",
                },
                "ordered_operation_kinds": ["hold_without_probe"],
                "ordered_effect_kinds": ["preserve_self_pacing"],
                "deferred_object_labels": ["deeper meaning"],
                "question_budget": 0,
                "question_pressure": 0.62,
                "defer_dominance": 0.74,
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_apply_inner_os_surface_policy_can_use_conversation_contract_aliases_only() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "conversation_contract": {
                    "focus_now": "what is visible first",
                    "leave_closed_for_now": ["deeper meaning"],
                    "response_action_now": {
                        "primary_operation": "hold_without_probe",
                        "question_budget": 0,
                        "question_pressure": 0.62,
                        "defer_dominance": 0.74,
                        "ordered_operations": ["hold_without_probe"],
                    },
                    "ordered_effects": ["preserve_self_pacing"],
                    "wanted_effect_on_other": [
                        {"effect": "preserve_self_pacing", "target": "what is visible first", "intensity": 0.68}
                    ],
                },
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_inner_os_surface_closing_can_follow_ordered_effects_without_legacy_kind_lists() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    closing = harness._inner_os_surface_closing(
        intent="check_in",
        question_bias=0.12,
        interaction_policy={
            "primary_object_operation": {"operation_kind": "offer_small_next_step"},
            "ordered_effect_kinds": ["enable_small_next_step"],
            "question_pressure": 0.18,
            "defer_dominance": 0.12,
        },
    )
    assert closing == "We can keep the next step connected from here."


def test_inner_os_surface_closing_can_follow_conversation_contract_aliases_only() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    closing = harness._inner_os_surface_closing(
        intent="check_in",
        question_bias=0.12,
        interaction_policy={
            "conversation_contract": {
                "response_action_now": {
                    "primary_operation": "offer_small_next_step",
                    "question_pressure": 0.18,
                    "defer_dominance": 0.12,
                },
                "ordered_effects": ["enable_small_next_step"],
                "wanted_effect_on_other": [
                    {"effect": "enable_small_next_step", "target": "next step", "intensity": 0.61}
                ],
            },
        },
    )
    assert closing == "We can keep the next step connected from here."


def test_build_inner_os_llm_guidance_uses_policy_packet_and_sequence() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
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
            "surface_opening_delay": "long",
            "surface_response_length": "short",
            "surface_sentence_temperature": "steady",
            "surface_pause_insertion": "visible_pause",
            "surface_certainty_style": "careful",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
            "partner_utterance_stance": "measured_check_in",
            "interaction_judgement_view": {
                "observed_signals": [{"signal_id": "observed:0", "signal_kind": "user_text", "text": "Can you stay with what is visible first?", "confidence": 1.0}],
                "inferred_signals": [{"signal_id": "inferred:0", "signal_kind": "detail_room", "statement": "system estimates that the other person has low room for detail right now", "strength": 0.24}],
                "selected_object_labels": ["what is visible first"],
                "deferred_object_labels": ["deeper meaning"],
                "active_operation_labels": ["hold_without_probe:what is visible first"],
                "intended_effect_labels": ["preserve_self_pacing:what is visible first"],
                "cues": ["observed:1", "inferred:1"],
            },
            "interaction_judgement_summary": {
                "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                "compact_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
            },
            "interaction_condition_report": {
                "scene_lines": ["system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。"],
                "relation_lines": ["system は、この相手には丁寧さを強めた方がよいと見ています。"],
                "memory_lines": ["system は、「harbor slope」を今回の記憶の足場にしています。"],
                "integration_lines": ["そのため system は、いまは詳しく聞くより、相手に話す余地を残す方がよいと見ています。"],
                "report_lines": [
                    "場面が効いていること: system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。",
                    "相手との関係が効いていること: system は、この相手には丁寧さを強めた方がよいと見ています。",
                ],
            },
            "conversation_contract": {
                "primary_object": "what is visible first",
                "focus_now": "what is visible first",
                "do_not_open_yet": ["deeper meaning"],
                "response_action_now": {
                    "primary_operation": "hold_without_probe",
                    "question_budget": 0,
                    "question_pressure": 0.72,
                },
                "wanted_effect_on_other": [
                    {"effect": "preserve_self_pacing", "target": "what is visible first", "intensity": 0.6}
                ],
            },
            "interaction_inspection_report": {
                "shared_observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "changed_sections": [],
                "case_reports": [
                    {
                        "case_id": "current_case",
                        "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                        "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                        "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                        "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                        "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                    }
                ],
                "report_lines": [
                    "共通して観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                    "current_case の確認結果です。",
                    "観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                ],
            },
            "interaction_audit_bundle": {
                "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                "scene_lines": ["system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。"],
                "relation_lines": ["system は、この相手には丁寧さを強めた方がよいと見ています。"],
                "memory_lines": ["system は、「harbor slope」を今回の記憶の足場にしています。"],
                "integration_lines": ["そのため system は、いまは詳しく聞くより、相手に話す余地を残す方がよいと見ています。"],
                "inspection_lines": ["current_case の確認結果です。"],
                "key_metrics": {"question_budget": 0, "question_pressure": 0.72},
                "report_lines": ["観測したこと: 相手は「Can you stay with what is visible first?」と言いました。"],
            },
            "interaction_audit_casebook": {
                "cases": [
                    {
                        "observed_text": "Can you stay with what is visible first?",
                        "condition_key": "attuned_presence|familiar|harbor slope",
                        "scene_family": "attuned_presence",
                        "relation_hint": "familiar",
                        "memory_anchor": "harbor slope",
                        "judgement_summary": {
                            "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        },
                        "audit_bundle": {
                            "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        },
                    }
                ]
            },
            "interaction_audit_report": {
                "changed_sections": ["inferred_lines"],
                "metric_differences": [],
                "report_lines": [
                    "current_case の監査結果です。",
                    "観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                ],
            },
            "interaction_audit_reference_case_ids": ["reference_1"],
            "interaction_audit_reference_case_meta": {
                "reference_1": {
                    "scene_family": "attuned_presence",
                    "relation_hint": "familiar",
                    "memory_anchor": "harbor slope",
                    "condition_key": "attuned_presence|familiar|harbor slope",
                }
            },
            "contact_reflection_state": {
                "reflection_style": "reflect_only",
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.62,
                    "reopening_pull": 0.41,
                    "affective_charge": 0.58,
                }
            },
        },
        user_text="Can you stay with what is visible first?",
        intent="clarify",
    )
    assert guidance["interaction_policy"]["response_strategy"] == "repair_then_attune"
    assert guidance["utterance_stance"] == "measured_check_in"
    assert guidance["surface_profile"]["response_length"] == "short"
    assert guidance["interaction_judgement_view"]["observed_signals"]
    assert guidance["interaction_judgement_view"]["selected_object_labels"] == ["what is visible first"]
    assert guidance["interaction_judgement_summary"]["observed_lines"]
    assert guidance["interaction_judgement_summary"]["selected_object_lines"] == ["今回 system が扱う対象は「what is visible first」です。"]
    assert guidance["interaction_condition_report"]["scene_lines"]
    assert guidance["interaction_condition_report"]["report_lines"]
    assert guidance["conversation_contract"]["focus_now"] == "what is visible first"
    assert guidance["conversation_contract"]["response_action_now"]["question_budget"] == 0
    assert guidance["interaction_inspection_report"]["case_reports"]
    assert any("current_case" in line for line in guidance["interaction_inspection_report"]["report_lines"])
    assert guidance["interaction_audit_bundle"]["report_lines"]
    assert guidance["interaction_audit_bundle"]["key_metrics"]["question_budget"] == 0
    assert guidance["interaction_audit_casebook"]["cases"]
    assert guidance["interaction_audit_report"]["report_lines"]
    assert guidance["interaction_audit_reference_case_ids"] == ["reference_1"]
    assert guidance["interaction_audit_reference_case_meta"]["reference_1"]["scene_family"] == "attuned_presence"
    assert guidance["surface_context_packet"]["conversation_phase"] == "clarify"
    assert guidance["surface_context_packet"]["constraints"]["no_generic_clarification"] is True
    assert guidance["surface_context_packet"]["shared_core"]["already_shared"] == ["Can you stay with what is visible first?"]
    assert guidance["surface_context_packet"]["response_role"]["secondary"] == "reflect_only"
    sequence = guidance["content_sequence"]
    assert isinstance(sequence, list)
    assert len(sequence) >= 2
    assert sequence[0]["act"] == "acknowledge_overreach"


def test_build_inner_os_llm_guidance_prunes_thread_reopening_sequence_with_surface_context_packet() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "recent_dialogue_state": {"state": "reopening_thread", "recent_anchor": "あの約束"},
                "discussion_thread_state": {"state": "revisit_issue", "topic_anchor": "あの約束"},
                "issue_state": {"state": "pausing_issue", "issue_anchor": "あの約束"},
            },
            "surface_response_length": "short",
            "surface_cultural_register": "soft_companion",
            "surface_group_register": "one_to_one",
            "surface_sentence_temperature": "gentle",
            "turn_delta": {
                "kind": "issue_pause",
                "preferred_act": "leave_return_point_from_anchor",
                "anchor_hint": "あの約束",
            },
            "surface_context_packet": {
                "conversation_phase": "issue_pause",
                "shared_core": {
                    "anchor": "あの約束",
                    "already_shared": ["あの約束"],
                    "not_yet_shared": [],
                },
                "response_role": {
                    "primary": "leave_return_point_from_anchor",
                    "secondary": "reflect_only",
                },
                "constraints": {
                    "no_generic_clarification": True,
                    "no_advice": True,
                    "max_questions": 0,
                    "keep_thread_visible": True,
                    "prefer_return_point": True,
                    "boundary_style": "soft_hold",
                },
                "surface_profile": {
                    "response_length": "short",
                    "cultural_register": "soft_companion",
                    "group_register": "one_to_one",
                    "sentence_temperature": "gentle",
                    "surface_mode": "",
                },
                "source_state": {
                    "recent_dialogue_state": "reopening_thread",
                    "discussion_thread_state": "revisit_issue",
                    "issue_state": "pausing_issue",
                    "turn_delta_kind": "issue_pause",
                    "green_guardedness": 0.4,
                    "green_reopening_pull": 0.6,
                    "green_affective_charge": 0.3,
                },
            },
        },
        user_text="前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
        intent="check_in",
    )

    sequence = guidance["content_sequence"]
    acts = [item["act"] for item in sequence]
    assert acts == ["leave_return_point_from_anchor"]


def test_build_inner_os_llm_guidance_prunes_deep_disclosure_sequence_when_questions_are_blocked() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "recent_dialogue_state": {"state": "continuing_thread"},
                "discussion_thread_state": {"state": "discussion_unresolved"},
                "issue_state": {"state": "exploring_issue"},
            },
            "surface_response_length": "short",
            "surface_cultural_register": "soft_companion",
            "surface_group_register": "one_to_one",
            "surface_sentence_temperature": "gentle",
            "turn_delta": {
                "kind": "discussion_unresolved",
                "preferred_act": "stay_with_present_need",
            },
            "surface_context_packet": {
                "conversation_phase": "discussion_unresolved",
                "shared_core": {
                    "anchor": "",
                    "already_shared": ["本当は、あのとき助けてほしかったって、まだ言えていないんです。"],
                    "not_yet_shared": ["助けてほしかった"],
                },
                "response_role": {
                    "primary": "stay_with_present_need",
                    "secondary": "reflect_only",
                },
                "constraints": {
                    "no_generic_clarification": True,
                    "no_advice": True,
                    "max_questions": 0,
                    "keep_thread_visible": True,
                    "prefer_return_point": False,
                    "boundary_style": "soft_hold",
                },
                "surface_profile": {
                    "response_length": "short",
                    "cultural_register": "soft_companion",
                    "group_register": "one_to_one",
                    "sentence_temperature": "gentle",
                    "surface_mode": "",
                },
                "source_state": {
                    "recent_dialogue_state": "continuing_thread",
                    "discussion_thread_state": "discussion_unresolved",
                    "issue_state": "exploring_issue",
                    "turn_delta_kind": "discussion_unresolved",
                    "green_guardedness": 0.7,
                    "green_reopening_pull": 0.2,
                    "green_affective_charge": 0.8,
                },
            },
        },
        user_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        intent="check_in",
    )

    sequence = guidance["content_sequence"]
    acts = [item["act"] for item in sequence]
    assert acts == ["reflect_hidden_need", "stay_with_present_need"]


def test_apply_inner_os_surface_profile_shapes_text_and_controls() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        text="I am trying to stay with what is grounded. I will avoid overreading the scene.",
        controls_used={"mode": "watch"},
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "long",
            "surface_response_length": "short",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "visible_pause",
            "surface_certainty_style": "tentative",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "soft_return",
        },
    )
    assert updated is not None
    assert (
        updated.text.startswith("... Gently,")
        or updated.text.startswith("... I think")
        or updated.text.startswith("... gently,")
        or updated.text.startswith("... I am")
        or updated.text.startswith("... I'm here.")
    )
    assert "I will avoid overreading the scene." not in updated.text
    assert updated.controls_used["inner_os_surface_profile"]["response_length"] == "short"
    assert updated.controls_used["inner_os_surface_profile"]["return_gaze_expectation"] == "soft_return"
    assert updated.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1


def test_apply_inner_os_surface_profile_uses_live_mismatch_to_shorten_and_carefully_shape() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        text="I can describe what I notice and then follow the thread a little further.",
        controls_used={"mode": "watch"},
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "brief",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "measured",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "live_opening_pace_mismatch": 0.66,
            "live_return_gaze_mismatch": 0.62,
            "stream_shared_attention_window_mean": 0.2,
        },
    )
    assert updated is not None
    assert updated.controls_used["inner_os_surface_profile"]["response_length"] == "short"
    assert updated.controls_used["inner_os_surface_profile"]["certainty_style"] == "careful"
    assert updated.controls_used["inner_os_surface_profile"]["pause_insertion"] == "visible_pause"
    assert updated.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1



def test_shape_inner_os_surface_profile_text_applies_surface_language_profile() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    compact = harness._shape_inner_os_surface_profile_text(
        "I am trying to stay with what is grounded.",
        surface_profile={
            "banter_move": "compact_wit",
            "lexical_variation_mode": "compact_varied",
            "group_register": "one_to_one",
        },
    )
    group_text = harness._shape_inner_os_surface_profile_text(
        "I can stay with what is visible first.",
        surface_profile={
            "banter_move": "thread_soften",
            "lexical_variation_mode": "group_attuned",
            "group_register": "threaded_group",
        },
    )

    assert compact.startswith("Short version:")
    assert "I'm trying" in compact
    assert group_text.startswith("For this thread,")


def test_apply_inner_os_surface_policy_adds_relational_probe_for_check_in_afterglow() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.36, "carry_gentleness": True, "interaction_afterglow_intent": "check_in"},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert "Would it help if I stay close to what is visible first?" in updated.text
    assert updated.text.startswith("Let me check in gently before I go further.")


def test_apply_inner_os_surface_policy_adds_reopening_line_when_recovery_returns() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.3, "allow_reopening": True, "recovery_reopening": 0.31},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert "I think we can open this a little more clearly now." in updated.text


def test_process_turn_injects_inner_os_metadata(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_social_update_strength"] = 0.64
    harness._last_gate_context["inner_os_identity_update_strength"] = 0.52
    harness._last_gate_context["inner_os_interaction_afterglow"] = 0.31
    harness._last_gate_context["inner_os_interaction_afterglow_intent"] = "clarify"
    harness._last_gate_context["inner_os_association_reweighting_focus"] = "repeated_links"
    harness._last_gate_context["inner_os_association_reweighting_reason"] = "repeated_insight_trace"
    harness._last_gate_context["inner_os_insight_terrain_shape_target"] = "soft_relation"
    result = harness.process_turn(
        user_text="do you remember this new place",
        context="previous framing",
        transferred_lessons=[
            {
                "kind": "transferred_learning",
                "policy_hint": "pause_and_observe_under_ambiguity",
                "summary": "observe before approaching when ambiguity is high",
            }
        ],
    )
    assert result.metrics["inner_os/stress"] >= 0.0
    assert result.metrics["inner_os/temporal_pressure_after"] >= 0.0
    assert result.metrics["inner_os/transferred_lessons_used"] == 1.0
    assert result.metrics["inner_os/interaction_afterglow"] > 0.0
    assert result.metrics["inner_os/replay_intensity"] >= 0.0
    assert result.metrics["inner_os/anticipation_tension"] >= 0.0
    assert result.metrics["inner_os/recovery_reopening"] >= 0.0
    assert result.metrics["inner_os/consolidation_priority"] >= 0.0
    assert result.metrics["inner_os/interference_pressure"] >= 0.0
    assert result.metrics["inner_os/prospective_memory_pull"] >= 0.0
    assert result.metrics["inner_os/object_affordance_bias"] >= 0.0
    assert result.metrics["inner_os/defensive_salience"] >= 0.0
    assert result.metrics["inner_os/surface_policy_active"] == 1.0
    assert result.metrics["inner_os/surface_policy_layered"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_clarify"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_check_in"] == 0.0
    assert result.response is not None
    assert result.response.text.startswith("Let me check one small thing before I go further.")
    assert (
        "Then I can answer a little more cleanly from there." in result.response.text
        or "I can stay with what is visible first." in result.response.text
    )
    assert result.response.retrieval_summary is not None
    assert "inner_os" in result.response.retrieval_summary
    assert result.response.controls is not None
    assert "inner_os" in result.response.controls
    assert 2 <= result.response.controls["inner_os"]["live_response_steps"] <= 4
    assert result.response.controls["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.response.controls["inner_os"]["headless_actuation"]["execution_mode"] in {"attuned_contact", "defer_with_presence", "repair_contact", "shared_progression", "stabilize_boundary", "stabilize_before_contact", "open_reflection"}
    assert result.response.controls["inner_os"]["headless_actuation"]["primary_action"]
    assert result.response.controls_used["inner_os_surface_profile"]["certainty_style"] in {"direct", "tentative", "careful"}
    assert result.response.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1
    assert result.response.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"
    assert "inner_os" in result.qualia_gate
    assert result.persona_meta["inner_os"]["route"] in {"reflex", "conscious", "watch"}
    assert result.persona_meta["inner_os"]["culture_id"] == "coastal"
    assert result.persona_meta["inner_os"]["community_id"] == "harbor_collective"
    assert result.response.controls["inner_os"]["hesitation_bias"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["tentative_bias"] >= 0.0
    assert round(result.response.controls["inner_os"]["expression_hints"]["express_now"] + result.response.controls["inner_os"]["expression_hints"]["hold_back"], 4) == 1.0
    assert result.persona_meta["inner_os"]["continuity_score"] >= 0.0
    assert result.persona_meta["inner_os"]["culture_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["community_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["terrain_transition_roughness"] >= 0.0
    assert result.persona_meta["inner_os"]["meaning_pacing"] in {"steady", "slow"}
    assert result.persona_meta["inner_os"]["interaction_pacing"] in {"flow", "check"}
    assert result.persona_meta["inner_os"]["actuation_execution_mode"] in {"attuned_contact", "defer_with_presence", "repair_contact", "shared_progression", "stabilize_boundary", "stabilize_before_contact", "open_reflection"}
    assert result.persona_meta["inner_os"]["actuation_primary_action"]
    assert result.persona_meta["inner_os"]["actuation_reply_permission"] in {"speak", "speak_briefly", "hold_or_brief", "speak_forward", "speak_minimal", "speak_reflective", "speak_expand"}
    assert result.persona_meta["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.persona_meta["inner_os"]["surface_policy_intent"] == "clarify"
    assert result.response.controls["inner_os"]["expression_hints"]["surface_opening_delay"] in {"brief", "measured", "long"}
    assert result.response.controls["inner_os"]["expression_hints"]["surface_response_length"] in {"short", "balanced", "reflective", "forward_leaning"}
    assert result.response.controls["inner_os"]["expression_hints"]["surface_certainty_style"] in {"direct", "tentative", "careful"}
    assert result.response.controls["inner_os"]["expression_hints"]["stream_update_count"] >= 2
    assert result.response.controls["inner_os"]["expression_hints"]["stream_shared_attention_window_mean"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["stream_strained_pause_window_mean"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["stream_repair_window_hold"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["opening_pace_windowed"] in {"ready", "measured", "held"}
    assert result.response.controls["inner_os"]["expression_hints"]["return_gaze_expectation"] in {"soft_return", "steady_return", "careful_return", "defer_return"}
    assert result.response.controls["inner_os"]["expression_hints"]["live_opening_pace_mismatch"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["live_return_gaze_mismatch"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["live_style_alignment"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["contact_field"]["points"]
    assert result.response.controls["inner_os"]["expression_hints"]["contact_dynamics"]["stabilized_points"]
    assert result.response.controls["inner_os"]["expression_hints"]["contact_reflection_state"]["reflection_style"] in {"reflect_then_question", "reflect_only", "boundary_only"}
    assert "green_kernel_composition" in result.response.controls["inner_os"]["expression_hints"]
    assert "field" in result.response.controls["inner_os"]["expression_hints"]["green_kernel_composition"]
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["qualia"]
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["gate"]
    assert len(result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["qualia"]) == len(
        result.response.controls["inner_os"]["expression_hints"]["qualia_axis_labels"]
    )
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_estimator_health"]["trust"] >= 0.0
    assert isinstance(result.response.controls["inner_os"]["expression_hints"]["qualia_protection_grad_x"], list)
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_source"] == "shared"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_version"] >= 1
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_expected_source"] == "shared"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_expected_mismatch"] is False
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"]["trust"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"]["felt_energy"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"] == build_expression_hints_from_gate_result(
        result.qualia_gate["inner_os"],
        existing_hints=result.qualia_gate["inner_os"]["expression_hints"],
    )["qualia_planner_view"]
    assert result.response.controls["inner_os"]["expression_hints"]["access_projection"]["regions"]
    assert "access_qualia_input" in result.response.controls["inner_os"]["expression_hints"]["access_projection"]["cues"]
    assert result.response.controls["inner_os"]["expression_hints"]["access_dynamics"]["stabilized_regions"]
    assert result.response.controls["inner_os"]["expression_hints"]["conscious_workspace"]["workspace_mode"] in {"preconscious", "latent_foreground", "foreground", "guarded_foreground"}
    assert isinstance(result.response.controls["inner_os"]["expression_hints"]["conscious_workspace_actionable_slice"], list)
    assert result.response.controls["inner_os"]["expression_hints"]["conscious_workspace_ignition_phase"] in {"dormant", "priming", "ignited", "guarded"}
    assert result.response.controls["inner_os"]["expression_hints"]["conversational_objects"]["objects"]
    assert result.response.controls["inner_os"]["expression_hints"]["object_operations"]["operations"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_effects"]["effects"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_view"]["observed_signals"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_view"]["inferred_signals"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_summary"]["observed_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_summary"]["operation_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_condition_report"]["scene_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_condition_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_inspection_report"]["case_reports"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_inspection_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_bundle"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_bundle"]["key_metrics"]["question_budget"] >= 0
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_casebook"]["cases"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["affective_position"]["z_aff"]
    assert result.response.controls["inner_os"]["expression_hints"]["terrain_readout"]["active_patch_label"]
    assert result.response.controls["inner_os"]["expression_hints"]["protection_mode"]["mode"] in {"monitor", "contain", "stabilize", "repair", "shield"}
    assert result.response.controls["inner_os"]["expression_hints"]["association_graph"]["state_hint"]["link_weights"] is not None
    assert result.response.controls["inner_os"]["expression_hints"]["insight_event"]["score"]["total"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["resonance_evaluation"]["recommended_family_id"]
    assert result.response.controls["inner_os"]["expression_hints"]["resonance_evaluation"]["estimated_other_person_state"]["detail_room_level"] in {"low", "medium", "high"}
    assert result.response.controls_used["inner_os_surface_profile"]["live_opening_pace_mismatch"] >= 0.0
    assert result.response.controls_used["inner_os_surface_profile"]["live_return_gaze_mismatch"] >= 0.0
    assert result.metrics["inner_os/stream_update_count"] >= 2.0
    assert result.metrics["inner_os/headless_wait_required"] in {0.0, 1.0}
    assert result.metrics["inner_os/resonance_score"] >= 0.0
    assert result.metrics["inner_os/qualia_hint_shared"] == 1.0
    assert result.metrics["inner_os/qualia_hint_fallback"] == 0.0
    assert result.metrics["inner_os/qualia_hint_none"] == 0.0
    assert result.metrics["inner_os/overnight_association_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_terrain_shape_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_bias_alignment_visible"] == 1.0
    assert result.metrics["inner_os/memory_write_winner_margin"] >= 0.0
    assert result.metrics["inner_os/protection_mode_winner_margin"] >= 0.0
    assert result.metrics["inner_os/memory_write_mode_prior_active"] == 1.0
    assert result.metrics["inner_os/memory_write_insight_prior_active"] in {0.0, 1.0}
    assert result.metrics["inner_os/workspace_winner_margin"] >= 0.0
    assert result.metrics["inner_os/association_graph_winner_margin"] >= 0.0
    assert result.metrics["inner_os/body_recovery_guard_winner_margin"] >= 0.0
    assert result.metrics["inner_os/body_homeostasis_score"] >= 0.0
    assert result.metrics["inner_os/body_homeostasis_winner_margin"] >= 0.0
    assert result.metrics["inner_os/initiative_readiness"] >= 0.0
    assert result.metrics["inner_os/commitment_winner_margin"] >= 0.0
    assert result.metrics["inner_os/relational_style_lexical_variation_bias"] >= 0.0
    assert result.metrics["inner_os/relational_style_banter_room"] >= 0.0
    assert result.metrics["inner_os/relational_continuity_score"] >= 0.0
    assert result.metrics["inner_os/relational_continuity_winner_margin"] >= 0.0
    assert result.metrics["inner_os/relation_competition_level"] >= 0.0
    assert result.metrics["inner_os/relation_competition_winner_margin"] >= 0.0
    assert result.metrics["inner_os/social_topology_score"] >= 0.0
    assert result.metrics["inner_os/social_topology_winner_margin"] >= 0.0
    assert result.metrics["inner_os/active_relation_total_people"] >= 0.0
    assert result.metrics["inner_os/commitment_accepted_cost"] >= 0.0
    assert result.metrics["inner_os/initiative_followup_bias"] >= 0.0
    assert result.metrics["inner_os/initiative_followup_winner_margin"] >= 0.0
    assert result.metrics["inner_os/terrain_plasticity_winner_margin"] >= 0.0
    assert result.metrics["inner_os/association_reinforcement_winner_margin"] >= 0.0
    assert result.metrics["inner_os/qualia_hint_expected_mismatch"] == 0.0
    assert result.persona_meta["inner_os"]["interaction_afterglow"] > 0.0
    assert result.persona_meta["inner_os"]["interaction_afterglow_intent"] == "clarify"
    assert result.persona_meta["inner_os"]["replay_intensity"] >= 0.0
    assert result.persona_meta["inner_os"]["anticipation_tension"] >= 0.0
    assert result.persona_meta["inner_os"]["contact_dynamics_mode"] in {"fresh", "guarded_fresh", "reentrant", "guarded_reentry"}
    assert result.persona_meta["inner_os"]["access_dynamics_mode"] in {"fresh_projection", "guarded_projection", "inertial_projection", "guarded_inertial_projection"}
    assert result.persona_meta["inner_os"]["conscious_workspace_mode"] in {"preconscious", "latent_foreground", "foreground", "guarded_foreground"}
    assert isinstance(result.persona_meta["inner_os"]["conscious_workspace_reportable_slice"], list)
    assert isinstance(result.persona_meta["inner_os"]["conscious_workspace_withheld_slice"], list)
    assert isinstance(result.persona_meta["inner_os"]["conscious_workspace_actionable_slice"], list)
    assert result.persona_meta["inner_os"]["conscious_workspace_winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["conscious_workspace_slot_scores"]
    assert result.persona_meta["inner_os"]["conscious_workspace_dominant_inputs"]
    assert result.persona_meta["inner_os"]["association_graph_winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["association_graph_dominant_inputs"]
    assert result.persona_meta["inner_os"]["qualia_hint_source"] == "shared"
    assert result.persona_meta["inner_os"]["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert result.persona_meta["inner_os"]["qualia_hint_expected_source"] == "shared"
    assert result.persona_meta["inner_os"]["qualia_hint_expected_mismatch"] is False
    assert result.persona_meta["inner_os"]["association_reweighting_focus"] == "repeated_links"
    assert result.persona_meta["inner_os"]["association_reweighting_reason"] == "repeated_insight_trace"
    assert result.persona_meta["inner_os"]["insight_terrain_shape_target"] == "soft_relation"
    assert result.persona_meta["inner_os"]["overnight_bias_roles"]["association_reweighting_focus"] == "repeated_links"
    assert result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["same_turn"]["protection_mode"] in {"monitor", "contain", "stabilize", "repair", "shield"}
    assert result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]["insight_terrain_shape_target"] == "soft_relation"
    assert result.persona_meta["inner_os"]["memory_write_class_bias"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["memory_write_class_bias"]["dominant_inputs"]
    assert result.persona_meta["inner_os"]["protection_mode_decision"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["body_recovery_guard"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["body_homeostasis_state"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["homeostasis_budget_state"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["initiative_readiness"]["score"] >= 0.0
    assert result.persona_meta["inner_os"]["commitment_state"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["commitment_state"]["target"] in {"hold", "stabilize", "repair", "bond_protect", "step_forward"}
    assert result.persona_meta["inner_os"]["relational_continuity_state"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["relation_competition_state"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["social_topology_state"]["winner_margin"] >= 0.0
    assert isinstance(result.persona_meta["inner_os"]["active_relation_table"]["entries"], list)
    assert result.persona_meta["inner_os"]["initiative_followup_bias"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["terrain_plasticity_update"]["winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["association_graph_state"]["winner_margin"] >= 0.0
    assert isinstance(result.persona_meta["inner_os"]["conversational_object_labels"], list)
    assert result.persona_meta["inner_os"]["conversational_object_pressure_balance"] >= 0.0
    assert result.persona_meta["inner_os"]["object_operation_question_budget"] >= 0
    assert result.persona_meta["inner_os"]["object_operation_question_pressure"] >= 0.0
    assert result.persona_meta["inner_os"]["object_operation_defer_dominance"] >= 0.0
    assert result.persona_meta["inner_os"]["judgement_observed_count"] >= 1
    assert result.persona_meta["inner_os"]["judgement_inferred_count"] >= 1
    assert isinstance(result.persona_meta["inner_os"]["judgement_selected_object_labels"], list)
    assert isinstance(result.persona_meta["inner_os"]["judgement_active_operation_labels"], list)
    assert result.persona_meta["inner_os"]["judgement_summary_observed"]
    assert result.persona_meta["inner_os"]["judgement_summary_operations"]
    assert result.persona_meta["inner_os"]["condition_scene_lines"]
    assert result.persona_meta["inner_os"]["condition_integration_lines"]
    assert result.persona_meta["inner_os"]["inspection_report_lines"]
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["protection_mode"] in {"monitor", "contain", "stabilize", "repair", "shield"}
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["social_topology_state"] in {"ambient", "one_to_one", "threaded_group", "public_visible", "hierarchical", ""}
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["active_group_thread_total"] >= 0
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["issue_state"] in {"ambient", "naming_issue", "exploring_issue", "pausing_issue", "resolving_issue", ""}
    assert result.persona_meta["inner_os"]["discussion_thread_registry_summary"]["total_threads"] >= 0
    assert "dominant_carry_channel" in result.persona_meta["inner_os"]["continuity_summary"]
    assert result.persona_meta["inner_os"]["resonance_recommended_family"]
    assert result.persona_meta["inner_os"]["audit_report_lines"]
    assert result.persona_meta["inner_os"]["audit_key_metrics"]["question_budget"] >= 0
    assert result.persona_meta["inner_os"]["audit_casebook_case_ids"]
    assert isinstance(result.persona_meta["inner_os"]["audit_reference_case_ids"], list)
    assert isinstance(result.persona_meta["inner_os"]["audit_reference_case_meta"], dict)
    assert result.persona_meta["inner_os"]["audit_comparison_report_lines"]
    assert result.persona_meta["inner_os"]["other_person_detail_room"] in {"low", "medium", "high"}
    assert harness._last_gate_context["inner_os_conscious_residue_strength"] >= 0.0
    assert harness._last_gate_context["inner_os_initiative_followup_bias"] >= 0.0
    assert harness._last_gate_context["inner_os_initiative_followup_state"] in {"hold", "reopen_softly", "offer_next_step"}


def test_process_turn_restores_initiative_followup_bias_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_followup.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_initiative_followup_bias"] = 0.41
    harness._last_gate_context["inner_os_initiative_followup_state"] = "offer_next_step"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )
    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["initiative_followup_state"] == "offer_next_step"
    assert overnight["initiative_followup_bias"] == 0.41
    assert harness._last_gate_context["inner_os_initiative_followup_state"] in {"hold", "reopen_softly", "offer_next_step"}
    assert harness._last_gate_context["inner_os_initiative_followup_bias"] >= 0.0


def test_process_turn_restores_commitment_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_commitment.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_commitment_target_focus"] = "repair"
    harness._last_gate_context["inner_os_commitment_state_focus"] = "commit"
    harness._last_gate_context["inner_os_commitment_carry_bias"] = 0.37
    harness._last_gate_context["inner_os_commitment_followup_focus"] = "reopen_softly"
    harness._last_gate_context["inner_os_commitment_mode_focus"] = "repair"
    harness._last_gate_context["inner_os_commitment_carry_reason"] = "commit:repair"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["commitment_target_focus"] == "repair"
    assert overnight["commitment_mode_focus"] == "repair"
    assert overnight["commitment_carry_bias"] == 0.37
    assert result.metrics["inner_os/overnight_commitment_bias_active"] == 1.0
    assert harness._last_gate_context["inner_os_commitment_target_focus"] in {"hold", "stabilize", "repair", "bond_protect", "step_forward"}


def test_process_turn_restores_agenda_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_agenda.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_agenda_focus"] = "repair"
    harness._last_gate_context["inner_os_agenda_bias"] = 0.22
    harness._last_gate_context["inner_os_agenda_reason"] = "repair_window"
    harness._last_gate_context["inner_os_agenda_window_focus"] = "next_private_window"
    harness._last_gate_context["inner_os_agenda_window_bias"] = 0.18
    harness._last_gate_context["inner_os_agenda_window_reason"] = "wait_for_private_window"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["agenda_focus"] == "repair"
    assert overnight["agenda_bias"] == 0.22
    assert overnight["agenda_reason"] == "repair_window"
    assert overnight["agenda_window_focus"] == "next_private_window"
    assert overnight["agenda_window_bias"] == 0.18
    assert overnight["agenda_window_reason"] == "wait_for_private_window"
    assert result.metrics["inner_os/overnight_agenda_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_agenda_window_bias_active"] == 1.0
    assert result.metrics["inner_os/agenda_winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["agenda_state"]["winner_margin"] >= 0.0


def test_process_turn_restores_learning_and_social_experiment_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_learning_mode.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_learning_mode_focus"] = "repair_probe"
    harness._last_gate_context["inner_os_learning_mode_carry_bias"] = 0.16
    harness._last_gate_context["inner_os_social_experiment_focus"] = "repair_signal_probe"
    harness._last_gate_context["inner_os_social_experiment_carry_bias"] = 0.14
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["learning_mode_focus"] == "repair_probe"
    assert overnight["learning_mode_carry_bias"] == 0.16
    assert overnight["social_experiment_focus"] == "repair_signal_probe"
    assert overnight["social_experiment_carry_bias"] == 0.14
    assert result.metrics["inner_os/overnight_learning_mode_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_social_experiment_bias_active"] == 1.0
    assert result.metrics["inner_os/learning_mode_winner_margin"] >= 0.0
    assert result.metrics["inner_os/social_experiment_winner_margin"] >= 0.0


def test_process_turn_restores_body_and_relational_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_body_rel.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_body_homeostasis_focus"] = "recovering"
    harness._last_gate_context["inner_os_body_homeostasis_carry_bias"] = 0.16
    harness._last_gate_context["inner_os_homeostasis_budget_focus"] = "recovering"
    harness._last_gate_context["inner_os_homeostasis_budget_bias"] = 0.09
    harness._last_gate_context["inner_os_relational_continuity_focus"] = "reopening"
    harness._last_gate_context["inner_os_relational_continuity_carry_bias"] = 0.14
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["body_homeostasis_focus"] == "recovering"
    assert overnight["body_homeostasis_carry_bias"] == 0.16
    assert overnight["homeostasis_budget_focus"] == "recovering"
    assert overnight["homeostasis_budget_bias"] == 0.09
    assert overnight["relational_continuity_focus"] == "reopening"
    assert overnight["relational_continuity_carry_bias"] == 0.14
    assert result.metrics["inner_os/overnight_body_homeostasis_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_homeostasis_budget_active"] == 1.0
    assert result.metrics["inner_os/overnight_relational_continuity_bias_active"] == 1.0
    assert result.persona_meta["inner_os"]["homeostasis_budget_state"]["winner_margin"] >= 0.0
    assert result.metrics["inner_os/homeostasis_budget_winner_margin"] >= 0.0


def test_process_turn_restores_expressive_style_history_and_banter_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_style_history.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_expressive_style_history_focus"] = "warm_companion"
    harness._last_gate_context["inner_os_expressive_style_history_bias"] = 0.08
    harness._last_gate_context["inner_os_banter_style_focus"] = "gentle_tease"
    harness._last_gate_context["inner_os_lexical_variation_carry_bias"] = 0.11
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["expressive_style_history_focus"] == "warm_companion"
    assert overnight["expressive_style_history_bias"] == 0.08
    assert overnight["banter_style_focus"] == "gentle_tease"
    assert overnight["lexical_variation_carry_bias"] == 0.11
    assert result.metrics["inner_os/overnight_expressive_style_history_active"] == 1.0
    assert result.metrics["inner_os/overnight_lexical_variation_carry_active"] == 1.0


def test_process_turn_restores_temperament_trace_into_persona_meta(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_temperament.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_temperament_trace"] = {
        "temperament_forward_trace": 0.44,
        "temperament_guard_trace": 0.18,
        "temperament_bond_trace": 0.22,
        "temperament_recovery_trace": 0.12,
    }
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    assert result.persona_meta["inner_os"]["temperament_trace"]["temperament_forward_trace"] >= 0.0
    assert result.metrics["inner_os/temperament_forward_trace"] >= 0.0
    assert harness._last_gate_context["inner_os_temperament_trace"]["temperament_guard_trace"] >= 0.0


def test_process_turn_emits_distillation_record_with_model_source(tmp_path: Path) -> None:
    @dataclass
    class _DistillHarness(_ProcessTurnHarness):
        def step(
            self,
            *,
            user_text: Optional[str] = None,
            context: Optional[str] = None,
            intent: Optional[str] = None,
            fast_only: bool = False,
            image_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            self._last_step_context = context
            response = SimpleNamespace(
                text="I will stay with what is visible first.",
                model="qwen-3.5-instruct",
                model_source="live_list",
                trace_id="hub-123",
                latency_ms=12.0,
                safety={"rating": "G"},
                controls_used={"mode": "watch"},
                retrieval_summary={"hits": [{"id": "ctx-1"}]},
                perception_summary={"text": "soft evening slope and signboard"},
                confidence=0.74,
                uncertainty_reason=(),
            )
            return {
                "talk_mode": "watch",
                "response_route": "watch",
                "metrics": {},
                "persona_meta": {},
                "heart": {},
                "shadow": None,
                "qualia_gate": {},
                "affect": None,
                "response": response,
                "memory_reference": None,
            }

    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _DistillHarness(_integration_hooks=hooks)
    harness._distillation_log_path = tmp_path / "inner_os_distillation.jsonl"
    harness._distillation_log_include_text = False

    _ = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    rows = [
        json.loads(line)
        for line in harness._distillation_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    record = rows[0]
    assert record["model"]["name"] == "qwen-3.5-instruct"
    assert record["model"]["source"] == "live_list"
    assert record["text_payload"] == {}
    assert record["input_fingerprint"]["user_text_sha256"]
    assert record["decision_snapshot"]["commitment_state"]["state"] in {"waver", "settle", "commit"}


def test_process_turn_exposes_transfer_package_and_runtime_seed(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    package = result.persona_meta["inner_os"]["transfer_package"]
    assert package["schema"] == "inner_os_transfer_package/v1"
    assert package["portable_state"]["same_turn"]["commitment_state"]["state"] in {"waver", "settle", "commit"}
    assert "runtime_seed" in package

    restored_seed = harness.build_inner_os_state_seed_from_transfer_package(package)
    assert "commitment_state_focus" in restored_seed
    assert "body_homeostasis_focus" in restored_seed
    assert "homeostasis_budget_focus" in restored_seed
    assert "relational_continuity_focus" in restored_seed
    assert "person_registry_snapshot" in restored_seed
    assert "temperament_forward_trace" in restored_seed
    assert result.metrics["inner_os/transfer_package_ready"] == 1.0


def test_process_turn_writes_transfer_package_snapshot_when_path_is_set(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._transfer_package_path = tmp_path / "inner_os_transfer_package.json"

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    payload = json.loads(harness._transfer_package_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inner_os_transfer_package/v1"
    assert payload["portable_state"]["same_turn"]["initiative_readiness"]["state"] in {"hold", "tentative", "ready"}
    assert result.metrics["inner_os/transfer_package_written"] == 1.0


def test_process_turn_writes_dashboard_snapshot_when_path_is_set(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._dashboard_snapshot_path = tmp_path / "inner_os_dashboard_snapshot.json"

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    payload = json.loads(harness._dashboard_snapshot_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inner_os_dashboard_snapshot/v1"
    assert payload["same_turn"]["social_topology_state"] in {
        "ambient",
        "one_to_one",
        "threaded_group",
        "public_visible",
        "hierarchical",
        "",
    }
    assert "temporal_membrane_mode" in payload["same_turn"]
    assert "temporal_membrane_focus" in payload["overnight"]
    assert "temporal_membrane_focus" in payload["transfer"]
    assert "temporal_alignment" in payload
    assert "focus_alignment" in payload["temporal_alignment"]
    assert "same_to_overnight_reentry_delta" in payload["temporal_alignment"]
    assert "contact_alignment" in payload
    assert "style" in payload["contact_alignment"]
    assert "dominant_carry_channel" in payload
    assert result.metrics["inner_os/dashboard_snapshot_ready"] == 1.0
    assert result.metrics["inner_os/dashboard_snapshot_written"] == 1.0


def test_apply_transfer_package_restores_working_memory_seed_and_gate_context() -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    package = {
        "runtime_seed": {
            "prev_qualia": [0.3],
            "commitment_state_focus": "commit",
            "temporal_membrane_focus": "coherent_reentry",
            "temporal_timeline_bias": 0.24,
            "temporal_reentry_bias": 0.19,
            "temporal_supersession_bias": 0.05,
            "temporal_continuity_bias": 0.17,
            "temporal_relation_reentry_bias": 0.14,
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.38,
            "body_homeostasis_focus": "recovering",
            "body_homeostasis_carry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.09,
            "relational_continuity_focus": "reopening",
            "relational_continuity_carry_bias": 0.14,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
            "temperament_forward_trace": 0.29,
        },
        "portable_state": {
            "carry": {
                "monument_carry": {
                    "memory_anchor": "harbor slope",
                    "semantic_seed_focus": "harbor slope",
                    "semantic_seed_anchor": "harbor slope",
                    "semantic_seed_strength": 0.73,
                    "semantic_seed_recurrence": 1.18,
                    "long_term_theme_focus": "quiet harbor routine",
                    "long_term_theme_anchor": "harbor slope",
                    "long_term_theme_strength": 0.61,
                    "long_term_theme_kind": "place",
                    "long_term_theme_summary": "quiet harbor routine",
                    "relation_seed_summary": "shared harbor thread",
                    "relation_seed_strength": 0.55,
                },
                "relationship_summary": {
                    "related_person_id": "person:harbor",
                    "attachment": 0.61,
                    "familiarity": 0.57,
                    "trust_memory": 0.49,
                    "partner_address_hint": "gentle",
                    "partner_timing_hint": "slow",
                    "partner_stance_hint": "careful",
                    "partner_social_interpretation": "future_open",
                },
            }
        },
    }

    restored = harness.apply_inner_os_transfer_package(package)

    assert restored["commitment_state_focus"] == "commit"
    assert harness._last_gate_context["inner_os_commitment_state_focus"] == "commit"
    assert harness._last_gate_context["inner_os_temporal_membrane_focus"] == "coherent_reentry"
    assert harness._last_gate_context["inner_os_autobiographical_thread_mode"] == "unfinished_thread"
    assert harness._last_gate_context["inner_os_autobiographical_thread_anchor"] == "harbor promise"
    assert harness._last_gate_context["inner_os_autobiographical_thread_strength"] == 0.38
    assert harness._last_gate_context["inner_os_temporal_timeline_bias"] == 0.24
    assert harness._last_gate_context["inner_os_temporal_reentry_bias"] == 0.19
    assert harness._last_gate_context["inner_os_body_homeostasis_focus"] == "recovering"
    assert harness._last_gate_context["inner_os_homeostasis_budget_focus"] == "recovering"
    assert harness._last_gate_context["inner_os_relational_continuity_focus"] == "reopening"
    assert harness._last_gate_context["inner_os_person_registry_snapshot"]["total_people"] == 2
    assert harness._last_gate_context["inner_os_group_thread_registry_snapshot"]["total_threads"] == 1
    assert harness._last_gate_context["inner_os_discussion_thread_registry_snapshot"]["dominant_issue_state"] == "pausing_issue"
    assert harness._last_gate_context["inner_os_group_thread_focus"] == "threaded_group"
    seed = harness._surface_world_state["working_memory_seed"]
    assert seed["semantic_seed_focus"] == "harbor slope"
    assert seed["long_term_theme_summary"] == "quiet harbor routine"
    assert seed["relation_seed_summary"] == "shared harbor thread"
    assert seed["group_thread_id"] == "threaded_group:person:friend|person:harbor"


def test_load_inner_os_transfer_package_returns_warm_start_summary() -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    summary = harness.load_inner_os_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
                "semantic_seed_strength": 0.64,
            },
            "initiative_followup_state": "reopen_softly",
            "commitment_target_focus": "repair",
            "temporal_membrane_focus": "same_group_reentry",
            "temporal_reentry_bias": 0.18,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.07,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
        }
    )

    assert summary["schema"] == "inner_os_transfer_package/v1"
    assert summary["migration"]["applied"] is True
    assert summary["source_model"]["name"] == "qwen-legacy"
    assert summary["initiative_followup_state"] == "reopen_softly"
    assert summary["commitment_target_focus"] == "repair"
    assert summary["temporal_membrane_focus"] == "same_group_reentry"
    assert summary["temporal_reentry_bias"] == 0.18
    assert summary["homeostasis_budget_focus"] == "recovering"
    assert summary["person_registry_total_people"] == 2
    assert summary["group_thread_total_threads"] == 1
    assert summary["discussion_thread_total_threads"] == 1
    assert summary["discussion_thread_dominant_anchor"] == "repair anchor"
    assert summary["discussion_thread_dominant_issue_state"] == "pausing_issue"
    assert summary["group_thread_focus"] == "threaded_group"
    assert summary["semantic_seed_anchor"] == "harbor slope"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_focus"] == "harbor slope"


def test_load_transfer_package_from_disk_restores_warm_start_seed(tmp_path: Path) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._transfer_package_path = tmp_path / "inner_os_transfer_package.json"
    harness._transfer_package_path.write_text(
        json.dumps(
            {
                "source_model": "qwen-legacy",
                "working_memory_seed": {
                    "memory_anchor": "harbor slope",
                    "semantic_seed_focus": "harbor slope",
                    "semantic_seed_anchor": "harbor slope",
                    "semantic_seed_strength": 0.68,
                    "related_person_id": "person:harbor",
                },
                "initiative_followup_state": "offer_next_step",
                "temperament_forward_trace": 0.32,
                "daily_carry_summary": {
                    "overnight_focus": {
                        "commitment_target_focus": "repair",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = harness._load_inner_os_transfer_package_from_disk()

    assert loaded is True
    assert harness._last_gate_context["inner_os_initiative_followup_state"] == "offer_next_step"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_anchor"] == "harbor slope"
    assert harness._last_gate_context["inner_os_transfer_package_migration"]["applied"] is True
    persisted = json.loads(harness._transfer_package_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "inner_os_transfer_package/v1"
    assert persisted["portable_state"]["carry"]["relationship_summary"]["related_person_id"] == "person:harbor"


def test_load_inner_os_transfer_package_from_path_returns_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    package_path = tmp_path / "transfer_package.json"
    package_path.write_text(
        json.dumps(
            {
                "schema": "inner_os_transfer_package/v1",
                "package_version": "v1",
                "source_model": {"name": "qwen-3.5-instruct", "source": "live_list"},
                "portable_state": {
                    "carry": {
                        "monument_carry": {
                            "semantic_seed_focus": "harbor slope",
                            "semantic_seed_anchor": "harbor slope",
                        }
                    }
                },
                "runtime_seed": {
                    "initiative_followup_state": "offer_next_step",
                    "commitment_target_focus": "step_forward",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = harness.load_inner_os_transfer_package(package_path)

    assert summary["migration"]["applied"] is False
    assert summary["source_model"]["name"] == "qwen-3.5-instruct"
    assert summary["initiative_followup_state"] == "offer_next_step"
    assert summary["commitment_target_focus"] == "step_forward"
    assert summary["semantic_seed_focus"] == "harbor slope"


def test_build_inner_os_model_swap_bundle_prefers_target_model_in_cache(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )
    bundle = harness.build_inner_os_model_swap_bundle(
        target_model="qwen-3.5-coder",
        result=result,
    )

    assert bundle["schema"] == "inner_os_model_swap_bundle/v1"
    assert bundle["target_model"] == "qwen-3.5-coder"
    assert bundle["selection_mode"] == "cache_preferred"
    assert bundle["transfer_package"]["schema"] == "inner_os_transfer_package/v1"
    assert terrain_llm.get_cached_selected_model("http://localhost:1234/v1") == "qwen-3.5-coder"


def test_warm_start_from_transfer_package_applies_target_model_preference(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    summary = harness.warm_start_from_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
            },
            "initiative_followup_state": "offer_next_step",
            "agenda_focus": "step_forward",
            "agenda_bias": 0.14,
        },
        target_model="qwen-3.5-coder",
    )

    assert summary["target_model"] == "qwen-3.5-coder"
    assert summary["selected_model_after_warm_start"] == "qwen-3.5-coder"
    assert harness._last_gate_context["inner_os_transfer_target_model"] == "qwen-3.5-coder"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_anchor"] == "harbor slope"


def test_process_turn_exposes_transfer_summary_after_warm_start(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    harness.warm_start_from_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
            },
            "initiative_followup_state": "offer_next_step",
            "commitment_target_focus": "repair",
            "agenda_focus": "repair",
            "agenda_bias": 0.18,
            "temporal_membrane_focus": "same_group_reentry",
            "temporal_timeline_bias": 0.22,
            "temporal_reentry_bias": 0.2,
            "temporal_supersession_bias": 0.04,
            "temporal_continuity_bias": 0.18,
            "temporal_relation_reentry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.08,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
        },
        target_model="qwen-3.5-coder",
    )

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    summary = result.persona_meta["inner_os"]["transfer_summary"]
    assert summary["migration"]["applied"] is True
    assert summary["semantic_seed_anchor"] == "harbor slope"
    assert summary["commitment_target_focus"] == "repair"
    assert summary["agenda_focus"] == "repair"
    assert summary["agenda_bias"] == 0.18
    assert summary["temporal_membrane_focus"] == "same_group_reentry"
    assert summary["temporal_reentry_bias"] == 0.2
    assert summary["homeostasis_budget_focus"] == "recovering"
    assert summary["person_registry_total_people"] == 2
    assert summary["group_thread_total_threads"] == 1
    assert summary["discussion_thread_total_threads"] == 1
    assert summary["discussion_thread_dominant_anchor"] == "repair anchor"
    assert summary["discussion_thread_dominant_issue_state"] == "pausing_issue"
    assert summary["group_thread_focus"] == "threaded_group"
    assert summary["target_model"] == "qwen-3.5-coder"
    assert result.metrics["inner_os/transfer_migration_active"] == 1.0
    assert result.metrics["inner_os/transfer_migration_from_legacy"] == 1.0
    assert result.metrics["inner_os/transfer_semantic_seed_visible"] == 1.0
    assert result.metrics["inner_os/transfer_target_model_requested"] == 1.0
    assert result.metrics["inner_os/person_registry_total_people"] == 2.0
    assert result.metrics["inner_os/group_thread_total_threads"] == 1.0
    assert result.metrics["inner_os/discussion_thread_total_threads"] == 1.0
    assert result.persona_meta["inner_os"]["group_thread_registry_summary"]["total_threads"] == 1
    assert result.persona_meta["inner_os"]["discussion_thread_registry_summary"]["total_threads"] >= 1
    assert result.persona_meta["inner_os"]["group_thread_focus"] == "threaded_group"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["group_thread_focus"] == "threaded_group"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["temporal_membrane_focus"] == "same_group_reentry"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["temporal_reentry_bias"] == 0.2


def test_process_turn_passes_working_memory_seed_into_inner_os_pre_turn(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._surface_world_state["working_memory_seed"] = {
        "semantic_seed_focus": "harbor slope",
        "semantic_seed_anchor": "harbor slope",
        "semantic_seed_strength": 0.78,
        "semantic_seed_recurrence": 1.22,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0


def test_process_turn_derives_working_memory_seed_from_eqnet_system(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness.eqnet_system = SimpleNamespace(
        _working_memory_signature_summary=lambda: {
            "dominant_focus": "harbor slope",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.56,
            "recurrence_weight": 1.24,
            "long_term_theme": {
                "focus": "harbor slope",
                "anchor": "harbor slope",
                "kind": "place",
                "strength": 0.61,
                "summary": "quiet harbor slope memory",
            },
        },
        _latest_nightly_working_memory_replay_summary=lambda: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.62,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/semantic_seed_recurrence"] > 0.0
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0
    assert result.persona_meta["inner_os"]["semantic_seed_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.61
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_kind"] == "place"


def test_build_context_payload_includes_working_memory_seed_tags() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._talk_mode = SimpleNamespace(value="watch")
    harness._last_gate_context["inner_os_working_memory_seed"] = {
        "semantic_seed_focus": "harbor slope",
        "semantic_seed_anchor": "harbor slope",
        "long_term_theme_focus": "harbor slope",
        "long_term_theme_anchor": "harbor slope",
        "long_term_theme_kind": "place",
        "long_term_theme_summary": "quiet harbor slope memory",
    }
    payload = harness._build_context_payload(
        user_text="remember this place",
        context_text="previous framing",
        metrics={},
        gate_ctx=SimpleNamespace(force_listen=False, text_input=True),
        route=SimpleNamespace(value="conscious"),
    )
    assert "wm_seed_focus:harbor_slope" in payload["context_tags"]
    assert "wm_seed_anchor:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_focus:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_anchor:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_kind:place" in payload["context_tags"]
    assert "ltm_theme_summary:quiet_harbor_slope_memory" in payload["context_tags"]


def test_process_turn_uses_conscious_mosaic_working_memory_seed_for_recall(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.42,
        },
    )
    result = harness.process_turn(user_text="harbor", context="previous framing")
    assert result.metrics["inner_os/conscious_seed_strength"] == 0.42


def test_process_turn_uses_conscious_mosaic_long_term_theme_as_residue_only(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {},
        latest_long_term_theme=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "kind": "place",
            "summary": "quiet harbor slope memory",
            "strength": 0.46,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.0
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] in {"", None}
    assert result.persona_meta["inner_os"]["current_focus"] == "place"
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0


def test_process_turn_corroborates_conscious_long_term_theme_with_nightly_theme(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {},
        latest_long_term_theme=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "kind": "place",
            "summary": "quiet harbor slope memory",
            "strength": 0.46,
        },
    )
    harness._latest_nightly_long_term_theme_summary = lambda: {
        "focus": "harbor slope",
        "anchor": "harbor slope",
        "kind": "place",
        "summary": "quiet harbor slope memory",
        "strength": 0.6,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.6
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_summary"] == "quiet harbor slope memory"


def test_process_turn_derives_pre_turn_seed_from_conscious_mosaic(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.44,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] == 0.44
    assert result.persona_meta["inner_os"]["semantic_seed_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}


def test_process_turn_derives_seed_from_nightly_long_term_theme(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_long_term_theme_summary = lambda: {
        "focus": "harbor slope",
        "anchor": "harbor slope",
        "kind": "place",
        "summary": "quiet harbor slope memory",
        "strength": 0.6,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.6
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_summary"] == "quiet harbor slope memory"


def test_process_turn_derives_identity_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_identity_arc_summary = lambda: {
        "arc_kind": "repairing_bond",
        "phase": "shifting",
        "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
        "open_tension": "timing_sensitive_reentry",
        "stability": 0.58,
        "memory_anchor": "harbor slope",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/identity_arc_visible"] == 1.0
    assert result.metrics["inner_os/identity_arc_stability"] == 0.58
    assert result.persona_meta["inner_os"]["identity_arc_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["identity_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["identity_arc_summary"].startswith("repair is gathering")
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["identity_arc_kind"] == "repairing_bond"


def test_process_turn_exposes_identity_arc_registry_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_identity_arc_registry_summary = lambda: {
        "dominant_arc_id": "repairing_bond::person:user::meaning::harbor",
        "dominant_arc_kind": "repairing_bond",
        "dominant_arc_phase": "shifting",
        "dominant_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
        "active_arc_count": 1,
        "total_arcs": 1,
        "top_arc_ids": ["repairing_bond::person:user::meaning::harbor"],
        "status_counts": {"active": 1},
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["identity_arc_registry_summary"]["dominant_arc_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["identity_arc_registry_dominant_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["identity_arc_registry_active_count"] == 1


def test_process_turn_derives_relation_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_relation_arc_summary = lambda: {
        "arc_kind": "repairing_relation",
        "phase": "shifting",
        "summary": "repair is gathering around a companion thread",
        "open_tension": "timing_sensitive_reentry",
        "stability": 0.56,
        "related_person_id": "person:harbor",
        "group_thread_id": "threaded_group:person:friend|person:harbor",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["relation_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["relation_arc_summary"] == "repair is gathering around a companion thread"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["relation_arc_summary"] == "repair is gathering around a companion thread"


def test_process_turn_exposes_relation_arc_registry_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_relation_arc_registry_summary = lambda: {
        "dominant_arc_id": "repairing_relation::person:harbor::companion",
        "dominant_arc_kind": "repairing_relation",
        "dominant_arc_phase": "shifting",
        "dominant_arc_summary": "repair is gathering around a companion thread",
        "dominant_person_id": "person:harbor",
        "dominant_group_thread_id": "threaded_group:person:friend|person:harbor",
        "active_arc_count": 1,
        "total_arcs": 1,
        "top_arc_ids": ["repairing_relation::person:harbor::companion"],
        "status_counts": {"active": 1},
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["relation_arc_registry_summary"]["dominant_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["relation_arc_registry_dominant_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["relation_arc_registry_active_count"] == 1


def test_process_turn_derives_group_relation_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_group_relation_arc_summary = lambda: {
        "arc_kind": "repairing_relation",
        "phase": "shifting",
        "summary": "repair is moving through a shared group thread in small steps",
        "group_thread_id": "threaded_group:person:friend|person:harbor",
        "topology_focus": "threaded_group",
        "boundary_mode": "same_group_reentry",
        "reentry_window_focus": "next_same_group_window",
        "dominant_person_id": "person:harbor",
        "stability": 0.61,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["group_relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["group_relation_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["group_relation_arc_boundary_mode"] == "same_group_reentry"
    assert result.persona_meta["inner_os"]["group_relation_arc_group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert result.persona_meta["inner_os"]["group_relation_arc"]["summary"] == "repair is moving through a shared group thread in small steps"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["group_relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["group_relation_boundary_mode"] == "same_group_reentry"


def test_process_turn_derives_relation_seed_from_nightly_partner_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_partner_relation_summary = lambda: {
        "person_id": "user",
        "summary": "gentle harbor companion thread",
        "memory_anchor": "harbor slope",
        "social_role": "companion",
        "attachment": 0.74,
        "familiarity": 0.69,
        "trust_memory": 0.71,
        "strength": 0.72,
        "address_hint": "companion",
        "timing_hint": "open",
        "stance_hint": "familiar",
        "social_interpretation": "familiar:companion:open",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["related_person_id"] == "user"
    assert result.persona_meta["inner_os"]["attachment"] >= 0.6
    assert result.persona_meta["inner_os"]["familiarity"] >= 0.58
    assert result.persona_meta["inner_os"]["trust_memory"] >= 0.6
    assert result.persona_meta["inner_os"]["partner_address_hint"] == "companion"
    assert result.persona_meta["inner_os"]["partner_timing_hint"] == "open"
    assert result.persona_meta["inner_os"]["partner_stance_hint"] == "familiar"


def test_visual_reflection_uses_degraded_surface_text(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _DegradedHarness(_integration_hooks=hooks)
    normalized = harness._normalize_perception_summary({"error": "vision_error", "detail": "500 Server Error"})
    response = SimpleNamespace(text="I am keeping the reading tentative.", perception_summary=None)
    response = harness._attach_visual_reflection(response, normalized)
    assert response is not None
    assert normalized is not None
    assert normalized["status"] == "degraded"
    assert "visual read did not settle" in response.text
