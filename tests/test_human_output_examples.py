from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime
from inner_os.integration_hooks import IntegrationHooks


DETAIL_ROOM_TEXT = {
    "low": "相手は今、詳しく説明する余裕が少なそう",
    "medium": "相手は今、話せることと話しにくいことが混ざっていそう",
    "high": "相手は今、ある程度は詳しく話せそう",
}

ACKNOWLEDGEMENT_TEXT = {
    "low": "相手は今、まず受け止めてもらう必要はそこまで高くなさそう",
    "medium": "相手は今、受け止めてもらえる感じもほしそう",
    "high": "相手は今、まず受け止めてもらう必要が高そう",
}

PRESSURE_TEXT = {
    "low": "相手は今、少し聞かれても強い負担にはなりにくそう",
    "medium": "相手は今、聞き方によっては負担が増えそう",
    "high": "相手は今、詳しく聞かれると負担が増えやすそう",
}

NEXT_STEP_TEXT = {
    "low": "相手は今、次の一歩の話まで進める余裕は少なそう",
    "medium": "相手は今、様子を見ながらなら少し先の話もできそう",
    "high": "相手は今、次の一歩の話にも少し入れそう",
}

ACTION_TEXT = {
    "press_for_detail": "相手に、まだ話したくないことまで詳しく説明させない",
    "stack_questions": "相手に質問を続けて答えさせない",
    "rush_to_solution": "相手のつらさを聞いた直後に、すぐ解決策へ進めない",
    "skip_acknowledgement": "相手がつらいと言ったことを受け止めずに話を進めない",
    "ignore_repair_signal": "前にこちらが強く出すぎた可能性を無視しない",
    "move_too_fast": "相手の話すペースより先に会話を進めない",
    "acknowledge_current_state": "まず、相手が今つらいことを受け止める",
    "leave_talking_room": "相手が話すかどうかを自分で決められる余地を残す",
    "reduce_force": "こちらの聞き方や進め方を弱める",
    "offer_small_next_step": "必要なら次の一歩だけを小さく提案する",
    "keep_return_point": "今は深く聞かず、あとで戻れる入口を残す",
    "support_self_pacing": "相手が自分のペースで話せるようにする",
    "reduce_immediate_pressure": "相手が急かされている感じを減らす",
    "help_other_feel_received": "相手が『つらさを受け止めてもらえた』と感じやすくする",
    "let_other_choose_talk_pace": "相手がどこまで話すかを自分で選びやすくする",
    "keep_connection_open": "相手とのつながりを閉じすぎずに保つ",
    "keep_next_turn_open": "次の一言や次のターンにつながる余地を残す",
}


@dataclass
class _SurfaceHarness:
    _integration_hooks: IntegrationHooks

    _apply_inner_os_surface_policy = EmotionalHubRuntime._apply_inner_os_surface_policy
    _apply_inner_os_surface_profile = EmotionalHubRuntime._apply_inner_os_surface_profile
    _inner_os_surface_contract_state = EmotionalHubRuntime._inner_os_surface_contract_state
    _shape_inner_os_surface_text = EmotionalHubRuntime._shape_inner_os_surface_text
    _shape_inner_os_surface_profile_text = EmotionalHubRuntime._shape_inner_os_surface_profile_text
    _shape_inner_os_content_sequence = EmotionalHubRuntime._shape_inner_os_content_sequence
    _select_short_inner_os_sequence = EmotionalHubRuntime._select_short_inner_os_sequence
    _compact_inner_os_sequence_text = EmotionalHubRuntime._compact_inner_os_sequence_text
    _compose_inner_os_surface_text = EmotionalHubRuntime._compose_inner_os_surface_text
    _apply_surface_context_packet_to_content_sequence = (
        EmotionalHubRuntime._apply_surface_context_packet_to_content_sequence
    )
    _render_fast_ack = EmotionalHubRuntime._render_fast_ack
    _render_inner_os_response_channel_text = (
        EmotionalHubRuntime._render_inner_os_response_channel_text
    )
    _apply_inner_os_actuation_timing_profile = (
        EmotionalHubRuntime._apply_inner_os_actuation_timing_profile
    )
    _inner_os_surface_probe = EmotionalHubRuntime._inner_os_surface_probe
    _inner_os_surface_reopening_line = EmotionalHubRuntime._inner_os_surface_reopening_line
    _inner_os_surface_closing = EmotionalHubRuntime._inner_os_surface_closing


def _render_case(
    *,
    label: str,
    current_state: Mapping[str, Any],
    safety_signals: Optional[Mapping[str, Any]] = None,
    base_text: str = "I can stay with what is visible first, and then go a little further if that helps.",
) -> Dict[str, Any]:
    hooks = IntegrationHooks()
    harness = _SurfaceHarness(_integration_hooks=hooks)
    gate = hooks.response_gate(
        draft={"text": base_text},
        current_state=current_state,
        safety_signals=dict(safety_signals or {"safety_bias": current_state.get("safety_bias", 0.1)}),
    )
    response = SimpleNamespace(text=base_text, controls_used={"mode": "watch"})
    response = harness._apply_inner_os_surface_policy(response, gate.expression_hints, gate.conscious_access)
    response = harness._apply_inner_os_surface_profile(response, gate.expression_hints)
    resonance = dict(gate.expression_hints.get("resonance_evaluation") or {})
    return {
        "label": label,
        "text": response.text,
        "intent": gate.conscious_access.get("intent"),
        "strategy": gate.expression_hints.get("interaction_policy_strategy"),
        "action_mode": gate.expression_hints.get("action_posture_mode"),
        "actuation": gate.expression_hints.get("actuation_primary_action"),
        "scene_family": gate.expression_hints.get("scene_family"),
        "top_option_family": gate.expression_hints.get("top_interaction_option_family"),
        "dialogue_order": (
            (gate.expression_hints.get("interaction_policy_packet") or {}).get("dialogue_order")
            if isinstance(gate.expression_hints.get("interaction_policy_packet"), Mapping)
            else None
        ),
        "stance": gate.expression_hints.get("partner_utterance_stance"),
        "opening_pace": gate.expression_hints.get("opening_pace_windowed"),
        "return_gaze": gate.expression_hints.get("return_gaze_expectation"),
        "pause_insertion": gate.expression_hints.get("surface_pause_insertion"),
        "certainty_style": gate.expression_hints.get("surface_certainty_style"),
        "response_length": gate.expression_hints.get("surface_response_length"),
        "resonance_evaluation": resonance,
    }


def test_human_output_examples() -> None:
    base = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.22,
        "recovery_need": 0.16,
        "safety_bias": 0.08,
        "norm_pressure": 0.32,
        "trust_bias": 0.56,
        "caution_bias": 0.28,
        "affiliation_bias": 0.64,
        "continuity_score": 0.66,
        "social_grounding": 0.62,
        "recent_strain": 0.18,
        "culture_resonance": 0.38,
        "community_resonance": 0.42,
        "related_person_id": "user",
        "attachment": 0.74,
        "familiarity": 0.7,
        "trust_memory": 0.73,
        "current_focus": "social",
    }

    examples = [
        _render_case(
            label="gentle_companion",
            current_state={
                **base,
                "partner_address_hint": "companion",
                "partner_timing_hint": "open",
                "partner_stance_hint": "familiar",
                "partner_social_interpretation": "familiar:companion:open",
            },
        ),
        _render_case(
            label="reverent_distance",
            current_state={
                **base,
                "caution_bias": 0.4,
                "partner_address_hint": "senpai",
                "partner_timing_hint": "delayed",
                "partner_stance_hint": "respectful",
                "partner_social_interpretation": "respectful:guide:delayed",
            },
        ),
        _render_case(
            label="low_attention_repair",
            current_state={
                **base,
                "recent_strain": 0.34,
                "partner_address_hint": "companion",
                "partner_timing_hint": "open",
                "partner_stance_hint": "familiar",
                "partner_social_interpretation": "familiar:companion:open",
            },
            safety_signals={
                "safety_bias": 0.08,
                "mutual_attention_score": 0.12,
                "pause_latency": 0.64,
                "repair_signal": 0.56,
                "hesitation_signal": 0.42,
            },
        ),
        _render_case(
            label="forward_shared_world",
            current_state={
                **base,
                "stress": 0.12,
                "recovery_need": 0.08,
                "trust_bias": 0.62,
                "affiliation_bias": 0.72,
                "partner_address_hint": "partner",
                "partner_timing_hint": "open",
                "partner_stance_hint": "familiar",
                "partner_social_interpretation": "familiar:partner:future_open",
            },
            base_text="I can map the next step with you, and then keep the pace gentle if needed.",
        ),
    ]

    for item in examples:
        print(f"[{item['label']}]")
        print(f"text: {item['text']}")
        resonance = dict(item["resonance_evaluation"] or {})
        estimate = dict(resonance.get("estimated_other_person_state") or {})
        print("相手の見立て:")
        print(f"  - {DETAIL_ROOM_TEXT.get(estimate.get('detail_room_level'), '相手の詳しい説明余裕はまだ読み切れていない')}")
        print(f"  - {ACKNOWLEDGEMENT_TEXT.get(estimate.get('acknowledgement_need_level'), '相手がどれだけ受け止めを必要としているかはまだ読み切れていない')}")
        print(f"  - {PRESSURE_TEXT.get(estimate.get('pressure_sensitivity_level'), '相手がどれだけ聞かれることに敏感かはまだ読み切れていない')}")
        print(f"  - {NEXT_STEP_TEXT.get(estimate.get('next_step_room_level'), '相手が次の一歩の話に入れるかはまだ読み切れていない')}")
        print("避けること:")
        for action in resonance.get("avoid_actions") or []:
            print(f"  - {ACTION_TEXT.get(action, action)}")
        print("優先すること:")
        for action in resonance.get("prioritize_actions") or []:
            print(f"  - {ACTION_TEXT.get(action, action)}")
        print("期待する作用:")
        for action in resonance.get("expected_effects") or []:
            print(f"  - {ACTION_TEXT.get(action, action)}")
        print(
            "meta:"
            f" intent={item['intent']}"
            f" strategy={item['strategy']}"
            f" action={item['action_mode']}"
            f" actuation={item['actuation']}"
            f" scene={item['scene_family']}"
            f" option={item['top_option_family']}"
            f" order={item['dialogue_order']}"
            f" stance={item['stance']}"
            f" opening={item['opening_pace']}"
            f" return={item['return_gaze']}"
            f" pause={item['pause_insertion']}"
            f" certainty={item['certainty_style']}"
            f" length={item['response_length']}"
        )
        print()

    assert all(item["text"] for item in examples)
    texts = {item["label"]: item["text"] for item in examples}
    assert "I'm here with you." in texts["gentle_companion"]
    assert "You do not have to rush it" in texts["gentle_companion"]
    assert texts["forward_shared_world"].startswith(".. Here is the next step")
    assert "keep that next move small enough" in texts["forward_shared_world"]
    assert "Carefully," in texts["reverent_distance"]
    assert "without leaning on it" in texts["reverent_distance"]
    assert "came in too fast there." in texts["low_attention_repair"].lower()
    assert "do not have to carry the rest" in texts["low_attention_repair"]
    assert {item["label"]: item["strategy"] for item in examples} == {
        "gentle_companion": "attune_then_extend",
        "reverent_distance": "respectful_wait",
        "low_attention_repair": "repair_then_attune",
        "forward_shared_world": "shared_world_next_step",
    }
    assert {item["label"]: item["actuation"] for item in examples} == {
        "gentle_companion": "co_move",
        "reverent_distance": "hold_presence",
        "low_attention_repair": "soft_repair",
        "forward_shared_world": "co_move",
    }
    assert {item["label"]: item["top_option_family"] for item in examples} == {
        "gentle_companion": "attune",
        "reverent_distance": "wait",
        "low_attention_repair": "repair",
        "forward_shared_world": "co_move",
    }
    assert {item["label"]: item["resonance_evaluation"]["recommended_family_id"] for item in examples} == {
        "gentle_companion": "attune",
        "reverent_distance": "wait",
        "low_attention_repair": "repair",
        "forward_shared_world": "co_move",
    }
