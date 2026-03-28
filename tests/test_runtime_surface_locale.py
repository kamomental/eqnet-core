# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.llm_hub import HubResponse
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_surface_profile_text_uses_japanese_careful_prefix() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    shaped = runtime._shape_inner_os_surface_profile_text(
        "ここを無理に決めなくて大丈夫です。",
        surface_profile={
            "certainty_style": "careful",
            "response_length": "short",
            "sentence_temperature": "gentle",
            "pause_insertion": "none",
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "careful_return",
        },
    )

    assert "Carefully," not in shaped
    assert shaped.startswith("… いまは、")


def test_runtime_surface_profile_keeps_one_guarded_raw_line_when_no_opening_line_is_planned() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    response = HubResponse(
        text="ここを無理に押し進めなくて大丈夫です。",
        model=None,
        trace_id="probe",
        latency_ms=0.0,
        controls_used={
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": (
                "まずは、『ちょっとだけ引っかかっていることがあるんだ』とだけ言ってみるので十分です。\n"
                "具体的に話す前に、その一言だけでも共有すると話しやすくなります。\n"
                "- 例: 少し困っていることがある"
            ),
        },
        safety={},
    )

    shaped = runtime._apply_inner_os_surface_profile(
        response,
        {
            "interaction_policy_packet": {
                "response_strategy": "respectful_wait",
                "dialogue_act": "check_in",
                "opening_move": "acknowledge_without_probe",
                "followup_move": "keep_shared_thread_visible",
                "closing_move": "leave_unfinished_part_closed_for_now",
                "primary_object_operation": {"operation_kind": "hold_without_probe"},
            },
            "interaction_policy_dialogue_act": "check_in",
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "careful",
            "surface_banter_move": "none",
            "surface_lexical_variation_mode": "plain",
            "surface_group_register": "ambient",
        },
    )

    assert shaped is not None
    assert "まずは、『ちょっとだけ引っかかっていることがあるんだ』" in shaped.text
    assert "ここを無理に押し進めなくて大丈夫です。" in shaped.text
    assert "- 例:" not in shaped.text
    assert shaped.controls_used["inner_os_guarded_narrative_bridge_used"] is True


def test_runtime_guarded_narrative_bridge_skips_english_when_locale_is_japanese() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    bridge = runtime._extract_guarded_narrative_bridge_text(
        "I'm here if you'd like to clarify or share more about what's on your mind.",
        locale="ja-JP",
    )

    assert bridge == ""


def test_runtime_guarded_narrative_bridge_skips_generic_help_phrase() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    bridge = runtime._extract_guarded_narrative_bridge_text(
        "何かお手伝いできることがあれば教えてくださいね。",
        locale="ja-JP",
    )

    assert bridge == ""


def test_runtime_surface_prefers_planned_opening_line_over_guarded_bridge() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    response = HubResponse(
        text="ここを無理に押し進めなくて大丈夫です。",
        model=None,
        trace_id="probe",
        latency_ms=0.0,
        controls_used={
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": "まずは「今、何が一番つらいか」を聞くことから始めてみてはいかがでしょう。",
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "We do not have to press this right now."},
                {
                    "act": "offer_small_opening_line",
                    "text": "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                },
                {
                    "act": "quiet_presence",
                    "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
                },
            ],
        },
        safety={},
    )

    shaped = runtime._apply_inner_os_surface_profile(
        response,
        {
            "interaction_policy_packet": {
                "response_strategy": "contain_then_stabilize",
                "dialogue_act": "clarify",
            },
            "interaction_policy_dialogue_act": "clarify",
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "careful",
            "surface_banter_move": "none",
            "surface_lexical_variation_mode": "plain",
            "surface_group_register": "ambient",
        },
    )

    assert shaped is not None
    assert "切り出すなら" in shaped.text
    assert "何が一番つらいか" not in shaped.text
    assert shaped.controls_used["inner_os_guarded_narrative_bridge_used"] is False


def test_runtime_surface_can_use_planned_sequence_from_expression_hint() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    response = HubResponse(
        text="ここを無理に押し進めなくて大丈夫です。",
        model=None,
        trace_id="probe",
        latency_ms=0.0,
        controls_used={
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": "まずは「今、何が一番つらいか」を聞くことから始めてみてはいかがでしょう。",
        },
        safety={},
    )

    shaped = runtime._apply_inner_os_surface_profile(
        response,
        {
            "planned_content_sequence": [
                {"act": "respect_boundary", "text": "We do not have to press this right now."},
                {
                    "act": "offer_small_opening_line",
                    "text": "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                },
                {
                    "act": "quiet_presence",
                    "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
                },
            ],
            "interaction_policy_packet": {
                "response_strategy": "contain_then_stabilize",
                "dialogue_act": "clarify",
            },
            "interaction_policy_dialogue_act": "clarify",
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "careful",
            "surface_banter_move": "none",
            "surface_lexical_variation_mode": "plain",
            "surface_group_register": "ambient",
        },
    )

    assert shaped is not None
    assert "切り出すなら" in shaped.text
    assert "何が一番つらいか" not in shaped.text
