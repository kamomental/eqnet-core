# -*- coding: utf-8 -*-

from inner_os.expression.surface_context_packet import build_surface_context_packet
from inner_os.expression.turn_delta import derive_turn_delta
from inner_os.recent_dialogue_state import derive_recent_dialogue_state
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_surface_context_packet_exposes_brightness_playfulness_and_tempo() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "continuing_thread"},
        turn_delta={"kind": "continuity_thread", "preferred_act": "keep_shared_thread_visible"},
        interaction_constraints={"avoid_obvious_advice": True},
        surface_profile={
            "response_length": "short",
            "cultural_register": "casual_shared",
            "group_register": "one_to_one",
            "sentence_temperature": "warm",
            "voice_texture": "light_playful",
        },
        live_engagement_state={
            "state": "pickup_comment",
            "score": 0.64,
            "primary_move": "pick_up_comment",
        },
        lightness_budget_state={
            "state": "open_play",
            "banter_room": 0.28,
            "playful_ceiling": 0.32,
        },
    ).to_dict()

    assert packet["surface_profile"]["brightness"] == "pickup_comment"
    assert packet["surface_profile"]["playfulness"] == 0.28
    assert packet["surface_profile"]["tempo"] == 0.64
    assert packet["surface_profile"]["voice_texture"] == "light_playful"
    assert packet["source_state"]["voice_texture"] == "light_playful"
    assert packet["source_state"]["lightness_budget_state"] == "open_play"
    assert packet["source_state"]["live_primary_move"] == "pick_up_comment"


def test_turn_delta_prefers_bright_continuity_when_lightness_room_is_open() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.52,
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.18,
                    "affective_charge": 0.22,
                    "reopening_pull": 0.14,
                }
            },
            "live_engagement_state": {
                "state": "pickup_comment",
                "score": 0.64,
                "primary_move": "pick_up_comment",
            },
            "lightness_budget_state": {
                "state": "open_play",
                "banter_room": 0.28,
                "playful_ceiling": 0.34,
            },
        },
        interaction_constraints={},
    )

    assert delta.kind == "bright_continuity"
    assert delta.preferred_act == "light_bounce"


def test_turn_delta_uses_voice_texture_as_bright_continuity_fallback() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.52,
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.18,
                    "affective_charge": 0.18,
                    "reopening_pull": 0.14,
                }
            },
            "surface_profile": {
                "voice_texture": "light_playful",
            },
        },
        interaction_constraints={},
    )

    assert delta.kind == "bright_continuity"
    assert delta.preferred_act == "shared_delight"


def test_recent_dialogue_state_treats_marked_followup_with_anchor_as_continuation() -> None:
    state = derive_recent_dialogue_state(
        "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        ["今日はちょっといいことがあって、帰り道が少し軽かったんです。"],
    )

    assert state.state == "reopening_thread"
    assert state.recent_anchor


def test_turn_delta_can_stay_bright_when_constraints_open_small_next_step() -> None:
    delta = derive_turn_delta(
        {
            "surface_profile": {"voice_texture": "light_playful"},
            "lightness_budget_state": {
                "state": "open_play",
                "banter_room": 0.32,
                "playful_ceiling": 0.34,
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.96,
                    "affective_charge": 0.24,
                    "reopening_pull": 0.16,
                }
            },
        },
        interaction_constraints={
            "keep_thread_visible": True,
            "allow_small_next_step": True,
        },
    )

    assert delta.kind == "bright_continuity"
    assert delta.preferred_act == "shared_delight"


def test_turn_delta_prefers_bright_continuity_for_shared_smile_even_when_guardedness_is_high() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.68,
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.88,
                    "affective_charge": 0.24,
                    "reopening_pull": 0.18,
                }
            },
            "live_engagement_state": {
                "state": "riff_with_comment",
                "score": 0.52,
                "primary_move": "riff_current_comment",
            },
            "lightness_budget_state": {
                "state": "open_play",
                "banter_room": 0.36,
                "playful_ceiling": 0.34,
            },
            "shared_moment_state": {
                "state": "shared_moment",
                "moment_kind": "laugh",
                "score": 0.74,
                "jointness": 0.72,
                "afterglow": 0.64,
            },
            "utterance_reason_packet": {
                "state": "active",
                "offer": "brief_shared_smile",
                "question_policy": "none",
            },
            "organism_state": {
                "dominant_posture": "play",
                "play_window": 0.44,
                "expressive_readiness": 0.58,
            },
            "surface_profile": {
                "voice_texture": "light_playful",
            },
        },
        interaction_constraints={},
    )

    assert delta.kind == "bright_continuity"
    assert delta.preferred_act == "light_bounce"


def test_turn_delta_overrides_reopening_thread_for_shared_smile_reentry() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.42,
                "reopen_pressure": 0.22,
                "recent_anchor": "前の流れはまだしんどさが残っていた",
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.82,
                    "affective_charge": 0.22,
                    "reopening_pull": 0.16,
                }
            },
            "live_engagement_state": {
                "state": "riff_with_comment",
                "score": 0.56,
                "primary_move": "riff_current_comment",
            },
            "lightness_budget_state": {
                "state": "open_play",
                "banter_room": 0.36,
                "playful_ceiling": 0.34,
            },
            "shared_moment_state": {
                "state": "shared_moment",
                "moment_kind": "laugh",
                "score": 0.72,
                "jointness": 0.7,
                "afterglow": 0.62,
            },
            "utterance_reason_packet": {
                "state": "active",
                "offer": "brief_shared_smile",
                "question_policy": "none",
            },
            "organism_state": {
                "dominant_posture": "play",
                "play_window": 0.44,
                "expressive_readiness": 0.58,
            },
            "surface_profile": {
                "voice_texture": "light_playful",
            },
        },
        interaction_constraints={},
    )

    assert delta.kind == "bright_continuity"
    assert delta.preferred_act == "light_bounce"
    assert "recent_dialogue:bright_reentry_override" in delta.rationale


def test_runtime_packet_pruning_prefers_bright_continuity_over_quiet_presence() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    pruned = runtime._apply_surface_context_packet_to_content_sequence(
        [
            {"act": "quiet_presence", "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。"},
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
            {"act": "keep_shared_thread_visible", "text": "いまここにある流れを、切らさずに持っておきたいです。"},
        ],
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "response_role": {"primary": "light_bounce"},
            "constraints": {"max_questions": 0},
            "shared_core": {},
            "surface_profile": {"voice_texture": "light_playful"},
        },
        surface_profile={"response_length": "short"},
    )

    assert [item["act"] for item in pruned] == [
        "shared_delight",
        "light_bounce",
    ]
