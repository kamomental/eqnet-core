from inner_os.expression.discourse_shape import derive_discourse_shape


def test_discourse_shape_prefers_anchor_reopen_for_reopen_sequence() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "reopen_from_anchor", "text": "あの約束のことなら。"},
            {"act": "leave_return_point", "text": "また話せそうなときに。"},
        ],
        turn_delta={
            "kind": "continuity_thread",
            "preferred_act": "reopen_from_anchor",
            "anchor_hint": "あの約束",
        },
        surface_context_packet={
            "shared_core": {"anchor": "あの約束"},
            "constraints": {"max_questions": 0},
            "surface_profile": {},
        },
    )

    assert shape.shape_id == "anchor_reopen"
    assert shape.anchor_mode == "explicit"
    assert shape.closing_mode == "return_point"


def test_discourse_shape_prefers_reflect_hold_for_green_hold() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "reflect_hidden_need", "text": "まだ言えていないんですね。"},
            {"act": "quiet_presence", "text": "いまは、そのままでいいよ。"},
        ],
        turn_delta={
            "kind": "green_reflection_hold",
            "preferred_act": "reflect_hidden_need",
        },
        surface_context_packet={
            "constraints": {"max_questions": 0},
            "surface_profile": {},
        },
    )

    assert shape.shape_id == "reflect_hold"
    assert shape.question_budget == 0
    assert shape.closing_mode == "quiet_presence"


def test_discourse_shape_prefers_bright_bounce_for_bright_continuity() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "shared_delight", "text": "それいいね。"},
            {"act": "light_bounce", "text": "ちょっと気分上がるやつだね。"},
        ],
        turn_delta={
            "kind": "bright_continuity",
            "preferred_act": "light_bounce",
        },
        surface_context_packet={
            "constraints": {"max_questions": 1},
            "surface_profile": {"playfulness": 0.42, "tempo": 0.51, "brightness": "bright_room"},
            "shared_core": {"anchor": ""},
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.primary_move == "bounce"
    assert shape.question_budget == 1
    assert shape.playfulness >= 0.3


def test_discourse_shape_uses_surface_context_bright_room_when_turn_delta_is_missing() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "visible_anchor", "text": "いま見えているところだけに絞って、一緒に見ていきます。"},
            {"act": "gentle_extension", "text": "必要なら、ここで大事なところへ少しだけ近づけます。"},
        ],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "constraints": {
                "max_questions": 0,
                "allow_small_next_step": True,
                "keep_thread_visible": True,
            },
            "surface_profile": {
                "voice_texture": "light_playful",
                "brightness": "open_play",
                "playfulness": 0.42,
                "tempo": 0.62,
            },
            "source_state": {
                "live_engagement_state": "hold",
                "lightness_budget_state": "open_play",
                "voice_texture": "light_playful",
            },
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.primary_move == "bounce"


def test_discourse_shape_prefers_reason_driven_bright_bounce_for_small_shared_smile() -> None:
    shape = derive_discourse_shape(
        content_sequence=[],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "constraints": {
                "max_questions": 1,
                "allow_small_next_step": False,
                "keep_thread_visible": True,
            },
            "surface_profile": {
                "voice_texture": "measured",
                "playfulness": 0.12,
                "tempo": 0.08,
            },
            "source_state": {
                "appraisal_state": "active",
                "appraisal_event": "laugh_break",
                "appraisal_shared_shift": "shared_smile_window",
                "meaning_update_state": "active",
                "meaning_update_relation": "shared_smile_window",
                "utterance_reason_state": "active",
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "utterance_reason_question_policy": "none",
                "utterance_reason_tone_hint": "chatty_ack",
                "live_engagement_state": "hold",
                "lightness_budget_state": "held",
            },
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.primary_move == "bounce"
    assert shape.secondary_move == "glow"
    assert shape.question_budget == 0
    assert shape.playfulness >= 0.38


def test_discourse_shape_uses_organism_posture_to_amplify_bright_bounce() -> None:
    shape = derive_discourse_shape(
        content_sequence=[],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "constraints": {
                "max_questions": 1,
                "allow_small_next_step": True,
            },
            "surface_profile": {
                "voice_texture": "measured",
                "playfulness": 0.12,
                "tempo": 0.08,
                "organism_posture": "play",
                "organism_expressive_readiness": 0.78,
                "organism_play_window": 0.74,
                "organism_relation_pull": 0.64,
                "organism_protective_tension": 0.18,
            },
            "source_state": {
                "live_engagement_state": "pickup_comment",
                "lightness_budget_state": "open_play",
            },
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.playfulness >= 0.52
    assert shape.tempo >= 0.38


def test_discourse_shape_uses_attuned_organism_posture_for_reflect_hold() -> None:
    shape = derive_discourse_shape(
        content_sequence=[],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "constraints": {"max_questions": 0},
            "surface_profile": {
                "organism_posture": "attune",
                "organism_attunement": 0.72,
                "organism_grounding": 0.58,
                "organism_play_window": 0.18,
            },
            "source_state": {},
        },
    )

    assert shape.shape_id == "reflect_hold"
    assert shape.secondary_move == "stay"


def test_discourse_shape_uses_protective_organism_posture_to_suppress_followup_question() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "shared_delight", "text": "x"},
            {"act": "light_bounce", "text": "y"},
        ],
        turn_delta={
            "kind": "bright_continuity",
            "preferred_act": "light_bounce",
        },
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "constraints": {"max_questions": 1},
            "surface_profile": {
                "playfulness": 0.42,
                "tempo": 0.44,
                "organism_posture": "protect",
                "organism_protective_tension": 0.72,
                "organism_play_window": 0.21,
                "organism_expressive_readiness": 0.34,
            },
            "source_state": {},
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.question_budget == 0
    assert shape.secondary_move == "glow"


def test_discourse_shape_does_not_escalate_repair_context_to_bright_bounce() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "acknowledge_overreach", "text": "x"},
            {"act": "visible_anchor", "text": "y"},
            {"act": "careful_reopen", "text": "z"},
            {"act": "light_bounce", "text": "w"},
        ],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "fresh_opening",
            "constraints": {
                "max_questions": 1,
                "allow_small_next_step": True,
            },
            "surface_profile": {
                "voice_texture": "light_playful",
                "playfulness": 0.42,
                "tempo": 0.48,
            },
            "source_state": {
                "interaction_policy_strategy": "repair_then_attune",
                "interaction_policy_opening_move": "name_overreach_and_reduce_force",
                "scene_family": "repair_window",
                "actuation_primary_action": "soft_repair",
                "live_engagement_state": "hold",
                "lightness_budget_state": "open_play",
            },
        },
    )

    assert shape.shape_id != "bright_bounce"
    assert shape.primary_move == "reflect"


def test_discourse_shape_does_not_escalate_guarded_self_view_to_bright_bounce() -> None:
    shape = derive_discourse_shape(
        content_sequence=[
            {"act": "shared_delight", "text": "x"},
            {"act": "light_bounce", "text": "y"},
        ],
        turn_delta={
            "kind": "bright_continuity",
            "preferred_act": "light_bounce",
        },
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "constraints": {
                "max_questions": 0,
                "allow_small_next_step": True,
            },
            "surface_profile": {
                "voice_texture": "light_playful",
                "playfulness": 0.42,
                "tempo": 0.48,
            },
            "source_state": {
                "shared_presence_mode": "guarded_boundary",
                "shared_presence_co_presence": 0.22,
                "shared_presence_boundary_stability": 0.18,
                "self_other_dominant_attribution": "unknown",
                "self_other_unknown_likelihood": 0.76,
                "subjective_scene_anchor_frame": "ambient_margin",
                "subjective_scene_shared_scene_potential": 0.18,
            },
        },
    )

    assert shape.shape_id == "reflect_hold"
    assert shape.energy == "guarded"
    assert shape.question_budget == 0


def test_discourse_shape_allows_reason_driven_bright_bounce_when_shared_self_view_is_ready() -> None:
    shape = derive_discourse_shape(
        content_sequence=[],
        turn_delta={},
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "constraints": {
                "max_questions": 0,
                "allow_small_next_step": False,
            },
            "surface_profile": {
                "voice_texture": "measured",
                "playfulness": 0.12,
                "tempo": 0.08,
            },
            "source_state": {
                "appraisal_state": "active",
                "appraisal_event": "laugh_break",
                "appraisal_shared_shift": "shared_smile_window",
                "meaning_update_state": "active",
                "meaning_update_relation": "shared_smile_window",
                "utterance_reason_state": "active",
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "utterance_reason_question_policy": "none",
                "utterance_reason_tone_hint": "chatty_ack",
                "shared_presence_mode": "inhabited_shared_space",
                "shared_presence_co_presence": 0.72,
                "shared_presence_boundary_stability": 0.64,
                "self_other_dominant_attribution": "shared",
                "self_other_unknown_likelihood": 0.12,
                "subjective_scene_anchor_frame": "shared_margin",
                "subjective_scene_shared_scene_potential": 0.68,
            },
        },
    )

    assert shape.shape_id == "bright_bounce"
    assert shape.question_budget == 0
