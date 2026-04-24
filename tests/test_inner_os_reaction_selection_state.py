from inner_os.actuation_plan import derive_actuation_plan
from inner_os.nonverbal_response_state import derive_nonverbal_response_state
from inner_os.presence_hold_state import derive_presence_hold_state
from inner_os.response_selection_state import derive_response_selection_state


def test_presence_hold_state_can_open_reentry_for_shared_laugh() -> None:
    state = derive_presence_hold_state(
        live_engagement_state={"state": "riff_with_comment", "score": 0.58},
        listener_action_state={
            "state": "warm_laugh_ack",
            "acknowledgement_room": 0.52,
            "laughter_room": 0.72,
            "filler_room": 0.48,
        },
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "afterglow": 0.62,
        },
        joint_state={
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.72,
            "shared_tension": 0.16,
            "common_ground": 0.68,
            "mutual_room": 0.62,
            "coupling_strength": 0.64,
        },
        organism_state={"dominant_posture": "play", "play_window": 0.7, "protective_tension": 0.18},
        external_field_state={"continuity_pull": 0.66, "safety_envelope": 0.34},
        terrain_dynamics_state={"recovery_gradient": 0.62, "barrier_height": 0.22},
    ).to_dict()

    assert state["state"] == "reentry_open"
    assert state["release_readiness"] > state["hold_room"]


def test_nonverbal_response_state_prefers_warm_laugh_ack_when_shared_smile_is_present() -> None:
    presence = derive_presence_hold_state(
        live_engagement_state={"state": "pickup_comment", "score": 0.52},
        listener_action_state={
            "state": "warm_laugh_ack",
            "acknowledgement_room": 0.48,
            "laughter_room": 0.68,
            "filler_room": 0.42,
        },
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.7,
            "afterglow": 0.58,
        },
        joint_state={
            "dominant_mode": "shared_attention",
            "shared_delight": 0.62,
            "shared_tension": 0.14,
            "common_ground": 0.64,
            "mutual_room": 0.58,
            "coupling_strength": 0.56,
        },
        organism_state={"dominant_posture": "attune", "play_window": 0.48, "protective_tension": 0.18},
        external_field_state={"continuity_pull": 0.54, "safety_envelope": 0.36},
        terrain_dynamics_state={"recovery_gradient": 0.56, "barrier_height": 0.24},
    ).to_dict()
    state = derive_nonverbal_response_state(
        listener_action_state={
            "state": "warm_laugh_ack",
            "score": 0.66,
            "token_profile": "soft_laugh",
            "filler_mode": "playful",
            "acknowledgement_room": 0.48,
            "laughter_room": 0.68,
        },
        presence_hold_state=presence,
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.7,
        },
        live_engagement_state={"state": "pickup_comment"},
        joint_state={"shared_delight": 0.62, "shared_tension": 0.14},
    ).to_dict()

    assert state["state"] == "warm_laugh_ack"
    assert state["response_kind"] == "backchannel"
    assert state["timing_bias"] == "near_turn_end"


def test_response_selection_state_prefers_speak_when_riff_action_is_active() -> None:
    state = derive_response_selection_state(
        primary_action="riff_current_comment",
        execution_mode="shared_progression",
        reply_permission="speak_forward",
        defer_dominance=0.18,
        live_engagement_state={"state": "riff_with_comment", "score": 0.58},
        presence_hold_state={
            "state": "reentry_open",
            "hold_room": 0.18,
            "backchannel_room": 0.64,
            "release_readiness": 0.68,
        },
        nonverbal_response_state={"state": "warm_laugh_ack", "response_kind": "backchannel", "score": 0.66},
        shared_moment_state={"state": "shared_moment", "score": 0.72},
    ).to_dict()

    assert state["channel"] == "speak"
    assert state["speak_room"] >= state["backchannel_room"]


def test_response_selection_state_can_prefer_backchannel_for_relation_reentry() -> None:
    state = derive_response_selection_state(
        primary_action="hold_presence",
        execution_mode="attuned_contact",
        reply_permission="speak_briefly",
        defer_dominance=0.12,
        live_engagement_state={"state": "hold", "score": 0.12},
        presence_hold_state={
            "state": "reentry_open",
            "hold_room": 0.22,
            "backchannel_room": 0.66,
            "release_readiness": 0.46,
        },
        nonverbal_response_state={
            "state": "bridge_ack_presence",
            "response_kind": "backchannel",
            "score": 0.54,
        },
        shared_moment_state={"state": "none", "score": 0.0},
        utterance_reason_packet={
            "state": "active",
            "relation_frame": "cross_context_bridge",
            "causal_frame": "reframing_cause",
            "memory_frame": "name_distant_link",
            "preserve": "keep_thread_visible",
        },
    ).to_dict()

    assert state["channel"] == "backchannel"
    assert "reason:relation_reentry" in state["dominant_inputs"]


def test_response_selection_state_can_hold_for_guarded_relation() -> None:
    state = derive_response_selection_state(
        primary_action="hold_presence",
        execution_mode="defer_with_presence",
        reply_permission="hold_or_brief",
        defer_dominance=0.18,
        live_engagement_state={"state": "hold", "score": 0.08},
        presence_hold_state={
            "state": "holding_space",
            "hold_room": 0.68,
            "backchannel_room": 0.14,
            "release_readiness": 0.12,
        },
        nonverbal_response_state={
            "state": "guarded_hold_presence",
            "response_kind": "hold",
            "score": 0.58,
        },
        shared_moment_state={"state": "none", "score": 0.0},
        utterance_reason_packet={
            "state": "active",
            "relation_frame": "unfinished_link",
            "causal_frame": "unfinished_thread_cause",
            "memory_frame": "keep_unfinished_link_near",
            "preserve": "do_not_overclaim",
        },
    ).to_dict()

    assert state["channel"] == "hold"
    assert "reason:guarded_relation" in state["dominant_inputs"]


def test_response_selection_state_joins_when_shared_presence_is_inhabited() -> None:
    state = derive_response_selection_state(
        primary_action="hold_presence",
        execution_mode="attuned_contact",
        reply_permission="speak_briefly",
        defer_dominance=0.08,
        live_engagement_state={"state": "hold", "score": 0.18},
        presence_hold_state={
            "state": "reentry_open",
            "hold_room": 0.16,
            "backchannel_room": 0.42,
            "release_readiness": 0.38,
        },
        nonverbal_response_state={
            "state": "bridge_ack_presence",
            "response_kind": "backchannel",
            "score": 0.34,
        },
        shared_moment_state={"state": "none", "score": 0.0},
        subjective_scene_state={
            "anchor_frame": "shared_margin",
            "shared_scene_potential": 0.68,
        },
        self_other_attribution_state={
            "dominant_attribution": "shared",
            "unknown_likelihood": 0.12,
        },
        shared_presence_state={
            "dominant_mode": "inhabited_shared_space",
            "co_presence": 0.72,
            "boundary_stability": 0.62,
        },
    ).to_dict()

    assert state["channel"] == "backchannel"
    assert "reason:shared_presence_join" in state["dominant_inputs"]


def test_response_selection_state_holds_when_self_view_is_uncertain() -> None:
    state = derive_response_selection_state(
        primary_action="riff_current_comment",
        execution_mode="shared_progression",
        reply_permission="speak_forward",
        defer_dominance=0.18,
        live_engagement_state={"state": "riff_with_comment", "score": 0.58},
        presence_hold_state={
            "state": "holding_space",
            "hold_room": 0.38,
            "backchannel_room": 0.2,
            "release_readiness": 0.18,
        },
        nonverbal_response_state={"state": "guarded_hold_presence", "response_kind": "hold", "score": 0.42},
        shared_moment_state={"state": "none", "score": 0.0},
        subjective_scene_state={
            "anchor_frame": "ambient_margin",
            "shared_scene_potential": 0.18,
        },
        self_other_attribution_state={
            "dominant_attribution": "unknown",
            "unknown_likelihood": 0.74,
        },
        shared_presence_state={
            "dominant_mode": "guarded_boundary",
            "co_presence": 0.24,
            "boundary_stability": 0.2,
        },
    ).to_dict()

    assert state["channel"] == "hold"
    assert "reason:self_view_guard" in state["dominant_inputs"]


def test_actuation_plan_exposes_reaction_first_states_for_shared_smile_riff() -> None:
    packet = {
        "response_strategy": "shared_world_next_step",
        "live_engagement_state": {
            "state": "riff_with_comment",
            "score": 0.58,
            "primary_move": "riff_current_comment",
        },
        "shared_moment_state": {
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.68,
            "afterglow": 0.62,
        },
        "listener_action_state": {
            "state": "warm_laugh_ack",
            "score": 0.66,
            "acknowledgement_room": 0.52,
            "laughter_room": 0.72,
            "filler_room": 0.48,
            "filler_mode": "playful",
            "token_profile": "soft_laugh",
        },
        "joint_state": {
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.72,
            "shared_tension": 0.16,
            "common_ground": 0.68,
            "joint_attention": 0.64,
            "mutual_room": 0.62,
            "coupling_strength": 0.64,
        },
        "organism_state": {
            "dominant_posture": "play",
            "play_window": 0.7,
            "protective_tension": 0.18,
        },
        "external_field_state": {"continuity_pull": 0.66, "safety_envelope": 0.34},
        "terrain_dynamics_state": {"recovery_gradient": 0.62, "barrier_height": 0.22},
        "conversation_contract": {
            "response_action_now": {
                "ordered_operations": ["offer_small_next_step"],
            }
        },
    }
    posture = {
        "engagement_mode": "co_move",
        "primary_operation_kind": "offer_small_next_step",
        "live_engagement_state": packet["live_engagement_state"],
        "shared_moment_state": packet["shared_moment_state"],
        "listener_action_state": packet["listener_action_state"],
        "joint_state": packet["joint_state"],
        "joint_mode": "delighted_jointness",
        "joint_reentry_room": 0.62,
        "organism_state": packet["organism_state"],
        "organism_posture_name": "play",
        "external_field_state": {
            "dominant_field": "continuity_field",
            "continuity_pull": 0.66,
            "safety_envelope": 0.34,
        },
        "external_field_name": "continuity_field",
        "terrain_dynamics_state": {
            "dominant_basin": "continuity_basin",
            "dominant_flow": "reenter",
            "recovery_gradient": 0.62,
            "barrier_height": 0.22,
        },
        "terrain_basin_name": "continuity_basin",
        "terrain_flow_name": "reenter",
    }
    actuation = derive_actuation_plan(packet, posture)

    assert actuation["response_channel"] == "speak"
    assert actuation["presence_hold_state"]["state"] == "reentry_open"
    assert actuation["nonverbal_response_state"]["state"] in {
        "warm_laugh_ack",
        "soft_ack_presence",
        "lead_in_presence",
    }
    assert actuation["response_selection_state"]["channel"] == "speak"


def test_actuation_plan_uses_self_view_states_in_response_selection() -> None:
    packet = {
        "response_strategy": "attuned_receive",
        "live_engagement_state": {
            "state": "hold",
            "score": 0.18,
            "primary_move": "stay_present",
        },
        "listener_action_state": {
            "state": "bridge_ack_presence",
            "score": 0.34,
            "acknowledgement_room": 0.28,
            "laughter_room": 0.0,
            "filler_room": 0.0,
        },
        "joint_state": {
            "dominant_mode": "shared_attention",
            "shared_delight": 0.28,
            "shared_tension": 0.12,
            "common_ground": 0.44,
            "joint_attention": 0.42,
            "mutual_room": 0.4,
            "coupling_strength": 0.38,
        },
        "organism_state": {
            "dominant_posture": "attune",
            "play_window": 0.22,
            "protective_tension": 0.18,
        },
        "external_field_state": {"continuity_pull": 0.42, "safety_envelope": 0.48},
        "terrain_dynamics_state": {"recovery_gradient": 0.46, "barrier_height": 0.26},
        "subjective_scene_state": {
            "anchor_frame": "shared_margin",
            "shared_scene_potential": 0.66,
        },
        "self_other_attribution_state": {
            "dominant_attribution": "shared",
            "unknown_likelihood": 0.12,
        },
        "shared_presence_state": {
            "dominant_mode": "inhabited_shared_space",
            "co_presence": 0.72,
            "boundary_stability": 0.62,
        },
        "conversation_contract": {
            "response_action_now": {
                "ordered_operations": ["stay_present"],
                "question_budget": 0,
            }
        },
    }
    posture = {
        "engagement_mode": "attuned_contact",
        "primary_operation_kind": "stay_present",
        "live_engagement_state": packet["live_engagement_state"],
        "listener_action_state": packet["listener_action_state"],
        "joint_state": packet["joint_state"],
        "organism_state": packet["organism_state"],
        "external_field_state": packet["external_field_state"],
        "terrain_dynamics_state": packet["terrain_dynamics_state"],
        "subjective_scene_state": packet["subjective_scene_state"],
        "self_other_attribution_state": packet["self_other_attribution_state"],
        "shared_presence_state": packet["shared_presence_state"],
    }

    actuation = derive_actuation_plan(packet, posture)

    assert actuation["response_channel"] in {"speak", "backchannel"}
    assert "attribution:shared" in actuation["response_selection_state"]["dominant_inputs"]
    assert (
        "reason:shared_presence_join"
        in actuation["response_selection_state"]["dominant_inputs"]
    )


def test_actuation_plan_can_select_hold_channel_for_guarded_presence() -> None:
    packet = {
        "response_strategy": "contain_then_stabilize",
        "live_engagement_state": {
            "state": "hold",
            "score": 0.18,
            "primary_move": "hold_presence",
        },
        "listener_action_state": {
            "state": "none",
            "score": 0.08,
            "acknowledgement_room": 0.12,
            "laughter_room": 0.06,
            "filler_room": 0.08,
            "filler_mode": "caregiver",
            "token_profile": "plain_ack",
        },
        "joint_state": {
            "dominant_mode": "strained_jointness",
            "shared_delight": 0.08,
            "shared_tension": 0.62,
            "common_ground": 0.18,
            "joint_attention": 0.14,
            "mutual_room": 0.12,
            "coupling_strength": 0.16,
        },
        "organism_state": {
            "dominant_posture": "protect",
            "play_window": 0.08,
            "protective_tension": 0.72,
        },
        "external_field_state": {"continuity_pull": 0.18, "safety_envelope": 0.74},
        "terrain_dynamics_state": {"recovery_gradient": 0.12, "barrier_height": 0.72},
    }
    actuation = derive_actuation_plan(packet, {})

    assert actuation["response_channel"] == "hold"
    assert actuation["reply_permission"] in {"hold_or_brief", "speak_minimal"}
    assert actuation["presence_hold_state"]["state"] in {"holding_space", "backchannel_ready_hold"}


def test_actuation_plan_can_surface_bridge_backchannel_without_shared_moment() -> None:
    packet = {
        "response_strategy": "respectful_wait",
        "live_engagement_state": {
            "state": "hold",
            "score": 0.12,
            "primary_move": "hold_presence",
        },
        "listener_action_state": {
            "state": "soft_ack",
            "score": 0.42,
            "acknowledgement_room": 0.56,
            "laughter_room": 0.08,
            "filler_room": 0.24,
            "filler_mode": "soft",
            "token_profile": "plain_ack",
        },
        "utterance_reason_packet": {
            "state": "active",
            "relation_frame": "cross_context_bridge",
            "causal_frame": "reframing_cause",
            "memory_frame": "name_distant_link",
            "preserve": "keep_thread_visible",
        },
        "joint_state": {
            "dominant_mode": "shared_attention",
            "shared_delight": 0.22,
            "shared_tension": 0.12,
            "common_ground": 0.62,
            "joint_attention": 0.58,
            "mutual_room": 0.54,
            "coupling_strength": 0.5,
        },
        "organism_state": {
            "dominant_posture": "attune",
            "play_window": 0.24,
            "protective_tension": 0.18,
        },
        "external_field_state": {"continuity_pull": 0.62, "safety_envelope": 0.46},
        "terrain_dynamics_state": {"recovery_gradient": 0.54, "barrier_height": 0.22},
    }
    actuation = derive_actuation_plan(packet, {})

    assert actuation["presence_hold_state"]["state"] == "reentry_open"
    assert actuation["nonverbal_response_state"]["state"] == "bridge_ack_presence"
    assert actuation["response_channel"] == "backchannel"
