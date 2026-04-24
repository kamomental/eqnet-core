# -*- coding: utf-8 -*-

from inner_os.expression.surface_context_packet import (
    SurfaceContextPacket,
    build_surface_context_packet,
)


def test_build_surface_context_packet_keeps_reopening_anchor_and_constraints() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={
            "state": "reopening_thread",
            "recent_anchor": "前に話していた約束",
        },
        discussion_thread_state={
            "state": "revisit_issue",
            "topic_anchor": "あの約束",
        },
        issue_state={"state": "pausing_issue"},
        turn_delta={
            "kind": "reopening_thread",
            "preferred_act": "reopen_from_anchor",
            "anchor_hint": "あの約束",
        },
        interaction_constraints={
            "avoid_obvious_advice": True,
            "keep_thread_visible": True,
            "prefer_return_point": True,
        },
        boundary_transform={"surface_mode": "soft_close"},
        residual_reflection={
            "focus": "まだ言えていないこと",
            "reasons": ["withheld_need"],
        },
        surface_profile={
            "response_length": "short",
            "cultural_register": "casual_shared",
            "group_register": "one_to_one",
            "sentence_temperature": "gentle",
        },
        contact_reflection_state={"reflection_style": "reflect_only"},
        green_kernel_composition={
            "field": {
                "guardedness": 0.72,
                "reopening_pull": 0.64,
                "affective_charge": 0.58,
            }
        },
        heartbeat_structure_state={
            "pulse_band": "lifted_pulse",
            "phase_window": "upswing",
            "dominant_reaction": "attune",
            "activation_drive": 0.46,
            "containment_bias": 0.22,
            "bounce_room": 0.34,
            "response_tempo": 0.41,
        },
        external_field_state={
            "dominant_field": "continuity_field",
            "social_mode": "one_to_one",
            "thread_mode": "reopening_thread",
            "environmental_load": 0.21,
            "social_pressure": 0.18,
            "continuity_pull": 0.67,
            "ambiguity_load": 0.24,
            "safety_envelope": 0.74,
            "novelty": 0.13,
        },
        terrain_dynamics_state={
            "dominant_basin": "continuity_basin",
            "dominant_flow": "reenter",
            "terrain_energy": 0.39,
            "entropy": 0.18,
            "ignition_pressure": 0.31,
            "barrier_height": 0.23,
            "recovery_gradient": 0.51,
            "basin_pull": 0.62,
        },
        dialogue_context={"user_text": "その続きなら、いま少し話せるかも。"},
    ).to_dict()

    assert packet["conversation_phase"] == "reopening_thread"
    assert packet["shared_core"]["anchor"] == "あの約束"
    assert "あの約束" in packet["shared_core"]["already_shared"]
    assert "まだ言えていないこと" in packet["shared_core"]["not_yet_shared"]
    assert packet["response_role"]["primary"] == "reopen_from_anchor"
    assert packet["response_role"]["secondary"] == "reflect_only"
    assert packet["constraints"]["no_generic_clarification"] is True
    assert packet["constraints"]["no_advice"] is True
    assert packet["constraints"]["max_questions"] == 0
    assert packet["constraints"]["keep_thread_visible"] is True
    assert packet["surface_profile"]["cultural_register"] == "casual_shared"
    assert packet["surface_profile"]["heartbeat_reaction"] == "attune"
    assert packet["surface_profile"]["heartbeat_pulse_band"] == "lifted_pulse"
    assert packet["surface_profile"]["heartbeat_tempo"] == 0.41
    assert packet["surface_profile"]["external_field_dominant"] == "continuity_field"
    assert packet["surface_profile"]["terrain_dominant_basin"] == "continuity_basin"
    assert packet["source_state"]["green_guardedness"] == 0.72
    assert packet["source_state"]["heartbeat_phase_window"] == "upswing"
    assert packet["source_state"]["heartbeat_response_tempo"] == 0.41
    assert packet["source_state"]["external_field_social_mode"] == "one_to_one"
    assert packet["source_state"]["terrain_recovery_gradient"] == 0.51


def test_surface_context_packet_is_mapping_compatible_contract() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "continuing_thread"},
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        surface_profile={"response_length": "short"},
    )

    assert isinstance(packet, SurfaceContextPacket)
    assert packet["conversation_phase"] == "bright_continuity"
    assert dict(packet)["response_role"]["primary"] == packet["response_role"]["primary"]


def test_build_surface_context_packet_carries_shared_moment_state() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "continuing_thread"},
        discussion_thread_state={"state": "continuing_thread"},
        issue_state={"state": "bright_issue"},
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        interaction_constraints={"allow_small_next_step": True},
        surface_profile={
            "response_length": "short",
            "cultural_register": "casual_shared",
            "group_register": "threaded_group",
            "sentence_temperature": "gentle",
        },
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.7,
            "afterglow": 0.62,
            "cue_text": "ちょっと笑えたこと",
        },
        dialogue_context={"user_text": "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"},
    ).to_dict()

    assert "ちょっと笑えたこと" in packet["shared_core"]["already_shared"]
    assert packet["surface_profile"]["shared_moment_kind"] == "laugh"
    assert packet["surface_profile"]["shared_moment_afterglow"] == 0.62
    assert packet["source_state"]["shared_moment_state"] == "shared_moment"
    assert packet["source_state"]["shared_moment_kind"] == "laugh"
    assert packet["source_state"]["shared_moment_score"] == 0.74


def test_build_surface_context_packet_carries_listener_action_state() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "continuing_thread"},
        discussion_thread_state={"state": "continuing_thread"},
        issue_state={"state": "bright_issue"},
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        interaction_constraints={"allow_small_next_step": True},
        surface_profile={"response_length": "short"},
        listener_action_state={
            "state": "warm_laugh_ack",
            "filler_mode": "playful",
            "token_profile": "soft_laugh",
            "score": 0.72,
        },
        dialogue_context={"user_text": "そのあとちょっと笑えることもあって。"},
    ).to_dict()

    assert packet["surface_profile"]["listener_action"] == "warm_laugh_ack"
    assert packet["surface_profile"]["listener_filler_mode"] == "playful"
    assert packet["surface_profile"]["listener_token_profile"] == "soft_laugh"
    assert packet["source_state"]["listener_action_state"] == "warm_laugh_ack"
    assert packet["source_state"]["listener_action_score"] == 0.72


def test_build_surface_context_packet_carries_joint_state() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "continuing_thread"},
        discussion_thread_state={"state": "continuing_thread"},
        issue_state={"state": "bright_issue"},
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        surface_profile={"response_length": "short"},
        joint_state={
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.72,
            "shared_tension": 0.18,
            "repair_readiness": 0.49,
            "common_ground": 0.66,
            "joint_attention": 0.61,
            "mutual_room": 0.57,
            "coupling_strength": 0.69,
        },
    ).to_dict()

    assert packet["surface_profile"]["joint_mode"] == "delighted_jointness"
    assert packet["surface_profile"]["joint_shared_delight"] == 0.72
    assert packet["source_state"]["joint_common_ground"] == 0.66
    assert packet["source_state"]["joint_coupling_strength"] == 0.69
