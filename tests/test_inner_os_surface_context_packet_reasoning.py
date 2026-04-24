from inner_os.expression.surface_context_packet import build_surface_context_packet


def test_surface_context_packet_carries_appraisal_and_reason_chain() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={"state": "bright_continuity"},
        discussion_thread_state={"state": "open_thread", "topic_anchor": "harbor"},
        issue_state={"state": "light_tension"},
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        interaction_constraints={"keep_thread_visible": True},
        surface_profile={"response_length": "short", "surface_mode": "held"},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.62,
            "afterglow": 0.58,
        },
        listener_action_state={
            "state": "warm_laugh_ack",
            "filler_mode": "playful",
            "token_profile": "soft_laugh",
            "score": 0.72,
        },
        joint_state={
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.74,
            "shared_tension": 0.18,
            "repair_readiness": 0.52,
            "common_ground": 0.68,
            "joint_attention": 0.63,
            "mutual_room": 0.57,
            "coupling_strength": 0.71,
        },
        appraisal_state={
            "state": "active",
            "background_state": "awkwardness_present",
            "moment_event": "laugh_break",
            "shared_shift": "shared_smile_window",
            "dominant_relation_type": "same_anchor",
            "dominant_relation_key": "harbor->promise",
            "relation_meta_type": "reinforces",
            "dominant_causal_type": "enabled_by",
            "dominant_causal_key": "harbor->promise:enabled_by",
            "memory_mode": "ignite",
            "recall_anchor": "harbor",
            "memory_resonance": 0.58,
            "easing_shift": 0.61,
        },
        meaning_update_state={
            "state": "active",
            "self_update": "guard_relaxes_for_moment",
            "relation_update": "shared_smile_window",
            "relation_frame": "same_anchor_link",
            "relation_key": "harbor->promise",
            "relation_meta_type": "reinforces",
            "causal_frame": "same_anchor_cause",
            "causal_key": "harbor->promise:enabled_by",
            "world_update": "small_moment_on_known_thread",
            "memory_update": "known_thread_returns",
            "recall_anchor": "harbor",
            "memory_resonance": 0.58,
            "preserve_guard": "keep_it_small_and_linked",
        },
        utterance_reason_packet={
            "state": "active",
            "reaction_target": "small_laugh_moment",
            "reason_frame": "name_shared_shift",
            "relation_frame": "same_anchor_link",
            "relation_key": "harbor->promise",
            "causal_frame": "same_anchor_cause",
            "causal_key": "harbor->promise:enabled_by",
            "memory_frame": "echo_known_thread",
            "memory_anchor": "harbor",
            "offer": "brief_shared_smile",
            "preserve": "keep_it_small_and_linked",
            "question_policy": "none",
            "tone_hint": "chatty_ack",
        },
        organism_state={
            "dominant_posture": "play",
            "relation_focus": "user",
            "social_mode": "one_to_one",
            "attunement": 0.72,
            "coherence": 0.68,
            "grounding": 0.61,
            "protective_tension": 0.22,
            "expressive_readiness": 0.7,
            "play_window": 0.76,
            "relation_pull": 0.66,
            "social_exposure": 0.18,
        },
        memory_dynamics_state={
            "dominant_mode": "ignite",
            "dominant_relation_type": "same_anchor",
            "relation_generation_mode": "ignited",
            "dominant_causal_type": "enabled_by",
            "causal_generation_mode": "anchored",
            "palace_mode": "anchored",
            "monument_mode": "rising",
            "ignition_mode": "active",
            "reconsolidation_mode": "settle",
            "recall_anchor": "harbor",
            "monument_salience": 0.61,
            "activation_confidence": 0.58,
            "memory_tension": 0.22,
        },
    ).to_dict()

    assert packet["surface_profile"]["appraisal_event"] == "laugh_break"
    assert packet["surface_profile"]["meaning_update_relation"] == "shared_smile_window"
    assert packet["surface_profile"]["appraisal_relation_type"] == "same_anchor"
    assert packet["surface_profile"]["appraisal_causal_type"] == "enabled_by"
    assert packet["surface_profile"]["meaning_update_relation_frame"] == "same_anchor_link"
    assert packet["surface_profile"]["meaning_update_causal_frame"] == "same_anchor_cause"
    assert packet["surface_profile"]["utterance_reason_relation_frame"] == "same_anchor_link"
    assert packet["surface_profile"]["utterance_reason_causal_frame"] == "same_anchor_cause"
    assert packet["surface_profile"]["utterance_reason_offer"] == "brief_shared_smile"
    assert packet["surface_profile"]["joint_mode"] == "delighted_jointness"
    assert packet["surface_profile"]["joint_shared_delight"] == 0.74
    assert packet["surface_profile"]["memory_dynamics_mode"] == "ignite"
    assert packet["surface_profile"]["memory_recall_anchor"] == "harbor"
    assert packet["source_state"]["appraisal_shared_shift"] == "shared_smile_window"
    assert packet["source_state"]["appraisal_dominant_relation_type"] == "same_anchor"
    assert packet["source_state"]["appraisal_dominant_relation_key"] == "harbor->promise"
    assert packet["source_state"]["appraisal_dominant_causal_type"] == "enabled_by"
    assert packet["source_state"]["appraisal_dominant_causal_key"] == "harbor->promise:enabled_by"
    assert packet["source_state"]["joint_common_ground"] == 0.68
    assert packet["source_state"]["joint_coupling_strength"] == 0.71
    assert packet["source_state"]["appraisal_memory_mode"] == "ignite"
    assert packet["source_state"]["meaning_update_memory"] == "known_thread_returns"
    assert packet["source_state"]["meaning_update_relation_frame"] == "same_anchor_link"
    assert packet["source_state"]["meaning_update_relation_key"] == "harbor->promise"
    assert packet["source_state"]["meaning_update_causal_frame"] == "same_anchor_cause"
    assert packet["source_state"]["meaning_update_causal_key"] == "harbor->promise:enabled_by"
    assert packet["source_state"]["meaning_update_world"] == "small_moment_on_known_thread"
    assert packet["source_state"]["utterance_reason_relation_frame"] == "same_anchor_link"
    assert packet["source_state"]["utterance_reason_relation_key"] == "harbor->promise"
    assert packet["source_state"]["utterance_reason_causal_frame"] == "same_anchor_cause"
    assert packet["source_state"]["utterance_reason_causal_key"] == "harbor->promise:enabled_by"
    assert packet["source_state"]["utterance_reason_memory_frame"] == "echo_known_thread"
    assert packet["source_state"]["memory_ignition_mode"] == "active"
    assert packet["source_state"]["memory_dominant_relation_type"] == "same_anchor"
    assert packet["source_state"]["memory_relation_generation_mode"] == "ignited"
    assert packet["source_state"]["memory_dominant_causal_type"] == "enabled_by"
    assert packet["source_state"]["memory_causal_generation_mode"] == "anchored"
    assert packet["source_state"]["utterance_reason_preserve"] == "keep_it_small_and_linked"
    assert packet["source_state"]["utterance_reason_question_policy"] == "none"
    assert packet["surface_profile"]["organism_posture"] == "play"
    assert packet["surface_profile"]["organism_play_window"] == 0.76
    assert packet["source_state"]["organism_relation_focus"] == "user"
    assert packet["source_state"]["organism_protective_tension"] == 0.22
