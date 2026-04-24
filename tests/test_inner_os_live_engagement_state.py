from types import SimpleNamespace

from inner_os.action_posture import ActionPostureContract, derive_action_posture
from inner_os.actuation_plan import ActuationPlanContract, derive_actuation_plan
from inner_os.live_engagement_state import derive_live_engagement_state
from inner_os.policy_packet import (
    _derive_field_strategy_override,
    derive_interaction_policy_packet,
)


def test_live_engagement_state_prefers_comment_pickup_in_streaming_ask_mode() -> None:
    state = derive_live_engagement_state(
        self_state={"mode": "streaming", "talk_mode": "ask"},
        initiative_readiness={"state": "ready", "score": 0.52},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.48},
        relational_style_memory_state={
            "state": "grounded_gentle",
            "banter_room": 0.24,
            "playful_ceiling": 0.2,
            "lexical_variation_bias": 0.18,
        },
        lightness_budget_state={"state": "light_ok", "banter_room": 0.26},
        social_topology_state={"state": "threaded_group", "score": 0.46},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
    ).to_dict()

    assert state["state"] == "pickup_comment"
    assert state["primary_move"] == "pick_up_comment"
    assert state["comment_pickup_room"] >= state["riff_room"]


def test_live_engagement_state_holds_when_not_streaming_and_guarded() -> None:
    state = derive_live_engagement_state(
        self_state={"mode": "chat", "talk_mode": "watch"},
        initiative_readiness={"state": "hold", "score": 0.18},
        initiative_followup_bias={"state": "hold", "score": 0.12},
        relational_style_memory_state={
            "state": "grounded_gentle",
            "banter_room": 0.08,
            "playful_ceiling": 0.08,
            "lexical_variation_bias": 0.06,
        },
        lightness_budget_state={"state": "grounded_only", "banter_room": 0.04},
        social_topology_state={"state": "one_to_one", "score": 0.2},
        body_recovery_guard={"state": "guarded", "score": 0.72},
        body_homeostasis_state={"state": "recovering", "score": 0.58},
        homeostasis_budget_state={"state": "depleted", "score": 0.64},
    ).to_dict()

    assert state["state"] == "hold"
    assert state["primary_move"] == "hold_presence"


def test_live_engagement_state_uses_shared_moment_to_open_pickup_room() -> None:
    state = derive_live_engagement_state(
        self_state={"mode": "chat", "talk_mode": "talk"},
        initiative_readiness={"state": "hold", "score": 0.24},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.22},
        relational_style_memory_state={
            "state": "warm_companion",
            "banter_room": 0.22,
            "playful_ceiling": 0.24,
            "lexical_variation_bias": 0.18,
        },
        lightness_budget_state={"state": "open_play", "banter_room": 0.32},
        social_topology_state={"state": "threaded_group", "score": 0.42},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.12},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.72,
            "afterglow": 0.64,
        },
        joint_state={
            "dominant_mode": "shared_attention",
            "shared_delight": 0.58,
            "common_ground": 0.62,
            "joint_attention": 0.64,
            "mutual_room": 0.48,
            "coupling_strength": 0.54,
        },
    ).to_dict()

    assert state["state"] in {"pickup_comment", "riff_with_comment"}
    assert "shared_moment" in state["dominant_inputs"]
    assert "joint_room_open" in state["dominant_inputs"]


def test_action_posture_and_actuation_plan_are_mapping_compatible_contracts() -> None:
    packet = {
        "response_strategy": "attune_then_extend",
        "conversation_contract": {
            "response_action_now": {
                "primary_operation": "offer_small_next_step",
                "ordered_operations": ["offer_small_next_step"],
                "question_budget": 1,
            },
            "ordered_effects": ["enable_small_next_step"],
        },
        "primary_object_operation": {"operation_kind": "offer_small_next_step"},
        "ordered_operation_kinds": ["offer_small_next_step"],
        "ordered_effect_kinds": ["enable_small_next_step"],
        "joint_state": {"dominant_mode": "shared_attention", "common_ground": 0.58},
        "terrain_dynamics_state": {"dominant_basin": "continuity_basin", "dominant_flow": "reentry"},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert isinstance(posture, ActionPostureContract)
    assert posture["engagement_mode"] == "co_move"
    assert dict(posture)["outcome_goal"] == posture["outcome_goal"]
    assert isinstance(actuation, ActuationPlanContract)
    assert actuation["execution_mode"] == "shared_progression"
    assert dict(actuation)["primary_action"] == actuation["primary_action"]


def test_live_engagement_state_prefers_pickup_for_chat_shared_laugh() -> None:
    state = derive_live_engagement_state(
        self_state={"mode": "chat", "talk_mode": "talk"},
        initiative_readiness={"state": "hold", "score": 0.18},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.2},
        relational_style_memory_state={
            "state": "warm_companion",
            "banter_room": 0.18,
            "playful_ceiling": 0.22,
            "lexical_variation_bias": 0.14,
        },
        lightness_budget_state={"state": "open_play", "banter_room": 0.34},
        social_topology_state={"state": "one_to_one", "score": 0.46},
        body_recovery_guard={"state": "open", "score": 0.06},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.72,
            "jointness": 0.58,
            "afterglow": 0.64,
        },
    ).to_dict()

    assert state["state"] in {"pickup_comment", "riff_with_comment"}
    assert "shared_moment_pickup_bonus" in state["dominant_inputs"]


def test_live_engagement_state_can_promote_watch_to_talk_for_shared_laugh_reply() -> None:
    state = derive_live_engagement_state(
        self_state={
            "mode": "chat",
            "talk_mode": "watch",
            "surface_user_text": "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        },
        initiative_readiness={"state": "ready", "score": 0.32},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.24},
        relational_style_memory_state={
            "state": "warm_companion",
            "banter_room": 0.22,
            "playful_ceiling": 0.22,
            "lexical_variation_bias": 0.14,
        },
        lightness_budget_state={"state": "open_play", "banter_room": 0.36},
        social_topology_state={"state": "one_to_one", "score": 0.44},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.72,
            "jointness": 0.58,
            "afterglow": 0.64,
        },
    ).to_dict()

    assert state["state"] in {"pickup_comment", "riff_with_comment"}
    assert "shared_moment_talk_override" in state["dominant_inputs"]


def test_policy_packet_exposes_live_engagement_state_for_streaming_context() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="comment:latest",
        current_risks=[],
        reportable_facts=["chat is active"],
        relation_bias_strength=0.34,
        related_person_ids=["audience"],
        partner_address_hint="audience",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "advance",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.58,
            "coherence_score": 0.56,
            "human_presence_signal": 0.7,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.04,
            future_loop_pull=0.22,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "social_topology": "threaded_group",
            "privacy_level": 0.16,
            "norm_pressure": 0.34,
            "environmental_load": 0.18,
            "scene_family": "co_present",
            "scene_tags": ["live_chat", "visible_comments"],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.68,
            "reportable_slice": ["chat is active"],
            "actionable_slice": ["pick up a visible comment"],
            "reportability_gate": {"gate_mode": "open"},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "medium",
                "acknowledgement_need_level": "medium",
                "pressure_sensitivity_level": "medium",
                "next_step_room_level": "medium",
            },
        },
        qualia_planner_view={
            "trust": 0.72,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.16,
            "body_load": 0.04,
            "protection_bias": 0.08,
            "felt_energy": 0.2,
        },
        terrain_readout={
            "value": 0.16,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.54,
            "avoid_bias": 0.1,
            "protect_bias": 0.14,
            "active_patch_index": 1,
            "active_patch_label": "visible_threshold",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.18,
            "reasons": ["stable_contact"],
        },
        self_state={
            "mode": "streaming",
            "talk_mode": "ask",
            "related_person_id": "audience",
            "stress": 0.14,
            "recovery_need": 0.08,
            "recent_strain": 0.08,
            "safety_bias": 0.1,
            "continuity_score": 0.62,
            "social_grounding": 0.68,
            "trust_memory": 0.7,
            "familiarity": 0.64,
            "attachment": 0.62,
        },
    )

    assert packet["live_engagement_state"]["state"] == "pickup_comment"
    assert packet["reaction_vs_overnight_bias"]["same_turn"]["live_primary_move"] == "pick_up_comment"


def test_policy_packet_relaxes_soft_containment_when_bright_room_is_open() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="comment:latest",
        current_risks=[],
        reportable_facts=["chat is active"],
        relation_bias_strength=0.42,
        related_person_ids=["audience"],
        partner_address_hint="audience",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "contain",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.66,
            "coherence_score": 0.62,
            "human_presence_signal": 0.76,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "social_topology": "threaded_group",
            "privacy_level": 0.14,
            "norm_pressure": 0.26,
            "environmental_load": 0.16,
            "scene_family": "co_present",
            "scene_tags": ["live_chat", "visible_comments"],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.7,
            "reportable_slice": ["chat is active"],
            "actionable_slice": ["pick up a visible comment"],
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.76,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.18,
            "body_load": 0.03,
            "protection_bias": 0.1,
            "felt_energy": 0.24,
        },
        terrain_readout={
            "value": 0.18,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.58,
            "avoid_bias": 0.08,
            "protect_bias": 0.22,
            "active_patch_index": 1,
            "active_patch_label": "visible_threshold",
        },
        protection_mode={
            "mode": "contain",
            "strength": 0.34,
            "winner_margin": 0.08,
            "reasons": ["keep_thread_soft"],
        },
        self_state={
            "mode": "streaming",
            "talk_mode": "ask",
            "related_person_id": "audience",
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.06,
            "safety_bias": 0.08,
            "continuity_score": 0.68,
            "social_grounding": 0.72,
            "trust_memory": 0.74,
            "familiarity": 0.68,
            "attachment": 0.66,
            "exploration_bias": 0.72,
            "caution_bias": 0.18,
            "person_registry_snapshot": {
                "persons": {
                    "audience": {
                        "adaptive_traits": {
                            "style_warmth_memory": 0.68,
                            "playful_ceiling": 0.74,
                            "advice_tolerance": 0.58,
                            "lexical_familiarity": 0.62,
                        }
                    }
                }
            },
        },
    )

    assert packet["live_engagement_state"]["state"] == "pickup_comment"
    assert packet["lightness_budget_state"]["state"] in {"open_play", "warm_only", "light_ok"}
    assert packet["response_strategy"] in {"attune_then_extend", "shared_world_next_step"}
    assert packet["response_strategy"] != "contain_then_stabilize"
    assert packet["bright_strategy_override"]["applied"] is True
    assert packet["listener_action_state"]["state"] in {"soft_ack", "playful_ack", "warm_laugh_ack"}


def test_policy_packet_uses_shared_world_next_step_for_shared_smile_even_with_softer_contact_scores() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="comment:latest",
        current_risks=[],
        reportable_facts=["chat is active"],
        relation_bias_strength=0.42,
        related_person_ids=["audience"],
        partner_address_hint="audience",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.46,
            "coherence_score": 0.44,
            "human_presence_signal": 0.44,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "social_topology": "threaded_group",
            "privacy_level": 0.14,
            "norm_pressure": 0.26,
            "environmental_load": 0.16,
            "scene_family": "ambient",
            "scene_tags": ["live_chat", "visible_comments"],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.7,
            "reportable_slice": ["chat is active"],
            "actionable_slice": ["pick up a visible comment"],
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.76,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.18,
            "body_load": 0.03,
            "protection_bias": 0.1,
            "felt_energy": 0.24,
        },
        terrain_readout={
            "value": 0.18,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.58,
            "avoid_bias": 0.08,
            "protect_bias": 0.12,
            "active_patch_index": 1,
            "active_patch_label": "visible_threshold",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.18,
            "winner_margin": 0.08,
            "reasons": ["keep_thread_soft"],
        },
        self_state={
            "mode": "streaming",
            "talk_mode": "ask",
            "related_person_id": "audience",
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "continuity_score": 0.68,
            "social_grounding": 0.72,
            "trust_memory": 0.74,
            "familiarity": 0.68,
            "attachment": 0.66,
            "exploration_bias": 0.72,
            "caution_bias": 0.18,
            "surface_user_text": "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.66,
            },
            "organism_state": {
                "dominant_posture": "play",
                "play_window": 0.44,
                "expressive_readiness": 0.58,
                "attunement": 0.62,
            },
            "person_registry_snapshot": {
                "persons": {
                    "audience": {
                        "adaptive_traits": {
                            "style_warmth_memory": 0.68,
                            "playful_ceiling": 0.74,
                            "advice_tolerance": 0.58,
                            "lexical_familiarity": 0.62,
                        }
                    }
                }
            },
        },
    )

    assert packet["shared_moment_state"]["state"] == "shared_moment"
    assert packet["utterance_reason_packet"]["offer"] == "brief_shared_smile"
    assert packet["bright_strategy_override"]["applied"] is True
    assert packet["response_strategy"] == "shared_world_next_step"


def test_field_strategy_override_promotes_shared_world_next_step_for_reentry_field() -> None:
    override = _derive_field_strategy_override(
        response_strategy="attune_then_extend",
        opening_move="stay_with_visible",
        followup_move="invite_visible_state",
        closing_move="hold_space",
        current_risks=[],
        clarify_context=False,
        clarify_opening_context=False,
        external_field_state={
            "dominant_field": "continuity_field",
            "continuity_pull": 0.74,
            "social_pressure": 0.18,
            "safety_envelope": 0.24,
        },
        terrain_dynamics_state={
            "dominant_basin": "recovery_basin",
            "dominant_flow": "reenter",
            "terrain_energy": 0.42,
            "entropy": 0.26,
            "ignition_pressure": 0.22,
            "barrier_height": 0.2,
            "recovery_gradient": 0.72,
            "basin_pull": 0.58,
        },
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.72,
            "afterglow": 0.66,
        },
        utterance_reason_packet={
            "state": "active",
            "offer": "brief_shared_smile",
            "question_policy": "none",
        },
        live_engagement_state={
            "state": "riff_with_comment",
            "primary_move": "riff_current_comment",
            "score": 0.58,
        },
        joint_state={
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.62,
            "shared_tension": 0.16,
            "common_ground": 0.66,
            "joint_attention": 0.68,
            "mutual_room": 0.52,
            "coupling_strength": 0.6,
        },
        organism_state={
            "dominant_posture": "play",
            "play_window": 0.44,
            "expressive_readiness": 0.58,
            "attunement": 0.62,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.12},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
        situation_risk_state={"state": "ordinary_context"},
        emergency_posture={"dialogue_permission": "allow_short"},
    )

    assert override["applied"] is True
    assert override["reason"] == "field_reentry_progression"
    assert override["response_strategy"] == "shared_world_next_step"
    assert override["joint_mode"] == "delighted_jointness"


def test_field_strategy_override_uses_respectful_wait_under_social_pressure_field() -> None:
    override = _derive_field_strategy_override(
        response_strategy="shared_world_next_step",
        opening_move="synchronize_then_propose",
        followup_move="map_next_step",
        closing_move="keep_pace_mutual",
        current_risks=[],
        clarify_context=False,
        clarify_opening_context=False,
        external_field_state={
            "dominant_field": "social_pressure_field",
            "continuity_pull": 0.34,
            "social_pressure": 0.78,
            "safety_envelope": 0.42,
        },
        terrain_dynamics_state={
            "dominant_basin": "steady_basin",
            "dominant_flow": "settle",
            "terrain_energy": 0.34,
            "entropy": 0.18,
            "ignition_pressure": 0.08,
            "barrier_height": 0.58,
            "recovery_gradient": 0.28,
            "basin_pull": 0.42,
        },
        shared_moment_state={"state": "none"},
        utterance_reason_packet={"state": "inactive", "question_policy": "soft"},
        live_engagement_state={"state": "hold", "primary_move": "hold_presence", "score": 0.18},
        joint_state={
            "dominant_mode": "strained_jointness",
            "shared_delight": 0.14,
            "shared_tension": 0.58,
            "common_ground": 0.32,
            "joint_attention": 0.28,
            "mutual_room": 0.22,
            "coupling_strength": 0.34,
        },
        organism_state={
            "dominant_posture": "attune",
            "play_window": 0.16,
            "expressive_readiness": 0.34,
            "attunement": 0.54,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        situation_risk_state={"state": "ordinary_context"},
        emergency_posture={"dialogue_permission": "allow_short"},
    )

    assert override["applied"] is True
    assert override["reason"] == "field_respectful_constraint"
    assert override["response_strategy"] == "respectful_wait"
    assert override["joint_mode"] == "strained_jointness"


def test_field_strategy_override_promotes_shared_world_next_step_for_relation_reframing_progression() -> None:
    override = _derive_field_strategy_override(
        response_strategy="attune_then_extend",
        opening_move="stay_with_visible",
        followup_move="invite_visible_state",
        closing_move="hold_space",
        current_risks=[],
        clarify_context=False,
        clarify_opening_context=False,
        external_field_state={
            "dominant_field": "continuity_field",
            "continuity_pull": 0.72,
            "social_pressure": 0.22,
            "safety_envelope": 0.26,
        },
        terrain_dynamics_state={
            "dominant_basin": "continuity_basin",
            "dominant_flow": "reenter",
            "terrain_energy": 0.38,
            "entropy": 0.22,
            "ignition_pressure": 0.18,
            "barrier_height": 0.18,
            "recovery_gradient": 0.68,
            "basin_pull": 0.62,
        },
        shared_moment_state={"state": "none"},
        utterance_reason_packet={
            "state": "active",
            "relation_frame": "cross_context_bridge",
            "causal_frame": "reframing_cause",
            "memory_frame": "name_distant_link",
            "preserve": "do_not_overclaim",
            "question_policy": "none",
        },
        live_engagement_state={
            "state": "pickup_comment",
            "primary_move": "pick_up_comment",
            "score": 0.46,
        },
        joint_state={
            "dominant_mode": "shared_attention",
            "shared_delight": 0.24,
            "shared_tension": 0.18,
            "common_ground": 0.62,
            "joint_attention": 0.66,
            "mutual_room": 0.54,
            "coupling_strength": 0.58,
        },
        organism_state={
            "dominant_posture": "attune",
            "play_window": 0.28,
            "expressive_readiness": 0.56,
            "attunement": 0.64,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        situation_risk_state={"state": "ordinary_context"},
        emergency_posture={"dialogue_permission": "allow_short"},
    )

    assert override["applied"] is True
    assert override["reason"] == "field_relation_reentry_progression"
    assert override["response_strategy"] == "shared_world_next_step"
    assert override["relation_frame"] == "cross_context_bridge"
    assert override["causal_frame"] == "reframing_cause"


def test_field_strategy_override_prefers_respectful_wait_for_guarded_relation_field() -> None:
    override = _derive_field_strategy_override(
        response_strategy="shared_world_next_step",
        opening_move="synchronize_then_propose",
        followup_move="map_next_step",
        closing_move="keep_pace_mutual",
        current_risks=[],
        clarify_context=False,
        clarify_opening_context=False,
        external_field_state={
            "dominant_field": "shifting_field",
            "continuity_pull": 0.48,
            "social_pressure": 0.32,
            "safety_envelope": 0.36,
        },
        terrain_dynamics_state={
            "dominant_basin": "steady_basin",
            "dominant_flow": "settle",
            "terrain_energy": 0.3,
            "entropy": 0.54,
            "ignition_pressure": 0.12,
            "barrier_height": 0.58,
            "recovery_gradient": 0.22,
            "basin_pull": 0.34,
        },
        shared_moment_state={"state": "none"},
        utterance_reason_packet={
            "state": "active",
            "relation_frame": "unfinished_link",
            "causal_frame": "unfinished_thread_cause",
            "memory_frame": "keep_unfinished_link_near",
            "preserve": "do_not_overclaim",
            "question_policy": "none",
        },
        live_engagement_state={"state": "hold", "primary_move": "hold_presence", "score": 0.18},
        joint_state={
            "dominant_mode": "strained_jointness",
            "shared_delight": 0.1,
            "shared_tension": 0.48,
            "common_ground": 0.28,
            "joint_attention": 0.3,
            "mutual_room": 0.24,
            "coupling_strength": 0.32,
        },
        organism_state={
            "dominant_posture": "attune",
            "play_window": 0.14,
            "expressive_readiness": 0.34,
            "attunement": 0.52,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        situation_risk_state={"state": "ordinary_context"},
        emergency_posture={"dialogue_permission": "allow_short"},
    )

    assert override["applied"] is True
    assert override["reason"] == "field_relation_guard"
    assert override["response_strategy"] == "respectful_wait"
    assert override["relation_frame"] == "unfinished_link"
    assert override["causal_frame"] == "unfinished_thread_cause"


def test_policy_packet_can_open_bright_strategy_from_lightness_playful_ceiling() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="comment:latest",
        current_risks=[],
        reportable_facts=["chat is active"],
        relation_bias_strength=0.36,
        related_person_ids=["audience"],
        partner_address_hint="audience",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.48,
            "coherence_score": 0.45,
            "human_presence_signal": 0.45,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.05,
            future_loop_pull=0.14,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "social_topology": "threaded_group",
            "privacy_level": 0.12,
            "norm_pressure": 0.22,
            "environmental_load": 0.14,
            "scene_family": "ambient",
            "scene_tags": ["live_chat", "visible_comments"],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.68,
            "reportable_slice": ["chat is active"],
            "actionable_slice": ["pick up a visible comment"],
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.74,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.16,
            "body_load": 0.03,
            "protection_bias": 0.08,
            "felt_energy": 0.22,
        },
        terrain_readout={
            "value": 0.16,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.56,
            "avoid_bias": 0.08,
            "protect_bias": 0.1,
            "active_patch_index": 1,
            "active_patch_label": "visible_threshold",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.16,
            "winner_margin": 0.08,
            "reasons": ["keep_thread_soft"],
        },
        self_state={
            "mode": "streaming",
            "talk_mode": "ask",
            "related_person_id": "audience",
            "stress": 0.1,
            "recovery_need": 0.08,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "continuity_score": 0.66,
            "social_grounding": 0.7,
            "trust_memory": 0.72,
            "familiarity": 0.64,
            "attachment": 0.62,
            "exploration_bias": 0.68,
            "caution_bias": 0.18,
            "surface_user_text": "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.64,
            },
            "organism_state": {
                "dominant_posture": "play",
                "play_window": 0.4,
                "expressive_readiness": 0.54,
                "attunement": 0.6,
            },
        },
    )

    assert packet["lightness_budget_state"]["playful_ceiling"] >= 0.18
    assert packet["bright_strategy_override"]["applied"] is True
    assert packet["response_strategy"] == "shared_world_next_step"


def test_live_engagement_state_biases_posture_and_actuation_for_streaming_pickup() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "attune_then_reflect",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.08},
        "qualia_planner_view": {
            "trust": 0.78,
            "degraded": False,
            "body_load": 0.02,
            "protection_bias": 0.06,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.18},
        "body_recovery_guard": {"state": "open", "score": 0.08},
        "body_homeostasis_state": {"state": "steady", "score": 0.1},
        "homeostasis_budget_state": {"state": "steady", "score": 0.1},
        "initiative_readiness": {"state": "ready", "score": 0.52},
        "commitment_state": {"state": "waver", "target": "hold", "score": 0.18},
        "learning_mode_state": {"state": "observe_only", "score": 0.12, "probe_room": 0.18},
        "social_experiment_loop_state": {"state": "watch_and_read", "score": 0.12, "probe_intensity": 0.16},
        "live_engagement_state": {"state": "pickup_comment", "score": 0.64, "primary_move": "pick_up_comment"},
        "relational_continuity_state": {"state": "holding_thread", "score": 0.42},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "threaded_group", "score": 0.46},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["live_engagement_name"] == "pickup_comment"
    assert "pick_up_comment" in posture["next_action_candidates"]
    assert "answer_visible_comment" in posture["next_action_candidates"]
    assert actuation["live_engagement_name"] == "pickup_comment"
    assert actuation["primary_action"] == "pick_up_comment"
    assert "answer_visible_comment" in actuation["action_queue"]
    assert "return_to_chat" in actuation["action_queue"]


def test_action_posture_and_actuation_follow_shared_smile_riff_even_when_thread_is_held() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "attune_then_extend",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.08},
        "qualia_planner_view": {
            "trust": 0.78,
            "degraded": False,
            "body_load": 0.02,
            "protection_bias": 0.06,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.18},
        "body_recovery_guard": {"state": "open", "score": 0.08},
        "body_homeostasis_state": {"state": "steady", "score": 0.1},
        "homeostasis_budget_state": {"state": "steady", "score": 0.1},
        "initiative_readiness": {"state": "hold", "score": 0.24},
        "commitment_state": {"state": "waver", "target": "hold", "score": 0.18},
        "learning_mode_state": {"state": "observe_only", "score": 0.12, "probe_room": 0.18},
        "social_experiment_loop_state": {"state": "watch_and_read", "score": 0.12, "probe_intensity": 0.16},
        "live_engagement_state": {"state": "riff_with_comment", "score": 0.52, "primary_move": "riff_current_comment"},
        "shared_moment_state": {
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.72,
            "afterglow": 0.64,
        },
        "listener_action_state": {"state": "warm_laugh_ack"},
        "utterance_reason_packet": {
            "state": "active",
            "offer": "brief_shared_smile",
            "question_policy": "none",
        },
        "joint_state": {
            "dominant_mode": "delighted_jointness",
            "shared_delight": 0.66,
            "shared_tension": 0.14,
            "common_ground": 0.64,
            "joint_attention": 0.68,
            "mutual_room": 0.52,
            "coupling_strength": 0.62,
        },
        "relational_continuity_state": {"state": "holding_thread", "score": 0.42},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "ambient", "score": 0.24},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "co_move"
    assert posture["outcome_goal"] == "share_small_shift"
    assert posture["joint_mode"] == "delighted_jointness"
    assert "riff_current_comment" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "shared_progression"
    assert actuation["primary_action"] == "riff_current_comment"
    assert actuation["joint_mode"] == "delighted_jointness"
    assert "weave_light_callback" in actuation["action_queue"]


def test_action_posture_and_actuation_follow_field_reentry_progression() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "shared_world_next_step",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.06},
        "qualia_planner_view": {
            "trust": 0.82,
            "degraded": False,
            "body_load": 0.02,
            "protection_bias": 0.04,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.16},
        "body_recovery_guard": {"state": "open", "score": 0.08},
        "body_homeostasis_state": {"state": "steady", "score": 0.12},
        "homeostasis_budget_state": {"state": "steady", "score": 0.12},
        "live_engagement_state": {"state": "pickup_comment", "score": 0.58, "primary_move": "pick_up_comment"},
        "shared_moment_state": {"state": "shared_moment", "moment_kind": "laugh", "score": 0.62, "jointness": 0.54, "afterglow": 0.42},
        "listener_action_state": {"state": "warm_laugh_ack"},
        "utterance_reason_packet": {"state": "active", "offer": "brief_shared_smile", "question_policy": "none"},
        "joint_state": {
            "dominant_mode": "shared_attention",
            "shared_delight": 0.54,
            "shared_tension": 0.16,
            "common_ground": 0.66,
            "joint_attention": 0.64,
            "mutual_room": 0.48,
            "coupling_strength": 0.58,
        },
        "organism_state": {"dominant_posture": "play", "play_window": 0.62, "relation_pull": 0.58},
        "external_field_state": {
            "dominant_field": "continuity_field",
            "continuity_pull": 0.64,
            "safety_envelope": 0.44,
        },
        "terrain_dynamics_state": {
            "dominant_basin": "recovery_basin",
            "dominant_flow": "reenter",
            "recovery_gradient": 0.56,
            "barrier_height": 0.24,
        },
        "relational_continuity_state": {"state": "holding_thread", "score": 0.46},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "ambient", "score": 0.2},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "co_move"
    assert posture["outcome_goal"] == "shared_progress"
    assert posture["joint_reentry_room"] >= 0.2
    assert "pick_up_comment" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "shared_progression"
    assert actuation["primary_action"] == "pick_up_comment"
    assert actuation["joint_reentry_room"] >= 0.2
    assert "keep_shared_thread" in actuation["action_queue"]


def test_action_posture_and_actuation_respect_field_guarded_constraint() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "respectful_wait",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.22},
        "qualia_planner_view": {
            "trust": 0.68,
            "degraded": False,
            "body_load": 0.04,
            "protection_bias": 0.08,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.22},
        "body_recovery_guard": {"state": "guarded", "score": 0.4},
        "body_homeostasis_state": {"state": "steady", "score": 0.18},
        "homeostasis_budget_state": {"state": "steady", "score": 0.2},
        "live_engagement_state": {"state": "hold_presence", "score": 0.28, "primary_move": "hold_presence"},
        "listener_action_state": {"state": "soft_ack"},
        "utterance_reason_packet": {"state": "active", "offer": "leave_return_point", "question_policy": "none"},
        "joint_state": {
            "dominant_mode": "strained_jointness",
            "shared_delight": 0.12,
            "shared_tension": 0.64,
            "common_ground": 0.28,
            "joint_attention": 0.24,
            "mutual_room": 0.18,
            "coupling_strength": 0.3,
        },
        "organism_state": {"dominant_posture": "protect", "protective_tension": 0.68},
        "external_field_state": {
            "dominant_field": "formal_field",
            "social_pressure": 0.72,
            "continuity_pull": 0.38,
            "safety_envelope": 0.34,
        },
        "terrain_dynamics_state": {
            "dominant_basin": "protective_basin",
            "dominant_flow": "contain",
            "recovery_gradient": 0.26,
            "barrier_height": 0.68,
        },
        "situation_risk_state": {"state": "guarded_context", "immediacy": 0.34},
        "relational_continuity_state": {"state": "holding_thread", "score": 0.44},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "hierarchical", "score": 0.54},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "wait"
    assert posture["boundary_mode"] in {"respectful", "protective"}
    assert posture["joint_guard_signal"] >= 0.3
    assert "leave_return_point" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "defer_with_presence"
    assert actuation["primary_action"] == "hold_presence"
    assert actuation["joint_guard_signal"] >= 0.3
    assert "respect_role_gradient" in actuation["action_queue"]


def test_action_posture_and_actuation_follow_relation_reentry_progression_without_shared_moment() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "shared_world_next_step",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.06},
        "qualia_planner_view": {
            "trust": 0.8,
            "degraded": False,
            "body_load": 0.02,
            "protection_bias": 0.04,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.16},
        "body_recovery_guard": {"state": "open", "score": 0.08},
        "body_homeostasis_state": {"state": "steady", "score": 0.12},
        "homeostasis_budget_state": {"state": "steady", "score": 0.12},
        "live_engagement_state": {"state": "pickup_comment", "score": 0.54, "primary_move": "pick_up_comment"},
        "shared_moment_state": {"state": "none"},
        "listener_action_state": {"state": "soft_ack"},
        "utterance_reason_packet": {
            "state": "active",
            "offer": "brief_ack",
            "question_policy": "none",
            "relation_frame": "cross_context_bridge",
            "causal_frame": "reframing_cause",
            "memory_frame": "name_distant_link",
            "preserve": "do_not_overclaim",
        },
        "joint_state": {
            "dominant_mode": "shared_attention",
            "shared_delight": 0.24,
            "shared_tension": 0.16,
            "common_ground": 0.68,
            "joint_attention": 0.64,
            "mutual_room": 0.52,
            "coupling_strength": 0.58,
        },
        "organism_state": {"dominant_posture": "attune", "play_window": 0.36, "relation_pull": 0.58},
        "external_field_state": {
            "dominant_field": "continuity_field",
            "continuity_pull": 0.66,
            "safety_envelope": 0.34,
        },
        "terrain_dynamics_state": {
            "dominant_basin": "continuity_basin",
            "dominant_flow": "reenter",
            "recovery_gradient": 0.58,
            "barrier_height": 0.22,
        },
        "relational_continuity_state": {"state": "holding_thread", "score": 0.44},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "ambient", "score": 0.2},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "co_move"
    assert posture["outcome_goal"] == "shared_progress"
    assert "keep_history_bridge_visible" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "shared_progression"
    assert actuation["primary_action"] == "pick_up_comment"
    assert "keep_history_bridge_visible" in actuation["action_queue"]


def test_action_posture_and_actuation_hold_guarded_relation_without_social_pressure_field() -> None:
    packet = {
        "conversation_contract": {"response_action_now": {}},
        "response_strategy": "respectful_wait",
        "constraint_field": {},
        "conscious_workspace": {},
        "terrain_readout": {"protect_bias": 0.18},
        "qualia_planner_view": {
            "trust": 0.72,
            "degraded": False,
            "body_load": 0.04,
            "protection_bias": 0.08,
        },
        "protection_mode": {"mode": "monitor", "strength": 0.18},
        "body_recovery_guard": {"state": "open", "score": 0.18},
        "body_homeostasis_state": {"state": "steady", "score": 0.16},
        "homeostasis_budget_state": {"state": "steady", "score": 0.18},
        "live_engagement_state": {"state": "hold_presence", "score": 0.22, "primary_move": "hold_presence"},
        "shared_moment_state": {"state": "none"},
        "listener_action_state": {"state": "soft_ack"},
        "utterance_reason_packet": {
            "state": "active",
            "offer": "leave_return_point",
            "question_policy": "none",
            "relation_frame": "unfinished_link",
            "causal_frame": "unfinished_thread_cause",
            "memory_frame": "keep_unfinished_link_near",
            "preserve": "do_not_overclaim",
        },
        "joint_state": {
            "dominant_mode": "strained_jointness",
            "shared_delight": 0.1,
            "shared_tension": 0.46,
            "common_ground": 0.26,
            "joint_attention": 0.24,
            "mutual_room": 0.22,
            "coupling_strength": 0.28,
        },
        "organism_state": {"dominant_posture": "attune", "protective_tension": 0.42},
        "external_field_state": {
            "dominant_field": "shifting_field",
            "social_pressure": 0.28,
            "continuity_pull": 0.46,
            "safety_envelope": 0.4,
        },
        "terrain_dynamics_state": {
            "dominant_basin": "steady_basin",
            "dominant_flow": "settle",
            "recovery_gradient": 0.2,
            "barrier_height": 0.58,
            "entropy": 0.54,
        },
        "situation_risk_state": {"state": "ordinary_context", "immediacy": 0.18},
        "relational_continuity_state": {"state": "holding_thread", "score": 0.44},
        "relation_competition_state": {"state": "ambient", "competition_level": 0.0},
        "social_topology_state": {"state": "ambient", "score": 0.24},
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "wait"
    assert "protect_unfinished_link" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "defer_with_presence"
    assert actuation["primary_action"] == "hold_presence"
    assert "protect_unfinished_link" in actuation["action_queue"]
