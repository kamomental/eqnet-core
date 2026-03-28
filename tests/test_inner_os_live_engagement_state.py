from types import SimpleNamespace

from inner_os.action_posture import derive_action_posture
from inner_os.actuation_plan import derive_actuation_plan
from inner_os.live_engagement_state import derive_live_engagement_state
from inner_os.policy_packet import derive_interaction_policy_packet


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
