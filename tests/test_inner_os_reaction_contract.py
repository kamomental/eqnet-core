from inner_os.expression.reaction_contract import derive_reaction_contract


def test_derive_reaction_contract_keeps_small_shared_smile_small_and_joined() -> None:
    contract = derive_reaction_contract(
        interaction_policy={
            "response_strategy": "shared_world_next_step",
            "conversation_contract": {"response_action_now": {"question_budget": 1}},
            "recent_dialogue_state": {"state": "continuing_thread"},
        },
        action_posture={
            "boundary_mode": "soft_hold",
            "question_budget": 1,
            "social_topology_name": "one_to_one",
        },
        actuation_plan={
            "execution_mode": "shared_progression",
            "response_channel": "speak",
            "wait_before_action": "",
        },
        discourse_shape={"shape_id": "bright_bounce", "question_budget": 0},
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "constraints": {"max_questions": 1},
            "surface_profile": {"response_length": "short"},
            "source_state": {
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "joint_common_ground": 0.67,
                "organism_social_mode": "near",
            },
        },
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
    )

    assert contract["scale"] == "small"
    assert contract["stance"] == "join"
    assert contract["initiative"] == "co_move"
    assert contract["question_budget"] == 0
    assert contract["interpretation_budget"] == "none"
    assert contract["response_channel"] == "speak"
    assert contract["continuity_mode"] == "continue"
    assert contract["distance_mode"] == "near"
    assert contract["closure_mode"] == "open_light"


def test_derive_reaction_contract_marks_guarded_wait_as_hold() -> None:
    contract = derive_reaction_contract(
        interaction_policy={
            "response_strategy": "respectful_wait",
            "conversation_contract": {"response_action_now": {"question_budget": 0}},
        },
        action_posture={
            "boundary_mode": "contain",
            "social_topology_name": "hierarchical",
        },
        actuation_plan={
            "execution_mode": "defer_with_presence",
            "response_channel": "hold",
            "wait_before_action": "brief",
        },
        discourse_shape={"shape_id": "reflect_hold"},
        surface_context_packet={
            "conversation_phase": "reopening_thread",
            "constraints": {"max_questions": 0, "prefer_return_point": True},
            "surface_profile": {"response_length": "short"},
            "source_state": {
                "recent_dialogue_state": "reopening_thread",
                "organism_protective_tension": 0.62,
                "external_field_social_pressure": 0.28,
            },
        },
        turn_delta={"kind": "issue_pause", "preferred_act": "leave_return_point_from_anchor"},
    )

    assert contract["stance"] == "hold"
    assert contract["initiative"] == "yield"
    assert contract["timing_mode"] == "held_open"
    assert contract["continuity_mode"] == "reopen"
    assert contract["distance_mode"] == "guarded"
    assert contract["closure_mode"] == "leave_open"


def test_derive_reaction_contract_uses_shared_presence_for_joined_near_stance() -> None:
    contract = derive_reaction_contract(
        interaction_policy={
            "response_strategy": "attuned_receive",
            "conversation_contract": {"response_action_now": {"question_budget": 0}},
        },
        action_posture={
            "boundary_mode": "soft_hold",
            "social_topology_name": "one_to_one",
        },
        actuation_plan={
            "execution_mode": "attuned_contact",
            "response_channel": "speak",
            "wait_before_action": "",
        },
        discourse_shape={"shape_id": "reflect_hold"},
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "source_state": {
                "shared_presence_mode": "inhabited_shared_space",
                "shared_presence_co_presence": 0.72,
                "shared_presence_boundary_stability": 0.64,
                "self_other_dominant_attribution": "shared",
                "self_other_unknown_likelihood": 0.12,
                "subjective_scene_shared_scene_potential": 0.68,
                "subjective_scene_familiarity": 0.58,
                "subjective_scene_anchor_frame": "shared_margin",
            },
        },
        turn_delta={"kind": "scene_companioning"},
    )

    assert contract["stance"] == "join"
    assert contract["initiative"] == "co_move"
    assert contract["distance_mode"] == "near"
    assert contract["continuity_mode"] == "continue"


def test_derive_reaction_contract_uses_uncertain_self_view_for_guarded_hold() -> None:
    contract = derive_reaction_contract(
        interaction_policy={
            "response_strategy": "attuned_receive",
            "conversation_contract": {"response_action_now": {"question_budget": 1}},
        },
        action_posture={
            "boundary_mode": "soft_hold",
            "social_topology_name": "one_to_one",
        },
        actuation_plan={
            "execution_mode": "attuned_contact",
            "response_channel": "speak",
            "wait_before_action": "",
        },
        discourse_shape={"shape_id": "reflect_hold"},
        surface_context_packet={
            "conversation_phase": "fresh_opening",
            "source_state": {
                "shared_presence_mode": "guarded_boundary",
                "shared_presence_co_presence": 0.24,
                "shared_presence_boundary_stability": 0.18,
                "self_other_dominant_attribution": "unknown",
                "self_other_unknown_likelihood": 0.76,
                "subjective_scene_shared_scene_potential": 0.22,
                "subjective_scene_familiarity": 0.12,
                "subjective_scene_anchor_frame": "ambient_margin",
            },
        },
        turn_delta={"kind": "uncertain_scene"},
    )

    assert contract["stance"] == "hold"
    assert contract["initiative"] == "yield"
    assert contract["distance_mode"] == "guarded"
    assert contract["interpretation_budget"] == "low"
