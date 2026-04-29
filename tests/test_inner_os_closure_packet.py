from inner_os.closure_packet import (
    ClosurePacket,
    coerce_closure_packet,
    derive_closure_packet,
)
from inner_os.expression.reaction_contract import derive_reaction_contract
from inner_os.evaluation.closure_packet_eval import (
    ClosurePacketExpectation,
    evaluate_closure_packet_against_expectation,
)


def test_closure_packet_projects_shared_anchor_without_deciding_contract() -> None:
    packet = derive_closure_packet(
        memory_dynamics_state={
            "dominant_link_key": "laugh|shared_margin",
            "dominant_relation_type": "same_anchor",
            "dominant_causal_type": "enabled_by",
            "monument_salience": 0.36,
            "activation_confidence": 0.42,
            "memory_tension": 0.18,
        },
        shared_presence_state={
            "dominant_mode": "inhabited_shared_space",
            "co_presence": 0.67,
            "boundary_stability": 0.61,
            "shared_scene_salience": 0.62,
        },
        joint_state={
            "common_ground": 0.64,
            "repair_readiness": 0.36,
            "shared_tension": 0.18,
        },
    )

    assert "laugh|shared_margin" in packet.dominant_basis_keys
    assert "shared_anchor" in packet.generated_affordances
    assert "gentle_join" in packet.generated_affordances
    assert packet.contract_bias["stance_bias"] == "join"
    assert "closure:shared_anchor" in packet.reason_tags()


def test_closure_packet_marks_reconstruction_risk_as_constraint_bias() -> None:
    packet = derive_closure_packet(
        memory_dynamics_state={
            "dominant_relation_type": "unfinished_carry",
            "dominant_causal_type": "reopened_by",
            "memory_tension": 0.72,
            "activation_confidence": 0.12,
            "forgetting_pressure": 0.58,
        },
        self_other_attribution_state={
            "dominant_attribution": "unknown",
            "unknown_likelihood": 0.74,
        },
        shared_presence_state={
            "dominant_mode": "guarded_boundary",
            "co_presence": 0.18,
            "boundary_stability": 0.2,
        },
        joint_state={
            "common_ground": 0.18,
            "shared_tension": 0.61,
        },
    )

    assert "do_not_overinterpret" in packet.generated_constraints
    assert "leave_return_point" in packet.generated_constraints
    assert "preserve_boundary" in packet.generated_constraints
    assert "unknown_attribution" in packet.inhibition_reasons
    assert packet.contract_bias["interpretation_budget_bias"] == "none"
    assert packet.contract_bias["closure_mode_bias"] == "leave_open"
    assert packet.contract_bias["response_channel_bias"] == "hold"


def test_coerce_closure_packet_preserves_mapping_boundary() -> None:
    packet = coerce_closure_packet(
        {
            "dominant_basis_keys": ["anchor", "anchor", "thread"],
            "generated_constraints": ["do_not_overinterpret"],
            "basis_confidence": 1.5,
            "closure_tension": -0.2,
            "reconstruction_risk": 0.61,
            "contract_bias": {"interpretation_budget_bias": "none"},
        }
    )

    assert isinstance(packet, ClosurePacket)
    assert packet.dominant_basis_keys == ("anchor", "thread")
    assert packet.basis_confidence == 1.0
    assert packet.closure_tension == 0.0
    assert packet.to_dict()["contract_bias"]["interpretation_budget_bias"] == "none"


def test_reaction_contract_accepts_closure_packet_as_reason_tags_only() -> None:
    contract = derive_reaction_contract(
        interaction_policy={
            "response_strategy": "shared_world_next_step",
            "conversation_contract": {"response_action_now": {"question_budget": 0}},
        },
        action_posture={
            "boundary_mode": "soft_hold",
            "question_budget": 0,
            "social_topology_name": "one_to_one",
        },
        actuation_plan={
            "execution_mode": "shared_progression",
            "response_channel": "speak",
        },
        discourse_shape={"shape_id": "bright_bounce", "question_budget": 0},
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "surface_profile": {"response_length": "short"},
            "closure_packet": {
                "generated_constraints": ["do_not_overinterpret"],
                "inhibition_reasons": ["reconstruction_risk"],
            },
            "source_state": {
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "joint_common_ground": 0.62,
                "organism_social_mode": "near",
            },
        },
    )

    assert contract["response_channel"] == "speak"
    assert contract["interpretation_budget"] == "none"
    assert "closure:do_not_overinterpret" in contract["reason_tags"]
    assert "closure:reconstruction_risk" in contract["reason_tags"]


def test_closure_packet_eval_hook_checks_constraints_and_bias() -> None:
    packet = derive_closure_packet(
        memory_dynamics_state={
            "dominant_relation_type": "unfinished_carry",
            "memory_tension": 0.72,
            "activation_confidence": 0.12,
        },
        self_other_attribution_state={"unknown_likelihood": 0.7},
        shared_presence_state={"boundary_stability": 0.22},
        joint_state={"shared_tension": 0.62},
    )

    result = evaluate_closure_packet_against_expectation(
        closure_packet=packet.to_dict(),
        expectation=ClosurePacketExpectation(
            scenario_name="guarded_closure",
            required_constraints=("do_not_overinterpret", "leave_return_point"),
            required_inhibitions=("unknown_attribution",),
            required_contract_bias={"interpretation_budget_bias": "none"},
        ),
    )

    assert result.passed is True
    assert result.to_dict()["score"] == 1.0
