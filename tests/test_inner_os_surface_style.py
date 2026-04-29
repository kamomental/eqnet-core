from inner_os.expression.llm_expression_bridge import build_llm_expression_request
from inner_os.expression.surface_style import derive_surface_style_decision


def test_surface_style_stays_plain_when_restraint_is_high() -> None:
    decision = derive_surface_style_decision(
        audit_projection={
            "audit_axes": {
                "culture_politeness_pressure": 0.7,
                "joint_common_ground": 0.8,
                "culture_joke_ratio_ceiling": 0.8,
            }
        }
    )

    assert decision.style_id == "plain"
    assert decision.reason == "restraint_overrides_style"


def test_surface_style_uses_warm_playful_for_shared_interest_and_lightness() -> None:
    decision = derive_surface_style_decision(
        audit_projection={
            "expression_context_state": {
                "culture": {
                    "shared_interest_affinity": 0.9,
                    "joke_ratio_ceiling": 0.8,
                    "politeness_pressure": 0.1,
                },
                "relational_style": {
                    "playful_ceiling": 0.8,
                },
            },
            "surface_context_source_state": {
                "utterance_reason_offer": "brief_shared_smile",
            },
        },
        joint_state={
            "common_ground": 0.8,
            "shared_delight": 0.8,
        },
        shared_presence={
            "co_presence": 0.8,
        },
    )

    assert decision.style_id == "warm_playful"
    assert decision.reason == "shared_warmth_and_playfulness"


def test_llm_expression_bridge_derives_surface_style_from_culture_context() -> None:
    request = build_llm_expression_request(
        input_text="さっきの続きなんだけど、あのあと少し笑えることもあって。",
        reaction_contract={
            "response_channel": "speak",
            "scale": "small",
            "question_budget": 0,
            "interpretation_budget": "none",
            "shape_id": "bright_bounce",
        },
        joint_state={
            "common_ground": 0.8,
            "shared_delight": 0.8,
        },
        shared_presence={
            "co_presence": 0.8,
        },
        audit_projection={
            "expression_context_state": {
                "culture": {
                    "shared_interest_affinity": 0.9,
                    "joke_ratio_ceiling": 0.8,
                    "politeness_pressure": 0.1,
                },
                "relational_style": {
                    "playful_ceiling": 0.8,
                },
            },
            "surface_context_source_state": {
                "utterance_reason_offer": "brief_shared_smile",
            },
        },
    )

    assert request.surface_policy["surface_style"] == "warm_playful"
    assert request.surface_policy["surface_style_reason"] == "shared_warmth_and_playfulness"
