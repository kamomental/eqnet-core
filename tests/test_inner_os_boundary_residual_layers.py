from inner_os.boundary_transformer import derive_boundary_transform_result
from inner_os.expression import DialogueContext, render_response
from inner_os.residual_reflector import derive_residual_reflection
from inner_os.access.models import ForegroundState


def test_boundary_transformer_marks_probing_acts_under_withhold_gate() -> None:
    transform = derive_boundary_transform_result(
        content_sequence=[
            {"act": "respect_boundary", "text": "We do not have to press this right now."},
            {"act": "clarify_question", "text": "What exactly happened there?"},
            {"act": "offer_next_step", "text": "Here is the next step."},
        ],
        interaction_constraints={
            "allow_small_next_step": False,
        },
        conversation_contract={
            "leave_closed_for_now": ["unfinished part"],
        },
        constraint_field={
            "reportability_limit": "withhold",
            "do_not_cross": ["force_reportability"],
        },
        reportability_gate={
            "gate_mode": "withhold",
        },
        current_risks=[],
    )

    assert transform.gate_mode == "withhold"
    assert "clarify_question" in transform.withheld_acts
    assert "offer_next_step" in transform.withheld_acts
    assert "unfinished part" in transform.deferred_topics
    assert transform.transformation_mode == "withhold"
    assert transform.residual_pressure > 0.0


def test_residual_reflector_preserves_unsaid_and_deferred_signals() -> None:
    reflection = derive_residual_reflection(
        boundary_transform={
            "withheld_acts": ["clarify_question"],
            "softened_acts": ["offer_small_opening_line"],
            "deferred_topics": ["unfinished part"],
            "residual_pressure": 0.62,
        },
        conversation_contract={},
    )

    assert reflection.mode == "withheld"
    assert reflection.focus == "unfinished part"
    assert "clarify_question" in reflection.withheld_acts
    assert "offer_small_opening_line" in reflection.softened_acts
    assert "deferred_topic" in reflection.reason_tokens
    assert reflection.strength >= 0.62


def test_response_planner_exposes_boundary_transform_and_residual_reflection() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["shared opening"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.8},
        affective_summary={
            "arousal": 0.18,
            "social_tension": 0.14,
            "stress": 0.12,
            "recovery_need": 0.1,
            "safety_bias": 0.08,
        },
    )

    plan = render_response(
        foreground,
        DialogueContext(
            user_text="少し話したいけど、まだ整理できていない",
        ),
    )

    assert "boundary_transform" in plan.llm_payload
    assert "residual_reflection" in plan.llm_payload
    assert plan.boundary_transform["gate_mode"] in {"open", "narrow", "withhold"}
    assert plan.residual_reflection["mode"] in {
        "none",
        "boundary_tension",
        "held_open",
        "softened",
        "withheld",
    }
    assert isinstance(plan.boundary_transform["candidate_decisions"], list)
