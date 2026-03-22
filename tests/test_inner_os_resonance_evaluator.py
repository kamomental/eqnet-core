from __future__ import annotations

from inner_os.affect_blend import AffectBlendState
from inner_os.conscious_workspace import ConsciousWorkspace
from inner_os.constraint_field import ConstraintField
from inner_os.interaction_option_search import InteractionOptionCandidate
from inner_os.resonance_evaluator import (
    evaluate_interaction_resonance,
    rerank_interaction_option_candidates,
)
from inner_os.scene_state import SceneState


def test_resonance_evaluator_prefers_low_pressure_candidate_when_detail_room_is_low() -> None:
    scene_state = SceneState(
        privacy_level=0.34,
        norm_pressure=0.68,
        safety_margin=0.28,
        environmental_load=0.66,
        scene_family="repair_window",
        scene_tags=("high_norm", "high_load"),
    )
    affect_blend = AffectBlendState(
        care=0.74,
        defense=0.68,
        distress=0.62,
        conflict_level=0.58,
        residual_tension=0.52,
        reportability_pressure=0.64,
        dominant_mode="care",
    )
    constraint_field = ConstraintField(
        body_cost=0.76,
        boundary_pressure=0.72,
        repair_pressure=0.7,
        shared_world_pressure=0.18,
        protective_bias=0.78,
        disclosure_limit="minimal",
        reportability_limit="withhold",
    )
    workspace = ConsciousWorkspace(
        workspace_mode="guarded_foreground",
        workspace_stability=0.42,
        withheld_slice=("person:user", "care"),
        actionable_slice=("care",),
    )
    candidates = [
        InteractionOptionCandidate(
            family_id="clarify",
            option_id="clarify:repair_window",
            relative_weight=0.38,
            disclosure_depth="light",
            timing_mode="narrow_check",
            boundary_mode="careful_precision",
            next_moves=("narrow_question",),
            rationale=(),
        ),
        InteractionOptionCandidate(
            family_id="wait",
            option_id="wait:repair_window",
            relative_weight=0.3,
            disclosure_depth="minimal",
            timing_mode="deferred_return",
            boundary_mode="respectful",
            next_moves=("hold_presence",),
            rationale=(),
        ),
        InteractionOptionCandidate(
            family_id="repair",
            option_id="repair:repair_window",
            relative_weight=0.32,
            disclosure_depth="minimal",
            timing_mode="slow_reentry",
            boundary_mode="softened",
            next_moves=("name_overreach",),
            rationale=(),
        ),
    ]

    evaluation = evaluate_interaction_resonance(
        scene_state=scene_state,
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        conscious_workspace=workspace,
        interaction_option_candidates=candidates,
    )
    ranked = rerank_interaction_option_candidates(
        interaction_option_candidates=candidates,
        resonance_evaluation=evaluation,
    )

    assert evaluation.estimated_other_person_state.detail_room_level == "low"
    assert evaluation.estimated_other_person_state.pressure_sensitivity_level == "high"
    assert evaluation.recommended_family_id in {"wait", "repair"}
    assert ranked[0].family_id in {"wait", "repair"}
    assert "press_for_detail" in evaluation.avoid_actions
    assert "leave_talking_room" in evaluation.prioritize_actions


def test_resonance_evaluator_can_prefer_small_next_step_when_room_is_open() -> None:
    scene_state = SceneState(
        privacy_level=0.74,
        norm_pressure=0.22,
        safety_margin=0.82,
        environmental_load=0.16,
        scene_family="shared_world",
        scene_tags=("private", "task:coordination"),
    )
    affect_blend = AffectBlendState(
        care=0.58,
        future_pull=0.76,
        shared_world_pull=0.82,
        confidence=0.72,
        conflict_level=0.18,
        residual_tension=0.16,
        reportability_pressure=0.42,
        dominant_mode="shared_world_pull",
    )
    constraint_field = ConstraintField(
        body_cost=0.14,
        boundary_pressure=0.18,
        repair_pressure=0.1,
        shared_world_pressure=0.84,
        protective_bias=0.18,
        disclosure_limit="medium",
        reportability_limit="open",
    )
    workspace = ConsciousWorkspace(
        workspace_mode="foreground",
        workspace_stability=0.72,
        reportable_slice=("next_step",),
        actionable_slice=("next_step",),
    )
    candidates = [
        InteractionOptionCandidate(
            family_id="attune",
            option_id="attune:shared_world",
            relative_weight=0.34,
            disclosure_depth="light",
            timing_mode="soft_start",
            boundary_mode="permeable",
            next_moves=("stay_visible",),
            rationale=(),
        ),
        InteractionOptionCandidate(
            family_id="co_move",
            option_id="co_move:shared_world",
            relative_weight=0.33,
            disclosure_depth="medium",
            timing_mode="step_forward",
            boundary_mode="forward_open",
            next_moves=("map_next_step",),
            rationale=(),
        ),
        InteractionOptionCandidate(
            family_id="wait",
            option_id="wait:shared_world",
            relative_weight=0.33,
            disclosure_depth="minimal",
            timing_mode="deferred_return",
            boundary_mode="respectful",
            next_moves=("hold_presence",),
            rationale=(),
        ),
    ]

    evaluation = evaluate_interaction_resonance(
        scene_state=scene_state,
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        conscious_workspace=workspace,
        interaction_option_candidates=candidates,
    )
    ranked = rerank_interaction_option_candidates(
        interaction_option_candidates=candidates,
        resonance_evaluation=evaluation,
    )

    assert evaluation.estimated_other_person_state.next_step_room_level == "high"
    assert evaluation.recommended_family_id == "co_move"
    assert ranked[0].family_id == "co_move"
    assert "offer_small_next_step" in evaluation.prioritize_actions
    assert "keep_next_turn_open" in evaluation.expected_effects
