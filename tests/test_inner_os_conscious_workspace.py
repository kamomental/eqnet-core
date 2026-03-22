from inner_os.affect_blend import AffectBlendState
from inner_os.conscious_workspace import ignite_conscious_workspace
from inner_os.constraint_field import derive_constraint_field
from inner_os.scene_state import derive_scene_state


def test_conscious_workspace_can_guard_reportability_under_pressure() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.2,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.72,
        safety_margin=0.24,
        environmental_load=0.66,
        current_risks=["danger"],
        active_goals=["repair"],
    )
    affect_blend = AffectBlendState(
        care=0.64,
        reverence=0.48,
        innocence=0.12,
        defense=0.78,
        future_pull=0.18,
        shared_world_pull=0.16,
        distress=0.72,
        confidence=0.18,
        conflict_level=0.58,
        residual_tension=0.62,
        reportability_pressure=0.68,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.74,
        recovery_need=0.58,
        safety_bias=0.76,
        recent_strain=0.62,
        current_risks=["danger"],
    )
    workspace = ignite_conscious_workspace(
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        current_focus="person:user",
        reportable_facts=["I want to say more here."],
        current_risks=["danger"],
        related_person_ids=["user"],
        interaction_option_candidates=[{"family_id": "repair", "relative_weight": 0.64}],
        memory_anchor="harbor slope",
        scene_state=scene_state.__dict__,
    )

    assert constraint_field.reportability_limit == "withhold"
    assert "contain" in constraint_field.admissible_families
    assert workspace.workspace_mode == "guarded_foreground"
    assert workspace.ignition_phase == "guarded"
    assert workspace.reportability_gate["gate_mode"] == "withhold"
    assert workspace.withheld_slice
    assert workspace.actionable_slice
    assert "person:user" in workspace.withheld_slice or "I want to say more here." in workspace.withheld_slice
    assert workspace.ignition_score > 0.0
    assert workspace.slot_scores
    assert workspace.winner_margin >= 0.0
    assert workspace.dominant_inputs
