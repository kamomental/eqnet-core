from inner_os.access_projection import project_access_regions
from inner_os.affect_blend import AffectBlendState
from inner_os.contact_dynamics import advance_contact_dynamics
from inner_os.contact_field import derive_contact_field
from inner_os.constraint_field import derive_constraint_field
from inner_os.scene_state import derive_scene_state


def test_contact_dynamics_can_reenter_previous_contact() -> None:
    scene_state = derive_scene_state(
        place_mode="relational_private",
        privacy_level=0.64,
        social_topology="one_to_one",
        task_phase="ongoing",
        temporal_phase="ongoing",
        norm_pressure=0.28,
        safety_margin=0.78,
        environmental_load=0.22,
        current_risks=[],
        active_goals=[],
    )
    affect_blend = AffectBlendState(
        care=0.68,
        reverence=0.2,
        innocence=0.12,
        defense=0.22,
        future_pull=0.34,
        shared_world_pull=0.42,
        distress=0.18,
        confidence=0.54,
        conflict_level=0.22,
        residual_tension=0.26,
        reportability_pressure=0.44,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.18,
        recovery_need=0.14,
        safety_bias=0.1,
        recent_strain=0.16,
        current_risks=[],
    )
    first_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["person:user"],
        current_risks=[],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.0,
    )
    first_dynamics = advance_contact_dynamics(
        contact_field=first_field.to_dict(),
        previous_residue=0.0,
    )
    second_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["person:user"],
        current_risks=[],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.38,
    )
    second_dynamics = advance_contact_dynamics(
        contact_field=second_field.to_dict(),
        previous_dynamics=first_dynamics.to_dict(),
        previous_workspace={
            "active_slots": [{"label": "person:user"}],
            "workspace_stability": 0.54,
            "recurrent_residue": 0.38,
        },
        previous_residue=0.38,
    )

    assert second_dynamics.dynamics_mode in {"reentrant", "guarded_reentry"}
    assert second_dynamics.reentry_bias > first_dynamics.reentry_bias
    assert second_dynamics.carryover_strength > 0.0
    assert second_dynamics.stabilized_points
    assert second_dynamics.stabilized_points[0].stabilized_activation >= second_dynamics.stabilized_points[0].base_intensity * 0.9


def test_access_projection_reads_stabilized_points_when_available() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.22,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.7,
        safety_margin=0.26,
        environmental_load=0.66,
        current_risks=["danger"],
        active_goals=["repair"],
    )
    affect_blend = AffectBlendState(
        care=0.62,
        reverence=0.42,
        innocence=0.1,
        defense=0.78,
        future_pull=0.16,
        shared_world_pull=0.14,
        distress=0.7,
        confidence=0.16,
        conflict_level=0.56,
        residual_tension=0.6,
        reportability_pressure=0.68,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.72,
        recovery_need=0.58,
        safety_bias=0.74,
        recent_strain=0.62,
        current_risks=["danger"],
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["person:user"],
        current_risks=["danger"],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.44,
    )
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_workspace={
            "active_slots": [{"label": "person:user"}],
            "workspace_stability": 0.52,
            "recurrent_residue": 0.44,
        },
        previous_residue=0.44,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )

    assert access_projection.regions
    assert any("contact_reentry_bias" in cue for cue in access_projection.cues)
    assert access_projection.actionable_slice
