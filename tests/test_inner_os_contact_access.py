from inner_os.access_projection import project_access_regions
from inner_os.affect_blend import AffectBlendState
from inner_os.contact_field import derive_contact_field
from inner_os.constraint_field import derive_constraint_field
from inner_os.scene_state import derive_scene_state


def test_contact_field_emits_relational_and_protective_points() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.24,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.7,
        safety_margin=0.28,
        environmental_load=0.64,
        current_risks=["danger"],
        active_goals=["repair"],
    )
    affect_blend = AffectBlendState(
        care=0.62,
        reverence=0.44,
        innocence=0.12,
        defense=0.76,
        future_pull=0.2,
        shared_world_pull=0.18,
        distress=0.68,
        confidence=0.18,
        conflict_level=0.56,
        residual_tension=0.6,
        reportability_pressure=0.64,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.72,
        recovery_need=0.58,
        safety_bias=0.74,
        recent_strain=0.6,
        current_risks=["danger"],
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["I want to say more here."],
        current_risks=["danger"],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.42,
    )

    assert contact_field.field_mode == "guarded"
    assert contact_field.dominant_point
    assert any(point.source_modality == "social" for point in contact_field.points)
    assert any(point.source_modality == "protective" for point in contact_field.points)
    assert any("person:user" in point.binding_tags for point in contact_field.points)


def test_access_projection_preserves_actionable_slice_under_withhold() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.22,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.72,
        safety_margin=0.24,
        environmental_load=0.68,
        current_risks=["danger"],
        active_goals=["repair"],
    )
    affect_blend = AffectBlendState(
        care=0.64,
        reverence=0.46,
        innocence=0.1,
        defense=0.8,
        future_pull=0.16,
        shared_world_pull=0.14,
        distress=0.72,
        confidence=0.16,
        conflict_level=0.58,
        residual_tension=0.62,
        reportability_pressure=0.7,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.74,
        recovery_need=0.6,
        safety_bias=0.76,
        recent_strain=0.64,
        current_risks=["danger"],
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["I want to say more here."],
        current_risks=["danger"],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.46,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )

    assert access_projection.projection_mode == "guarded_projection"
    assert not access_projection.reportable_slice
    assert access_projection.withheld_slice
    assert access_projection.actionable_slice
    assert any(region.source == "protective" for region in access_projection.regions)
