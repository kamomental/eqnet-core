# -*- coding: utf-8 -*-

from inner_os.access_projection import project_access_regions
from inner_os.affect_blend import AffectBlendState
from inner_os.contact_dynamics import advance_contact_dynamics
from inner_os.contact_field import derive_contact_field
from inner_os.contact_reflection_state import derive_contact_reflection_state
from inner_os.constraint_field import derive_constraint_field
from inner_os.scene_state import derive_scene_state


def test_contact_reflection_state_allows_reflect_then_question_when_contact_is_open() -> None:
    scene_state = derive_scene_state(
        place_mode="private_room",
        privacy_level=0.82,
        social_topology="one_to_one",
        task_phase="ongoing",
        temporal_phase="present",
        norm_pressure=0.18,
        safety_margin=0.84,
        environmental_load=0.16,
        current_risks=[],
        active_goals=["attune"],
    )
    affect_blend = AffectBlendState(
        care=0.62,
        reverence=0.22,
        innocence=0.16,
        defense=0.16,
        future_pull=0.22,
        shared_world_pull=0.26,
        distress=0.18,
        confidence=0.48,
        conflict_level=0.14,
        residual_tension=0.12,
        reportability_pressure=0.76,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.18,
        recovery_need=0.14,
        safety_bias=0.2,
        recent_strain=0.12,
        current_risks=[],
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["そのことはまだ残っている。"],
        current_risks=[],
        related_person_ids=["user"],
        memory_anchor="港での約束",
        previous_residue=0.12,
    )
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_residue=0.1,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )

    reflection_state = derive_contact_reflection_state(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        access_projection=access_projection.to_dict(),
        constraint_field=constraint_field.to_dict(),
        current_risks=[],
    )

    assert reflection_state.state == "open_reflection"
    assert reflection_state.reflection_style == "reflect_then_question"
    assert reflection_state.transmit_share >= reflection_state.absorb_share


def test_contact_reflection_state_prefers_reflect_only_under_guarded_contact() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.22,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.74,
        safety_margin=0.22,
        environmental_load=0.68,
        current_risks=["danger"],
        active_goals=["repair"],
    )
    affect_blend = AffectBlendState(
        care=0.66,
        reverence=0.32,
        innocence=0.1,
        defense=0.82,
        future_pull=0.18,
        shared_world_pull=0.12,
        distress=0.74,
        confidence=0.16,
        conflict_level=0.6,
        residual_tension=0.64,
        reportability_pressure=0.68,
        dominant_mode="care",
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend,
        stress=0.78,
        recovery_need=0.62,
        safety_bias=0.78,
        recent_strain=0.66,
        current_risks=["danger"],
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=scene_state.__dict__,
        current_focus="person:user",
        reportable_facts=["そのことはまだ残っている。"],
        current_risks=["danger"],
        related_person_ids=["user"],
        memory_anchor="港での約束",
        previous_residue=0.42,
    )
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_residue=0.38,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )

    reflection_state = derive_contact_reflection_state(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        access_projection=access_projection.to_dict(),
        constraint_field=constraint_field.to_dict(),
        current_risks=["danger"],
    )

    assert reflection_state.reflection_style in {"reflect_only", "boundary_only"}
    assert reflection_state.absorb_share >= reflection_state.transmit_share or reflection_state.block_share >= 0.34
