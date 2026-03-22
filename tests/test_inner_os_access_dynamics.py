from inner_os.access_dynamics import advance_access_dynamics
from inner_os.access_projection import project_access_regions
from inner_os.affect_blend import AffectBlendState
from inner_os.contact_dynamics import advance_contact_dynamics
from inner_os.contact_field import derive_contact_field
from inner_os.conscious_workspace import ignite_conscious_workspace
from inner_os.constraint_field import derive_constraint_field
from inner_os.interaction.models import LiveInteractionRegulation, RelationalMood, SituationState
from inner_os.interaction_option_search import generate_interaction_option_candidates
from inner_os.scene_state import derive_scene_state


def test_access_dynamics_can_hold_membrane_inertia_across_turns() -> None:
    scene_state = derive_scene_state(
        place_mode="relational_private",
        privacy_level=0.64,
        social_topology="one_to_one",
        task_phase="ongoing",
        temporal_phase="ongoing",
        norm_pressure=0.28,
        safety_margin=0.78,
        environmental_load=0.24,
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
    contact_field = derive_contact_field(
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
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_residue=0.0,
    )
    first_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )
    second_dynamics = advance_access_dynamics(
        access_projection=first_projection.to_dict(),
        previous_access_dynamics=first_projection.to_dict(),
        previous_workspace={
            "active_slots": [{"label": "person:user"}],
            "workspace_stability": 0.52,
            "recurrent_residue": 0.34,
        },
        previous_residue=0.34,
        current_risks=[],
    )

    assert second_dynamics.dynamics_mode in {"inertial_projection", "guarded_inertial_projection"}
    assert second_dynamics.membrane_inertia > 0.0
    assert second_dynamics.stabilized_regions
    assert second_dynamics.actionable_slice
    assert any("access_membrane_inertia" in cue for cue in second_dynamics.cues)


def test_conscious_workspace_can_read_access_dynamics_stabilized_regions() -> None:
    scene_state = derive_scene_state(
        place_mode="public_threshold",
        privacy_level=0.22,
        social_topology="public_visible",
        task_phase="repair",
        temporal_phase="rupture",
        norm_pressure=0.7,
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
        reportable_facts=["person:user"],
        current_risks=["danger"],
        related_person_ids=["user"],
        memory_anchor="harbor slope",
        previous_residue=0.46,
    )
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_workspace={
            "active_slots": [{"label": "person:user"}],
            "workspace_stability": 0.5,
            "recurrent_residue": 0.42,
        },
        previous_residue=0.42,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )
    access_dynamics = advance_access_dynamics(
        access_projection=access_projection.to_dict(),
        previous_workspace={
            "active_slots": [{"label": "person:user"}],
            "workspace_stability": 0.5,
            "recurrent_residue": 0.42,
        },
        previous_residue=0.42,
        current_risks=["danger"],
    )
    option_candidates = generate_interaction_option_candidates(
        scene_state=scene_state,
        situation_state=SituationState(
            scene_mode="co_present",
            repair_window_open=True,
            shared_attention=0.18,
            social_pressure=0.64,
            continuity_weight=0.34,
            current_phase="rupture",
        ),
        relational_mood=RelationalMood(
            future_pull=0.16,
            reverence=0.46,
            innocence=0.1,
            care=0.64,
            shared_world_pull=0.14,
            confidence_signal=0.16,
        ),
        live_regulation=LiveInteractionRegulation(
            past_loop_pull=0.18,
            future_loop_pull=0.14,
            fantasy_loop_pull=0.08,
            shared_attention_active=0.16,
            strained_pause=0.62,
            repair_window_open=True,
            distance_expectation="respectful_distance",
        ),
        constraint_field=constraint_field.to_dict(),
    )
    workspace = ignite_conscious_workspace(
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        current_focus="person:user",
        reportable_facts=["person:user"],
        current_risks=["danger"],
        related_person_ids=["user"],
        interaction_option_candidates=option_candidates,
        memory_anchor="harbor slope",
        scene_state=scene_state.__dict__,
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        access_projection=access_projection.to_dict(),
        access_dynamics=access_dynamics.to_dict(),
    )

    assert workspace.active_slots
    assert any(slot.source in {"social", "protective"} for slot in workspace.active_slots)
    assert any("access_dynamics:" in cue for cue in workspace.cues)
