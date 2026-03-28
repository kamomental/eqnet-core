from inner_os.green_kernel_contracts import (
    SharedInnerField,
    build_green_kernel_composition,
    project_affective_green_delta,
    project_boundary_transform_delta,
    project_memory_green_delta,
    project_relational_green_delta,
    project_residual_carry_delta,
)


def test_green_kernel_projection_contracts_are_typed_and_instantiable() -> None:
    field = SharedInnerField()
    assert field.memory_activation == 0.0
    assert field.cues == ()


def test_memory_green_projection_uses_temporal_and_thread_signals() -> None:
    delta = project_memory_green_delta(
        temporal_membrane_bias={
            "timeline_coherence": 0.42,
            "reentry_pull": 0.58,
            "relation_reentry_pull": 0.31,
            "continuity_pressure": 0.36,
            "cues": ["temporal_membrane_reentry"],
        },
        memory_evidence_bundle={
            "facts_current": [{"id": "m1"}],
            "timeline_events": [{"id": "t1"}],
            "source_refs": ["episode:1"],
        },
        autobiographical_thread={
            "strength": 0.41,
            "reasons": ["discussion_registry"],
        },
    )

    assert delta.source == "memory_green"
    assert delta.memory_activation > 0.0
    assert delta.reopening_pull > 0.0
    assert "temporal_membrane_reentry" in delta.cues


def test_green_kernel_composition_keeps_boundary_and_residual_separate() -> None:
    composition = build_green_kernel_composition(
        temporal_membrane_bias={
            "timeline_coherence": 0.4,
            "reentry_pull": 0.54,
            "relation_reentry_pull": 0.22,
            "continuity_pressure": 0.28,
            "supersession_pressure": 0.12,
            "cues": ["temporal_membrane_reentry"],
        },
        memory_evidence_bundle={
            "facts_current": [{"id": "m1"}],
            "timeline_events": [{"id": "t1"}],
            "source_refs": ["episode:1"],
        },
        affect_blend_state={
            "care": 0.52,
            "future_pull": 0.37,
            "shared_world_pull": 0.31,
            "distress": 0.29,
            "conflict_level": 0.44,
            "residual_tension": 0.33,
            "reportability_pressure": 0.4,
            "defense": 0.46,
            "confidence": 0.58,
            "cues": ["blend_conflict_high"],
        },
        recent_dialogue_state={
            "thread_carry": 0.62,
            "reopen_pressure": 0.51,
            "dominant_inputs": ["history_overlap"],
        },
        discussion_thread_state={
            "thread_visibility": 0.57,
            "revisit_readiness": 0.48,
            "dominant_inputs": ["revisit_marker"],
        },
        issue_state={
            "question_pressure": 0.18,
            "pause_readiness": 0.43,
            "resolution_readiness": 0.21,
            "dominant_inputs": ["pause_marker"],
        },
        contact_reflection_state={
            "state": "guarded_reflection",
            "reflection_style": "reflect_only",
            "transmit_share": 0.31,
            "reflect_share": 0.44,
            "absorb_share": 0.25,
            "block_share": 0.0,
            "cues": ["contact_reflection_guarded"],
        },
        boundary_transform={
            "gate_mode": "narrow",
            "residual_pressure": 0.48,
            "cues": ["boundary_softened_candidate"],
        },
        residual_reflection={
            "mode": "withheld",
            "strength": 0.62,
            "cues": ["residual_withheld"],
        },
        autobiographical_thread={
            "mode": "unfinished_thread",
            "strength": 0.38,
            "reasons": ["residual_reflection"],
        },
    )

    assert composition.memory_delta.source == "memory_green"
    assert composition.affective_delta.source == "affective_green"
    assert composition.relational_delta.source == "relational_green"
    assert composition.boundary_delta.source == "boundary_operator"
    assert composition.residual_delta.source == "residual_operator"
    assert composition.field.memory_activation > 0.0
    assert composition.field.affective_charge > 0.0
    assert composition.field.relational_pull > 0.0
    assert composition.field.boundary_tension > 0.0
    assert composition.field.residual_tension > 0.0
    assert "residual_withheld" in composition.field.cues
    assert "contact_reflection_guarded" in composition.field.cues


def test_projection_helpers_can_be_used_independently() -> None:
    affective = project_affective_green_delta(
        affect_blend_state={
            "care": 0.46,
            "distress": 0.33,
            "conflict_level": 0.41,
            "residual_tension": 0.29,
            "reportability_pressure": 0.35,
            "defense": 0.51,
            "confidence": 0.49,
        },
        temporal_membrane_bias={"supersession_pressure": 0.14},
        contact_reflection_state={"absorb_share": 0.22, "block_share": 0.08},
    )
    relational = project_relational_green_delta(
        recent_dialogue_state={"thread_carry": 0.55, "reopen_pressure": 0.44},
        discussion_thread_state={"thread_visibility": 0.48, "revisit_readiness": 0.39},
        issue_state={"question_pressure": 0.16, "pause_readiness": 0.37, "resolution_readiness": 0.19},
        autobiographical_thread={"strength": 0.34},
    )
    boundary = project_boundary_transform_delta(
        boundary_transform={"gate_mode": "narrow", "residual_pressure": 0.4},
        contact_reflection_state={"absorb_share": 0.2, "block_share": 0.1},
    )
    residual = project_residual_carry_delta(
        residual_reflection={"mode": "withheld", "strength": 0.58},
        autobiographical_thread={"mode": "unfinished_thread", "strength": 0.33},
    )

    assert affective.affective_charge > 0.0
    assert affective.guardedness > 0.0
    assert relational.relational_pull > 0.0
    assert relational.reopening_pull > 0.0
    assert boundary.boundary_tension > 0.0
    assert residual.residual_tension > 0.0
