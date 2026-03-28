from inner_os.sleep_consolidation_core import SleepConsolidationCore


def test_sleep_consolidation_core_prefers_restabilize_under_high_recovery_and_field_dwell() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "recovery_need": 0.82,
            "roughness_dwell": 0.74,
            "defensive_dwell": 0.66,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.54,
            "replay_horizon": 1,
        },
        persistence_state={
            "recent_strain": 0.72,
            "social_grounding": 0.28,
            "continuity_score": 0.31,
        },
        development_state={
            "belonging": 0.34,
            "trust_bias": 0.38,
            "norm_pressure": 0.48,
            "role_commitment": 0.41,
        },
    )
    assert snapshot.mode == "restabilize"
    assert snapshot.sleep_pressure >= 0.62
    assert "recovery" in snapshot.summary


def test_sleep_consolidation_core_prefers_reconsolidate_when_communal_profile_and_replay_align() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "recovery_need": 0.24,
            "ritual_memory": 0.84,
            "institutional_memory": 0.72,
            "roughness_dwell": 0.14,
            "defensive_dwell": 0.08,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.26,
            "replay_horizon": 2,
        },
        memory_orchestration={
            "reuse_trajectory": 0.72,
            "consolidation_priority": 0.82,
            "monument_salience": 0.64,
            "prospective_memory_pull": 0.44,
            "interference_pressure": 0.18,
            "conscious_mosaic_recentness": 0.68,
        },
        persistence_state={
            "recent_strain": 0.28,
            "social_grounding": 0.74,
            "continuity_score": 0.58,
            "culture_resonance": 0.79,
            "community_resonance": 0.83,
        },
        development_state={
            "belonging": 0.7,
            "trust_bias": 0.66,
            "norm_pressure": 0.71,
            "role_commitment": 0.64,
        },
        personality_state={
            "reflective_bias": 0.56,
            "caution_bias": 0.33,
        },
    )
    assert snapshot.mode == "reconsolidate"
    assert snapshot.replay_priority >= 0.54
    assert snapshot.reconsolidation_priority >= 0.54
    assert snapshot.autobiographical_pull > 0.0


def test_sleep_consolidation_core_prefers_abstract_when_consolidation_is_clean_and_stable() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "recovery_need": 0.18,
            "roughness_dwell": 0.04,
            "defensive_dwell": 0.02,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.12,
            "replay_horizon": 3,
        },
        memory_orchestration={
            "reuse_trajectory": 0.26,
            "consolidation_priority": 0.78,
            "monument_salience": 0.32,
            "prospective_memory_pull": 0.16,
            "interference_pressure": 0.1,
            "conscious_mosaic_recentness": 0.82,
        },
        persistence_state={
            "recent_strain": 0.12,
            "social_grounding": 0.76,
            "continuity_score": 0.81,
            "culture_resonance": 0.52,
            "community_resonance": 0.58,
        },
        development_state={
            "belonging": 0.64,
            "trust_bias": 0.68,
            "norm_pressure": 0.5,
            "role_commitment": 0.62,
        },
        personality_state={
            "reflective_bias": 0.22,
            "caution_bias": 0.28,
        },
    )
    assert snapshot.mode == "abstract"
    assert snapshot.abstraction_readiness >= 0.5
    assert snapshot.identity_preservation_bias > 0.0


def test_sleep_consolidation_core_uses_partner_relation_bias() -> None:
    core = SleepConsolidationCore()
    baseline = core.snapshot(
        current_state={
            "recovery_need": 0.22,
            "roughness_dwell": 0.12,
            "defensive_dwell": 0.08,
            "attachment": 0.78,
            "familiarity": 0.72,
            "trust_memory": 0.74,
            "relation_seed_strength": 0.76,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.18,
            "replay_horizon": 2,
        },
        memory_orchestration={
            "reuse_trajectory": 0.54,
            "consolidation_priority": 0.58,
            "monument_salience": 0.42,
            "prospective_memory_pull": 0.36,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.48,
        },
        persistence_state={
            "recent_strain": 0.24,
            "social_grounding": 0.62,
            "continuity_score": 0.57,
            "culture_resonance": 0.42,
            "community_resonance": 0.46,
        },
        development_state={
            "belonging": 0.48,
            "trust_bias": 0.52,
            "norm_pressure": 0.44,
            "role_commitment": 0.5,
        },
    )
    with_partner = core.snapshot(
        current_state={
            "recovery_need": 0.22,
            "roughness_dwell": 0.12,
            "defensive_dwell": 0.08,
            "related_person_id": "user",
            "attachment": 0.78,
            "familiarity": 0.72,
            "trust_memory": 0.74,
            "relation_seed_strength": 0.76,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.18,
            "replay_horizon": 2,
        },
        memory_orchestration={
            "reuse_trajectory": 0.54,
            "consolidation_priority": 0.58,
            "monument_salience": 0.42,
            "prospective_memory_pull": 0.36,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.48,
        },
        persistence_state={
            "recent_strain": 0.24,
            "social_grounding": 0.62,
            "continuity_score": 0.57,
            "culture_resonance": 0.42,
            "community_resonance": 0.46,
        },
        development_state={
            "belonging": 0.48,
            "trust_bias": 0.52,
            "norm_pressure": 0.44,
            "role_commitment": 0.5,
        },
    )
    assert with_partner.replay_priority > baseline.replay_priority
    assert with_partner.reconsolidation_priority > baseline.reconsolidation_priority
    assert with_partner.autobiographical_pull > baseline.autobiographical_pull
    assert with_partner.identity_preservation_bias > baseline.identity_preservation_bias


def test_sleep_consolidation_core_reads_memory_write_class_bias() -> None:
    core = SleepConsolidationCore()
    body_risk = core.snapshot(
        current_state={
            "recovery_need": 0.22,
            "roughness_dwell": 0.1,
            "defensive_dwell": 0.08,
            "memory_write_class": "body_risk",
            "memory_write_class_counts": {"body_risk": 4, "episodic": 1},
        },
        forgetting_snapshot={"forgetting_pressure": 0.18, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.36,
            "consolidation_priority": 0.42,
            "monument_salience": 0.24,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.16,
            "conscious_mosaic_recentness": 0.38,
        },
        persistence_state={"recent_strain": 0.22, "social_grounding": 0.58, "continuity_score": 0.52},
        development_state={"belonging": 0.46, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.48},
    )
    safe_repeat = core.snapshot(
        current_state={
            "recovery_need": 0.22,
            "roughness_dwell": 0.1,
            "defensive_dwell": 0.08,
            "memory_write_class": "safe_repeat",
            "memory_write_class_counts": {"safe_repeat": 4, "episodic": 1},
        },
        forgetting_snapshot={"forgetting_pressure": 0.18, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.36,
            "consolidation_priority": 0.42,
            "monument_salience": 0.24,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.16,
            "conscious_mosaic_recentness": 0.38,
        },
        persistence_state={"recent_strain": 0.22, "social_grounding": 0.58, "continuity_score": 0.52},
        development_state={"belonging": 0.46, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.48},
    )

    assert body_risk.memory_class_focus == "body_risk"
    assert body_risk.terrain_reweighting_bias > 0.0
    assert body_risk.sleep_pressure > safe_repeat.sleep_pressure
    assert safe_repeat.memory_class_focus == "safe_repeat"
    assert safe_repeat.abstraction_readiness >= body_risk.abstraction_readiness


def test_sleep_consolidation_core_reads_insight_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "memory_write_class": "episodic",
            "memory_write_class_counts": {"episodic": 2},
            "insight_class_focus": "reframed_relation",
            "insight_class_counts": {
                "reframed_relation": 3,
                "insight_trace": 1,
            },
            "insight_link_counts": {
                "bond:user|memory:shared_thread": 3,
                "felt:care|external:shared_thread": 2,
            },
            "insight_terrain_shape_bias": 0.22,
            "insight_terrain_shape_reason": "reframed_relation",
            "insight_anchor_center": [0.18, -0.04, 0.12],
            "insight_anchor_dispersion": 0.31,
        },
        forgetting_snapshot={"forgetting_pressure": 0.16, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.28,
            "consolidation_priority": 0.34,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.26,
        },
        persistence_state={"recent_strain": 0.18, "social_grounding": 0.6, "continuity_score": 0.56},
        development_state={"belonging": 0.48, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.46},
    )

    assert snapshot.insight_class_focus == "reframed_relation"
    assert snapshot.insight_reframing_bias > 0.0
    assert snapshot.association_reweighting_bias > 0.0
    assert snapshot.association_reweighting_focus == "repeated_links"
    assert snapshot.association_reweighting_reason == "repeated_insight_trace"
    assert snapshot.insight_terrain_shape_bias > 0.0
    assert snapshot.insight_terrain_shape_reason == "reframed_relation"
    assert snapshot.insight_terrain_shape_target == "soft_relation"
    assert snapshot.insight_anchor_center == (0.18, -0.04, 0.12)
    assert snapshot.insight_anchor_dispersion == 0.31


def test_sleep_consolidation_core_reads_commitment_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "memory_write_class": "repair_trace",
            "memory_write_class_counts": {"repair_trace": 2, "episodic": 1},
            "commitment_target_focus": "repair",
            "commitment_state_focus": "commit",
            "commitment_target_counts": {"repair": 3, "hold": 1},
            "commitment_carry_bias": 0.38,
            "commitment_followup_focus": "reopen_softly",
            "commitment_mode_focus": "repair",
            "commitment_carry_reason": "commit:repair",
        },
        forgetting_snapshot={"forgetting_pressure": 0.16, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.28,
            "consolidation_priority": 0.34,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.26,
        },
        persistence_state={"recent_strain": 0.18, "social_grounding": 0.6, "continuity_score": 0.56},
        development_state={"belonging": 0.48, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.46},
    )

    assert snapshot.commitment_target_focus == "repair"
    assert snapshot.commitment_state_focus == "commit"
    assert snapshot.commitment_carry_bias == 0.38
    assert snapshot.commitment_followup_focus == "reopen_softly"
    assert snapshot.commitment_mode_focus == "repair"
    assert snapshot.commitment_carry_reason == "commit:repair"


def test_sleep_consolidation_core_derives_agenda_window_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "agenda_window_state": {
                "state": "next_same_culture_window",
                "reason": "hold_for_same_culture_context",
                "score": 0.58,
                "winner_margin": 0.14,
                "deferral_budget": 0.66,
                "carry_target": "same_culture_window",
            },
        },
        forgetting_snapshot={"forgetting_pressure": 0.16, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.28,
            "consolidation_priority": 0.34,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.26,
        },
        persistence_state={"recent_strain": 0.18, "social_grounding": 0.6, "continuity_score": 0.56},
        development_state={"belonging": 0.48, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.46},
    )

    assert snapshot.agenda_window_focus == "next_same_culture_window"
    assert snapshot.agenda_window_bias > 0.0
    assert snapshot.agenda_window_reason == "hold_for_same_culture_context"
    assert snapshot.agenda_window_carry_target == "same_culture_window"


def test_sleep_consolidation_core_derives_learning_and_social_experiment_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "learning_mode_state": {
                "state": "repair_probe",
                "score": 0.58,
                "winner_margin": 0.14,
                "probe_room": 0.48,
            },
            "social_experiment_loop_state": {
                "state": "repair_signal_probe",
                "score": 0.54,
                "winner_margin": 0.12,
                "probe_intensity": 0.42,
            },
        },
        forgetting_snapshot={"forgetting_pressure": 0.16, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.28,
            "consolidation_priority": 0.34,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.26,
        },
        persistence_state={"recent_strain": 0.18, "social_grounding": 0.6, "continuity_score": 0.56},
        development_state={"belonging": 0.48, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.46},
    )

    assert snapshot.learning_mode_focus == "repair_probe"
    assert snapshot.learning_mode_carry_bias > 0.0
    assert snapshot.social_experiment_focus == "repair_signal_probe"
    assert snapshot.social_experiment_carry_bias > 0.0


def test_sleep_consolidation_core_derives_temporal_membrane_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "temporal_membrane_mode": "reentry",
            "temporal_timeline_coherence": 0.44,
            "temporal_reentry_pull": 0.58,
            "temporal_supersession_pressure": 0.09,
            "temporal_continuity_pressure": 0.36,
            "temporal_relation_reentry_pull": 0.41,
        },
        forgetting_snapshot={"forgetting_pressure": 0.16, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.28,
            "consolidation_priority": 0.34,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.12,
            "interference_pressure": 0.14,
            "conscious_mosaic_recentness": 0.26,
        },
        persistence_state={"recent_strain": 0.18, "social_grounding": 0.6, "continuity_score": 0.56},
        development_state={"belonging": 0.48, "trust_bias": 0.5, "norm_pressure": 0.42, "role_commitment": 0.46},
    )

    assert snapshot.temporal_membrane_focus == "reentry"
    assert snapshot.temporal_timeline_bias > 0.0
    assert snapshot.temporal_reentry_bias > 0.0
    assert snapshot.temporal_continuity_bias > 0.0
    assert snapshot.temporal_relation_reentry_bias > 0.0


def test_sleep_consolidation_core_derives_temperament_sleep_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "temperament_forward_trace": 0.62,
            "temperament_guard_trace": 0.14,
            "temperament_bond_trace": 0.28,
            "temperament_recovery_trace": 0.08,
        },
        forgetting_snapshot={"forgetting_pressure": 0.12, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.32,
            "consolidation_priority": 0.42,
            "monument_salience": 0.18,
            "prospective_memory_pull": 0.14,
            "interference_pressure": 0.12,
            "conscious_mosaic_recentness": 0.34,
        },
        persistence_state={"recent_strain": 0.12, "social_grounding": 0.64, "continuity_score": 0.6},
        development_state={"belonging": 0.5, "trust_bias": 0.52, "norm_pressure": 0.38, "role_commitment": 0.44},
    )

    assert snapshot.temperament_focus == "forward"
    assert snapshot.temperament_forward_bias > 0.0
    assert snapshot.temperament_guard_bias >= 0.0
    assert snapshot.temperament_bond_bias >= 0.0
    assert snapshot.temperament_recovery_bias >= 0.0


def test_sleep_consolidation_core_derives_body_and_relational_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "body_homeostasis_state": {
                "state": "recovering",
                "score": 0.58,
                "winner_margin": 0.16,
            },
            "relational_continuity_state": {
                "state": "reopening",
                "score": 0.54,
                "winner_margin": 0.12,
            },
        },
        forgetting_snapshot={"forgetting_pressure": 0.14, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.31,
            "consolidation_priority": 0.4,
            "monument_salience": 0.16,
            "prospective_memory_pull": 0.1,
            "interference_pressure": 0.13,
            "conscious_mosaic_recentness": 0.29,
        },
        persistence_state={"recent_strain": 0.2, "social_grounding": 0.58, "continuity_score": 0.55},
        development_state={"belonging": 0.46, "trust_bias": 0.5, "norm_pressure": 0.41, "role_commitment": 0.45},
    )

    assert snapshot.body_homeostasis_focus == "recovering"
    assert snapshot.body_homeostasis_carry_bias > 0.0
    assert snapshot.relational_continuity_focus == "reopening"
    assert snapshot.relational_continuity_carry_bias > 0.0


def test_sleep_consolidation_core_derives_expressive_style_carry_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "expressive_style_state": {
                "state": "warm_companion",
                "score": 0.61,
                "winner_margin": 0.16,
                "lightness_room": 0.42,
                "continuity_weight": 0.58,
            },
        },
        forgetting_snapshot={"forgetting_pressure": 0.14, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.31,
            "consolidation_priority": 0.4,
            "monument_salience": 0.16,
            "prospective_memory_pull": 0.1,
            "interference_pressure": 0.13,
            "conscious_mosaic_recentness": 0.29,
        },
        persistence_state={"recent_strain": 0.16, "social_grounding": 0.62, "continuity_score": 0.61},
        development_state={"belonging": 0.5, "trust_bias": 0.53, "norm_pressure": 0.36, "role_commitment": 0.42},
    )

    assert snapshot.expressive_style_focus == "warm_companion"
    assert snapshot.expressive_style_carry_bias > 0.0


def test_sleep_consolidation_core_derives_expressive_history_and_banter_carry() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "expressive_style_state": {
                "state": "warm_companion",
                "score": 0.61,
                "winner_margin": 0.16,
                "lightness_room": 0.42,
                "continuity_weight": 0.58,
            },
            "expressive_style_history_focus": "warm_companion",
            "expressive_style_history_bias": 0.08,
            "relational_style_memory_state": {
                "state": "light_playful",
                "banter_style": "gentle_tease",
                "lexical_variation_bias": 0.38,
            },
            "banter_style_focus": "gentle_tease",
            "lexical_variation_carry_bias": 0.1,
        },
        forgetting_snapshot={"forgetting_pressure": 0.14, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.31,
            "consolidation_priority": 0.4,
            "monument_salience": 0.16,
            "prospective_memory_pull": 0.1,
            "interference_pressure": 0.13,
            "conscious_mosaic_recentness": 0.29,
        },
        persistence_state={"recent_strain": 0.16, "social_grounding": 0.62, "continuity_score": 0.61},
        development_state={"belonging": 0.5, "trust_bias": 0.53, "norm_pressure": 0.36, "role_commitment": 0.42},
    )

    assert snapshot.expressive_style_history_focus == "warm_companion"
    assert snapshot.expressive_style_history_bias > 0.0
    assert snapshot.banter_style_focus == "gentle_tease"
    assert snapshot.lexical_variation_carry_bias > 0.0


def test_sleep_consolidation_core_derives_homeostasis_budget_bias() -> None:
    core = SleepConsolidationCore()
    snapshot = core.snapshot(
        current_state={
            "homeostasis_budget_state": "recovering",
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.12,
            "body_homeostasis_state": {
                "state": "recovering",
                "score": 0.52,
                "winner_margin": 0.14,
            },
        },
        forgetting_snapshot={"forgetting_pressure": 0.14, "replay_horizon": 2},
        memory_orchestration={
            "reuse_trajectory": 0.31,
            "consolidation_priority": 0.4,
            "monument_salience": 0.16,
            "prospective_memory_pull": 0.1,
            "interference_pressure": 0.13,
            "conscious_mosaic_recentness": 0.29,
        },
        persistence_state={"recent_strain": 0.2, "social_grounding": 0.58, "continuity_score": 0.55},
        development_state={"belonging": 0.46, "trust_bias": 0.5, "norm_pressure": 0.41, "role_commitment": 0.45},
    )

    assert snapshot.homeostasis_budget_focus == "recovering"
    assert snapshot.homeostasis_budget_bias > 0.0
