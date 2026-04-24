from emot_terrain_lab.sleep.inner_os_bridge import (
    build_inner_os_sleep_snapshot,
    build_inner_os_sleep_snapshot_for_system,
    write_inner_os_sleep_snapshot_for_system,
)


class _StubMemoryLayer:
    def __init__(self, size: int, attr: str) -> None:
        setattr(self, attr, [object()] * size)


class _StubSystem:
    def __init__(self) -> None:
        self.l1 = _StubMemoryLayer(18, "experiences")
        self.l2 = _StubMemoryLayer(7, "episodes")
        self.l3 = _StubMemoryLayer(4, "patterns")

    def rest_state(self) -> dict:
        return {
            "active": True,
            "fatigue_streak": 2,
            "history": [
                {"triggers": {"fatigue": True, "loop": False, "overload": False}},
                {"triggers": {"fatigue": True, "loop": True, "overload": False}},
                {"triggers": {"fatigue": False, "loop": True, "overload": True}},
            ],
        }

    def field_metrics_state(self) -> list[dict]:
        return [{"entropy": 8.4, "enthalpy": 0.62}]


def test_build_inner_os_sleep_snapshot_wraps_legacy_signals() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={
            "active": True,
            "fatigue_streak": 2,
            "history": [{"triggers": {"fatigue": True, "loop": True, "overload": False}}],
        },
        latest_field_metrics={"entropy": 8.8, "enthalpy": 0.66},
        nightly_summary={
            "forgetting": {"forgetting_pressure": 0.52, "replay_horizon": 1},
            "culture_stats": {"harbor": {"count": 8, "mean_politeness": 0.52, "mean_intimacy": 0.41}},
            "policy_feedback": {"enabled": True, "intimacy_after": 0.48},
            "inner_os_sleep_autobiographical_thread_mode": "unfinished_thread",
            "inner_os_sleep_autobiographical_thread_anchor": "harbor promise",
            "inner_os_sleep_autobiographical_thread_focus": "unfinished promise",
            "inner_os_sleep_autobiographical_thread_strength": 0.44,
            "inner_os_partner_relation_summary": {
                "person_id": "user",
                "attachment": 0.74,
                "familiarity": 0.68,
                "trust_memory": 0.72,
                "strength": 0.76,
            },
        },
        memory_inventory={"l1_count": 16, "l2_count": 8, "l3_count": 3, "recent_experiences": 10},
        persistence_state={"recent_strain": 0.63, "social_grounding": 0.36, "continuity_score": 0.42},
        development_state={"belonging": 0.46, "trust_bias": 0.44, "norm_pressure": 0.58, "role_commitment": 0.51},
    )

    assert result["schema"] == "inner_os_sleep_consolidation_snapshot/v1"
    assert result["snapshot"]["mode"] in {"restabilize", "defragment", "reconsolidate", "replay", "abstract", "settle"}
    assert result["derived_inputs"]["current_state"]["recovery_need"] > 0.0
    assert result["derived_inputs"]["current_state"]["related_person_id"] == "user"
    assert result["derived_inputs"]["current_state"]["relation_seed_strength"] == 0.76
    assert result["derived_inputs"]["current_state"]["autobiographical_thread_mode"] == "unfinished_thread"
    assert result["snapshot"]["autobiographical_thread_anchor"] == "harbor promise"
    assert result["snapshot"]["growth_state"]["relational_trust"] >= 0.0
    assert result["snapshot"]["growth_state"]["dominant_transition"]
    assert result["snapshot"]["growth_replay_axes"]["bond"]["value"] >= 0.0
    assert result["snapshot"]["memory_dynamics_state"]["dominant_mode"] in {
        "stabilize",
        "reconsolidate",
        "ignite",
        "protect",
        "prospect",
    }
    assert result["snapshot"]["memory_dynamics_state"]["monument_salience"] >= 0.0
    assert result["snapshot"]["memory_dynamics_axes"]["salience"]["value"] >= 0.0
    assert result["snapshot"]["memory_dynamics_axes"]["consolidation"]["value"] >= 0.0
    assert result["derived_inputs"]["forgetting_snapshot"]["replay_horizon"] == 1
    assert result["derived_inputs"]["memory_orchestration"]["consolidation_priority"] > 0.0


def test_build_inner_os_sleep_snapshot_for_system_reads_existing_emotional_memory_system_shape() -> None:
    result = build_inner_os_sleep_snapshot_for_system(
        _StubSystem(),
        nightly_summary={
            "culture_stats": {"town": {"count": 10, "mean_politeness": 0.6, "mean_intimacy": 0.38}},
            "resonance": {"summary": {"energy": 0.44, "objective": 0.36, "corr": 0.22}},
        },
        persistence_state={"recent_strain": 0.52, "social_grounding": 0.48, "continuity_score": 0.51},
        development_state={"belonging": 0.58, "trust_bias": 0.54, "norm_pressure": 0.49, "role_commitment": 0.56},
        personality_state={"reflective_bias": 0.34, "caution_bias": 0.29},
    )

    assert result["schema"] == "inner_os_sleep_consolidation_snapshot/v1"
    assert result["derived_inputs"]["current_state"]["roughness_dwell"] > 0.0
    assert result["derived_inputs"]["memory_orchestration"]["reuse_trajectory"] > 0.0
    assert result["snapshot"]["identity_preservation_bias"] > 0.0


def test_write_inner_os_sleep_snapshot_for_system_persists_json(tmp_path) -> None:
    out_path = tmp_path / "inner_os_sleep_snapshot.json"
    payload = write_inner_os_sleep_snapshot_for_system(
        _StubSystem(),
        out_path=out_path,
        nightly_summary={
            "culture_stats": {"town": {"count": 6, "mean_politeness": 0.55, "mean_intimacy": 0.31}},
        },
    )

    assert out_path.exists()
    written = out_path.read_text(encoding="utf-8")
    assert "inner_os_sleep_consolidation_snapshot/v1" in written
    assert payload["schema"] == "inner_os_sleep_consolidation_snapshot/v1"


def test_build_inner_os_sleep_snapshot_carries_partner_relation_summary_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.2, "enthalpy": 0.34},
        nightly_summary={
            "inner_os_partner_relation_summary": {
                "person_id": "user",
                "attachment": 0.78,
                "familiarity": 0.73,
                "trust_memory": 0.75,
                "strength": 0.81,
                "address_hint": "companion",
                "timing_hint": "open",
                "stance_hint": "familiar",
                "social_interpretation": "familiar:companion:open",
            },
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["related_person_id"] == "user"
    assert current_state["attachment"] == 0.78
    assert current_state["familiarity"] == 0.73
    assert current_state["trust_memory"] == 0.75
    assert current_state["relation_seed_strength"] == 0.81
    assert current_state["partner_address_hint"] == "companion"
    assert current_state["partner_timing_hint"] == "open"
    assert current_state["partner_stance_hint"] == "familiar"
    assert current_state["partner_social_interpretation"] == "familiar:companion:open"


def test_build_inner_os_sleep_snapshot_carries_partner_relation_registry_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.2, "enthalpy": 0.34},
        nightly_summary={
            "inner_os_partner_relation_registry_summary": {
                "dominant_person_id": "user",
                "top_person_ids": ["user", "friend"],
                "total_people": 2,
                "persons": {
                    "user": {
                        "person_id": "user",
                        "adaptive_traits": {
                            "attachment": 0.78,
                            "familiarity": 0.73,
                            "trust_memory": 0.75,
                            "continuity_score": 0.69,
                            "social_grounding": 0.63,
                        },
                        "address_hint": "companion",
                        "timing_hint": "open",
                        "stance_hint": "familiar",
                        "social_interpretation": "familiar:companion:open",
                    },
                    "friend": {
                        "person_id": "friend",
                        "adaptive_traits": {
                            "attachment": 0.42,
                            "familiarity": 0.38,
                            "trust_memory": 0.36,
                            "continuity_score": 0.31,
                            "social_grounding": 0.28,
                        },
                    },
                },
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:user|friend",
                "top_thread_ids": ["threaded_group:user|friend"],
                "total_threads": 1,
                "threads": {
                    "threaded_group:user|friend": {
                        "thread_id": "threaded_group:user|friend",
                        "dominant_person_id": "user",
                        "top_person_ids": ["user", "friend"],
                        "total_people": 2,
                        "continuity_score": 0.62,
                    }
                },
            },
            "inner_os_discussion_thread_registry_summary": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
                "thread_scores": {"repair_anchor": 0.71},
                "uncertainty": 0.18,
            },
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["person_registry_snapshot"]["dominant_person_id"] == "user"
    assert current_state["group_thread_registry_snapshot"]["dominant_thread_id"] == "threaded_group:user|friend"
    assert current_state["discussion_thread_registry_snapshot"]["dominant_anchor"] == "repair anchor"
    assert current_state["related_person_id"] == "user"
    assert current_state["related_person_ids"] == ["user", "friend"]
    assert current_state["attachment"] == 0.78
    assert current_state["partner_address_hint"] == "companion"


def test_build_inner_os_sleep_snapshot_carries_memory_class_summary_and_terrain_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.2, "enthalpy": 0.34},
        nightly_summary={
            "inner_os_memory_class_summary": {
                "dominant_class": "bond_protection",
                "counts": {"bond_protection": 3, "episodic": 1},
            },
            "inner_os_insight_summary": {
                "dominant_insight_class": "reframed_relation",
                "insight_class_counts": {"reframed_relation": 2, "insight_trace": 1},
                "insight_link_counts": {"bond:user|memory:harbor_slope": 3},
                "association_reweighting_bias": 0.33,
                "association_reweighting_focus": "reframed_links",
                "association_reweighting_reason": "reframed_relation",
                "insight_reframing_bias": 0.27,
                "insight_terrain_shape_bias": 0.18,
                "insight_terrain_shape_reason": "reframed_relation",
                "insight_terrain_shape_target": "soft_relation",
                "insight_anchor_center": [0.12, -0.04, 0.08],
                "insight_anchor_dispersion": 0.26,
            },
            "inner_os_sleep_memory_class_focus": "repair_trace",
            "inner_os_sleep_terrain_reweighting_bias": 0.46,
            "inner_os_sleep_insight_class_focus": "reframed_relation",
            "inner_os_sleep_insight_reframing_bias": 0.29,
            "inner_os_sleep_association_reweighting_bias": 0.35,
            "inner_os_sleep_association_reweighting_focus": "reframed_links",
            "inner_os_sleep_association_reweighting_reason": "reframed_relation",
            "inner_os_sleep_insight_terrain_shape_bias": 0.21,
            "inner_os_sleep_insight_terrain_shape_reason": "reframed_relation",
            "inner_os_sleep_insight_terrain_shape_target": "soft_relation",
            "inner_os_sleep_insight_anchor_center": [0.14, -0.02, 0.1],
            "inner_os_sleep_insight_anchor_dispersion": 0.24,
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["memory_write_class"] == "bond_protection"
    assert current_state["memory_write_class_counts"] == {"bond_protection": 3, "episodic": 1}
    assert current_state["terrain_reweighting_bias"] == 0.46
    assert current_state["insight_class_focus"] == "reframed_relation"
    assert current_state["insight_link_counts"] == {"bond:user|memory:harbor_slope": 3}
    assert current_state["association_reweighting_bias"] == 0.35
    assert current_state["association_reweighting_focus"] == "reframed_links"
    assert current_state["association_reweighting_reason"] == "reframed_relation"
    assert current_state["insight_reframing_bias"] == 0.29
    assert current_state["insight_terrain_shape_bias"] == 0.21
    assert current_state["insight_terrain_shape_reason"] == "reframed_relation"
    assert current_state["insight_terrain_shape_target"] == "soft_relation"
    assert current_state["insight_anchor_center"] == [0.14, -0.02, 0.1]
    assert current_state["insight_anchor_dispersion"] == 0.24


def test_build_inner_os_sleep_snapshot_carries_commitment_summary_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.1, "enthalpy": 0.32},
        nightly_summary={
            "inner_os_commitment_summary": {
                "dominant_target": "repair",
                "dominant_state": "commit",
                "dominant_reason": "repair_trace",
                "target_counts": {"repair": 3, "hold": 1},
                "state_counts": {"commit": 3, "settle": 1},
                "weighted_target_counts": {"repair": 1.8, "hold": 0.4},
                "commitment_carry_bias": 0.34,
                "commitment_followup_focus": "reopen_softly",
                "commitment_mode_focus": "repair",
            },
            "inner_os_sleep_commitment_target_focus": "repair",
            "inner_os_sleep_commitment_state_focus": "commit",
            "inner_os_sleep_commitment_carry_bias": 0.39,
            "inner_os_sleep_commitment_followup_focus": "reopen_softly",
            "inner_os_sleep_commitment_mode_focus": "repair",
            "inner_os_sleep_commitment_carry_reason": "commit:repair",
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["commitment_target_focus"] == "repair"
    assert current_state["commitment_state_focus"] == "commit"
    assert current_state["commitment_target_counts"] == {"repair": 3, "hold": 1}
    assert current_state["commitment_carry_bias"] == 0.39
    assert current_state["commitment_followup_focus"] == "reopen_softly"
    assert current_state["commitment_mode_focus"] == "repair"
    assert current_state["commitment_carry_reason"] == "commit:repair"


def test_build_inner_os_sleep_snapshot_carries_agenda_summary_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.1, "enthalpy": 0.31},
        nightly_summary={
            "inner_os_agenda_summary": {
                "dominant_agenda": "repair",
                "dominant_reason": "repair_window",
                "state_counts": {"repair": 3, "revisit": 1},
                "weighted_state_counts": {"repair": 1.6, "revisit": 0.5},
                "agenda_carry_bias": 0.27,
            },
            "inner_os_sleep_agenda_focus": "repair",
            "inner_os_sleep_agenda_bias": 0.31,
            "inner_os_sleep_agenda_reason": "repair_window",
            "inner_os_sleep_agenda_window_focus": "next_same_group_window",
            "inner_os_sleep_agenda_window_bias": 0.17,
            "inner_os_sleep_agenda_window_reason": "wait_for_group_thread",
            "inner_os_sleep_agenda_window_carry_target": "same_group_window",
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["agenda_focus"] == "repair"
    assert current_state["agenda_bias"] == 0.31
    assert current_state["agenda_reason"] == "repair_window"
    assert current_state["agenda_window_focus"] == "next_same_group_window"
    assert current_state["agenda_window_bias"] == 0.17
    assert current_state["agenda_window_reason"] == "wait_for_group_thread"
    assert current_state["agenda_window_carry_target"] == "same_group_window"


def test_build_inner_os_sleep_snapshot_carries_learning_and_social_experiment_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.0, "enthalpy": 0.3},
        nightly_summary={
            "inner_os_sleep_learning_mode_focus": "repair_probe",
            "inner_os_sleep_learning_mode_carry_bias": 0.15,
            "inner_os_sleep_social_experiment_focus": "repair_signal_probe",
            "inner_os_sleep_social_experiment_carry_bias": 0.13,
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["learning_mode_focus"] == "repair_probe"
    assert current_state["learning_mode_carry_bias"] == 0.15
    assert current_state["social_experiment_focus"] == "repair_signal_probe"
    assert current_state["social_experiment_carry_bias"] == 0.13


def test_build_inner_os_sleep_snapshot_carries_temporal_membrane_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 4.0, "enthalpy": 0.3},
        nightly_summary={
            "inner_os_sleep_temporal_membrane_focus": "reentry",
            "inner_os_sleep_temporal_timeline_bias": 0.12,
            "inner_os_sleep_temporal_reentry_bias": 0.17,
            "inner_os_sleep_temporal_supersession_bias": 0.04,
            "inner_os_sleep_temporal_continuity_bias": 0.11,
            "inner_os_sleep_temporal_relation_reentry_bias": 0.09,
        },
        memory_inventory={"l1_count": 8, "l2_count": 4, "l3_count": 1, "recent_experiences": 4},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["temporal_membrane_focus"] == "reentry"
    assert current_state["temporal_timeline_bias"] == 0.12
    assert current_state["temporal_reentry_bias"] == 0.17
    assert current_state["temporal_supersession_bias"] == 0.04
    assert current_state["temporal_continuity_bias"] == 0.11
    assert current_state["temporal_relation_reentry_bias"] == 0.09


def test_build_inner_os_sleep_snapshot_carries_temperament_sleep_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 3.8, "enthalpy": 0.28},
        nightly_summary={
            "inner_os_sleep_temperament_focus": "forward",
            "inner_os_sleep_temperament_forward_bias": 0.11,
            "inner_os_sleep_temperament_guard_bias": 0.03,
            "inner_os_sleep_temperament_bond_bias": 0.05,
            "inner_os_sleep_temperament_recovery_bias": 0.02,
        },
        memory_inventory={"l1_count": 6, "l2_count": 3, "l3_count": 1, "recent_experiences": 3},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["temperament_focus"] == "forward"
    assert current_state["temperament_forward_bias"] == 0.11
    assert current_state["temperament_guard_bias"] == 0.03
    assert current_state["temperament_bond_bias"] == 0.05
    assert current_state["temperament_recovery_bias"] == 0.02


def test_build_inner_os_sleep_snapshot_carries_body_and_relational_sleep_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 3.9, "enthalpy": 0.29},
        nightly_summary={
            "inner_os_sleep_body_homeostasis_focus": "recovering",
            "inner_os_sleep_body_homeostasis_carry_bias": 0.15,
            "inner_os_sleep_relational_continuity_focus": "reopening",
            "inner_os_sleep_relational_continuity_carry_bias": 0.12,
            "inner_os_sleep_group_thread_focus": "threaded_group",
            "inner_os_sleep_group_thread_carry_bias": 0.1,
        },
        memory_inventory={"l1_count": 6, "l2_count": 3, "l3_count": 1, "recent_experiences": 3},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["body_homeostasis_focus"] == "recovering"
    assert current_state["body_homeostasis_carry_bias"] == 0.15
    assert current_state["relational_continuity_focus"] == "reopening"
    assert current_state["relational_continuity_carry_bias"] == 0.12
    assert current_state["group_thread_focus"] == "threaded_group"
    assert current_state["group_thread_carry_bias"] == 0.1


def test_build_inner_os_sleep_snapshot_carries_expressive_style_sleep_bias_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 3.9, "enthalpy": 0.29},
        nightly_summary={
            "inner_os_sleep_expressive_style_focus": "warm_companion",
            "inner_os_sleep_expressive_style_carry_bias": 0.1,
            "inner_os_sleep_expressive_style_history_focus": "warm_companion",
            "inner_os_sleep_expressive_style_history_bias": 0.08,
            "inner_os_sleep_banter_style_focus": "gentle_tease",
            "inner_os_sleep_lexical_variation_carry_bias": 0.11,
        },
        memory_inventory={"l1_count": 6, "l2_count": 3, "l3_count": 1, "recent_experiences": 3},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["expressive_style_focus"] == "warm_companion"
    assert current_state["expressive_style_carry_bias"] == 0.1
    assert current_state["expressive_style_history_focus"] == "warm_companion"
    assert current_state["expressive_style_history_bias"] == 0.08
    assert current_state["banter_style_focus"] == "gentle_tease"
    assert current_state["lexical_variation_carry_bias"] == 0.11


def test_build_inner_os_sleep_snapshot_carries_homeostasis_budget_into_current_state() -> None:
    result = build_inner_os_sleep_snapshot(
        rest_state={"active": False, "history": []},
        latest_field_metrics={"entropy": 3.9, "enthalpy": 0.29},
        nightly_summary={
            "inner_os_sleep_homeostasis_budget_focus": "recovering",
            "inner_os_sleep_homeostasis_budget_bias": 0.13,
        },
        memory_inventory={"l1_count": 6, "l2_count": 3, "l3_count": 1, "recent_experiences": 3},
    )

    current_state = result["derived_inputs"]["current_state"]
    assert current_state["homeostasis_budget_focus"] == "recovering"
    assert current_state["homeostasis_budget_bias"] == 0.13
