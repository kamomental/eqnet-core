import json
from pathlib import Path

import jsonschema

from ops import nightly


def test_nightly_json_matches_schema(tmp_path) -> None:
    """Validate that the JSON payload produced by nightly matches the schema."""
    report = {
        "config_snapshot": {
            "ignition": {"theta_on": 0.62, "theta_off": 0.48, "dwell_steps": 8},
            "telemetry": {"log_path": "telemetry/ignition-%Y%m%d.jsonl"},
        },
        "field_state": {
            "S_mean": 0.42,
            "H_mean": 0.51,
            "rho_mean": 0.48,
            "Ignition_mean": 0.46,
            "valence_mean": -0.03,
            "arousal_mean": 0.12,
            "rho_I_corr": 0.27,
            "S_I_corr": -0.13,
            "H_I_corr": 0.05,
            "valence_I_corr": 0.18,
            "arousal_I_corr": 0.21,
        },
        "plots": {
            "ignition_timeseries": "reports/plots/ignition_timeseries.png",
            "rho_vs_I_scatter": "reports/plots/rho_vs_I_scatter.png",
            "memory_graph": "reports/plots/memory_graph.png",
            "affective_map": "reports/plots/affective_map.png",
            "culture_resonance": "reports/plots/culture_resonance.png",
            "culture_trend": "reports/plots/culture_trend.png",
            "resonance_objective": "reports/plots/resonance_objective.png",
            "resonance_bayes_trace": "reports/plots/resonance_bayes_trace.png",
            "vision_counts": "reports/plots/vision_counts.png",
            "vision_pose": "reports/plots/vision_pose.png",
        },
        "tuning_suggestion": {
            "reason": "rho/I correlation low (0.270)",
            "theta_on": {"current": 0.62, "suggested": 0.6},
            "theta_off": {"current": 0.48, "suggested": 0.46},
        },
        "warnings": ["demo warning"],
        "affective_stats": {
            "valence": {"mean": 0.1, "std": 0.02, "q25": 0.05, "q50": 0.1, "q75": 0.15, "count": 200},
            "arousal": {"mean": 0.2, "std": 0.03, "q25": 0.1, "q50": 0.2, "q75": 0.25, "count": 200}
        },
        "vision_snapshot": {
            "events": 24,
            "detections_total": 96,
            "counts_by_kind": {"handshake": 12, "person": 50},
            "mean_valence": 0.08,
            "mean_arousal": 0.22,
            "pose_mean": {"yaw_mean": -1.2, "pitch_mean": 0.4, "roll_mean": 0.1},
            "fps_mean": 29.8,
            "ts_last": 1_730_000_000_000,
        },
        "resonance": {
            "pairs": [
                {
                    "agents": ["A", "B"],
                    "rho_corr": 0.45,
                    "rho_corr_ci95": [0.2, 0.7],
                    "rho_cross_corr_peak": 0.5,
                    "rho_cross_corr_lag": 2,
                    "rho_cross_corr_lag_refined": 1.8,
                    "energy": 0.03,
                    "n_eff": 120,
                    "objective": 0.41,
                    "partial_corr": 0.4,
                    "granger": {"lag": 1, "a_to_b_f": 2.1, "b_to_a_f": 0.9},
                }
            ],
            "partial_corr": 0.4,
            "granger": {"lag": 1, "a_to_b_f": 2.1, "b_to_a_f": 0.9}
        },
        "culture_stats": {
            "JP_basic": {
                "count": 120.0,
                "mean_valence": 0.12,
                "mean_arousal": 0.35,
                "mean_rho": 0.48,
                "mean_politeness": 0.7,
                "mean_intimacy": 0.3,
            }
        },
        "resonance_history_path": "reports/resonance_history.jsonl",
        "resonance_bayes_trace_path": "reports/resonance_bayes_trace.jsonl",
        "culture_history_path": "reports/culture_history.jsonl",
        "policy_feedback": {
            "enabled": True,
            "politeness_before": 0.5,
            "politeness_after": 0.52,
            "delta": 0.02,
            "intimacy_before": 0.48,
            "intimacy_after": 0.49,
            "intimacy_delta": 0.01,
            "corr": 0.68,
            "corr_source": "resonance_summary",
            "reason": "resonance_high",
            "reference_tag": "JP_basic",
            "reference_rho": 0.52,
            "reference_valence": 0.12,
            "alerts": ["demo_alert"],
            "vision_adjustment": {
                "politeness_delta": 0.01,
                "intimacy_delta": 0.005,
                "details": [
                    {
                        "kind": "handshake",
                        "count": 4,
                        "effective": 0.2,
                        "normalize": "events",
                        "politeness_delta": 0.01,
                        "intimacy_delta": 0.005,
                    }
                ],
                "counts_by_kind": {"handshake": 4},
                "normalisers": {"events": 24, "detections": 96},
            },
        },
        "policy_feedback_history_path": "reports/policy_feedback_history.jsonl",
        "inner_os_sleep_snapshot_path": "reports/nightly_inner_os_sleep.json",
        "inner_os_sleep_mode": "reconsolidate",
        "inner_os_working_memory_snapshot_path": "reports/inner_os_working_memory_snapshot.json",
        "inner_os_working_memory_focus": "meaning",
        "inner_os_working_memory_readiness": 0.64,
        "inner_os_working_memory_replay_bias": {
            "focus": "meaning",
            "anchor": "harbor slope",
            "strength": 0.58,
            "matched_events": 2,
            "boost_mean": 0.041,
            "boost_max": 0.062,
            "top_matches": [
                {"id": "trace-1", "alignment": 0.82, "boost": 0.062},
            ],
        },
        "inner_os_long_term_theme_summary": {
            "focus": "meaning",
            "anchor": "harbor slope",
            "kind": "meaning",
            "summary": "fragile harbor promise",
            "strength": 0.63,
        },
        "inner_os_identity_arc_summary": {
            "summary_version": "v1",
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "dominant_driver": "theme:meaning",
            "supporting_drivers": ["theme:meaning", "memory:bond_protection", "commitment:repair"],
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.58,
            "memory_anchor": "harbor slope",
            "related_person_id": "user",
            "group_thread_focus": "threaded_group:user|friend",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "meaning",
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
        "inner_os_identity_arc_registry_summary": {
            "dominant_arc_id": "repairing_bond::user::meaning::harbor_slope",
            "dominant_arc_kind": "repairing_bond",
            "dominant_arc_phase": "shifting",
            "dominant_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "active_arc_count": 1,
            "total_arcs": 1,
            "top_arc_ids": ["repairing_bond::user::meaning::harbor_slope"],
            "status_counts": {"active": 1},
            "top_arcs": [
                {
                    "arc_id": "repairing_bond::user::meaning::harbor_slope",
                    "arc_kind": "repairing_bond",
                    "phase": "shifting",
                    "status": "active",
                    "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
                    "learning_mode_focus": "repair_probe",
                    "social_experiment_focus": "repair_signal_probe"
                }
            ]
        },
        "inner_os_relation_arc_summary": {
            "summary_version": "v1",
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread",
            "dominant_driver": "person:user",
            "supporting_drivers": ["person:user", "learning:repair_probe"],
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.56,
            "related_person_id": "user",
            "group_thread_id": "threaded_group:user|friend",
            "social_role": "companion",
            "community_id": "harbor_collective",
            "culture_id": "JP_basic",
            "topology_focus": "threaded_group",
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
        "inner_os_group_relation_arc_summary": {
            "summary_version": "v1",
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is moving through a shared group thread in small steps",
            "group_thread_id": "threaded_group:user|friend",
            "topology_focus": "threaded_group",
            "boundary_mode": "same_group_reentry",
            "reentry_window_focus": "next_same_group_window",
            "dominant_person_id": "user",
            "social_role": "companion",
            "community_id": "harbor_collective",
            "culture_id": "JP_basic",
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.61,
        },
        "inner_os_relation_arc_registry_summary": {
            "dominant_arc_id": "repairing_relation::user::companion",
            "dominant_arc_kind": "repairing_relation",
            "dominant_arc_phase": "shifting",
            "dominant_arc_summary": "repair is gathering around a companion thread",
            "dominant_person_id": "user",
            "dominant_group_thread_id": "threaded_group:user|friend",
            "active_arc_count": 1,
            "total_arcs": 1,
            "top_arc_ids": ["repairing_relation::user::companion"],
            "status_counts": {"active": 1},
            "top_arcs": [
                {
                    "arc_id": "repairing_relation::user::companion",
                    "arc_kind": "repairing_relation",
                    "phase": "shifting",
                    "status": "active",
                    "summary": "repair is gathering around a companion thread",
                    "learning_mode_focus": "repair_probe",
                    "social_experiment_focus": "repair_signal_probe"
                }
            ]
        },
        "inner_os_memory_class_summary": {
            "dominant_class": "bond_protection",
            "dominant_reason": "bond_protection_pressure",
            "counts": {"bond_protection": 3, "episodic": 1},
            "weighted_counts": {"bond_protection": 2.4, "episodic": 0.8},
            "recent_records": 4,
            "lookback_hours": 120,
        },
        "inner_os_agenda_summary": {
            "dominant_agenda": "repair",
            "dominant_reason": "repair_trace",
            "state_counts": {"repair": 3, "hold": 1},
            "weighted_state_counts": {"repair": 1.7, "hold": 0.4},
            "agenda_carry_bias": 0.33,
            "recent_records": 4,
            "lookback_hours": 120,
        },
        "inner_os_commitment_summary": {
            "dominant_target": "repair",
            "dominant_state": "commit",
            "dominant_reason": "repair_trace",
            "target_counts": {"repair": 3, "hold": 1},
            "state_counts": {"commit": 2, "settle": 1, "waver": 1},
            "weighted_target_counts": {"repair": 1.7, "hold": 0.4},
            "commitment_carry_bias": 0.34,
            "commitment_followup_focus": "reopen_softly",
            "commitment_mode_focus": "repair",
            "recent_records": 4,
            "lookback_hours": 120,
        },
        "inner_os_insight_summary": {
            "dominant_insight_class": "reframed_relation",
            "dominant_reframed_topic": "harbor thread",
            "insight_class_counts": {"reframed_relation": 2, "insight_trace": 1},
            "weighted_class_counts": {"reframed_relation": 1.8, "insight_trace": 0.6},
            "insight_link_counts": {"bond:user|memory:harbor_thread": 3},
            "association_reweighting_bias": 0.36,
            "association_reweighting_focus": "reframed_links",
            "association_reweighting_reason": "reframed_relation",
            "insight_reframing_bias": 0.28,
            "insight_terrain_shape_bias": 0.16,
            "insight_terrain_shape_reason": "reframed_relation",
            "insight_terrain_shape_target": "soft_relation",
            "insight_anchor_center": [0.16, -0.04, 0.12],
            "insight_anchor_dispersion": 0.27,
            "recent_records": 3,
            "lookback_hours": 120,
        },
        "inner_os_partner_relation_summary": {
            "person_id": "user",
            "summary": "gentle harbor companion thread",
            "memory_anchor": "harbor slope",
            "social_role": "companion",
            "social_interpretation": "familiar:companion:open",
            "address_hint": "companion",
            "timing_hint": "open",
            "stance_hint": "familiar",
            "attachment": 0.72,
            "familiarity": 0.68,
            "trust_memory": 0.7,
            "strength": 0.71,
        },
        "inner_os_partner_relation_registry_summary": {
            "dominant_person_id": "user",
            "top_person_ids": ["user", "friend"],
            "total_people": 2,
            "uncertainty": 0.12,
            "persons": {
                "user": {
                    "person_id": "user",
                    "adaptive_traits": {
                        "attachment": 0.72,
                        "familiarity": 0.68,
                        "trust_memory": 0.7,
                        "continuity_score": 0.66,
                        "social_grounding": 0.61
                    }
                }
            }
        },
        "inner_os_group_thread_registry_summary": {
            "dominant_thread_id": "threaded_group:user|friend",
            "top_thread_ids": ["threaded_group:user|friend"],
            "total_threads": 1,
            "thread_scores": {"threaded_group:user|friend": 0.71},
            "uncertainty": 0.19,
            "threads": {
                "threaded_group:user|friend": {
                    "thread_id": "threaded_group:user|friend",
                    "dominant_person_id": "user",
                    "top_person_ids": ["user", "friend"],
                    "total_people": 2,
                    "continuity_score": 0.66,
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
            "threads": {
                "repair_anchor": {
                    "thread_id": "repair_anchor",
                    "anchor": "repair anchor",
                    "last_issue_state": "pausing_issue",
                    "count": 2,
                    "confidence": 0.48,
                }
            },
        },
        "inner_os_daily_carry_summary": {
            "summary_version": "v1",
            "same_turn_focus": {
                "memory_class": "bond_protection",
                "commitment_target": "repair",
                "commitment_state": "commit",
                "insight_class": "reframed_relation",
                "identity_arc_kind": "repairing_bond",
                "group_thread_dominant_thread": "threaded_group:user|friend",
                "group_thread_total_threads": 1,
            },
            "overnight_focus": {
                "memory_class_focus": "bond_protection",
                "commitment_target_focus": "repair",
                "association_focus": "reframed_links",
                "terrain_shape_target": "soft_relation",
                "temperament_focus": "forward",
                "body_homeostasis_focus": "recovering",
                "relational_continuity_focus": "reopening",
                "group_thread_focus": "threaded_group",
                "identity_arc_kind": "repairing_bond",
                "identity_arc_phase": "shifting",
                "identity_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
                "expressive_style_focus": "warm_companion",
            },
            "carry_strengths": {
                "terrain_reweighting": 0.44,
                "commitment_carry": 0.37,
                "association_reweighting": 0.38,
                "insight_reframing": 0.31,
                "insight_terrain_shape": 0.19,
                "body_homeostasis_carry": 0.14,
                "relational_continuity_carry": 0.11,
                "group_thread_carry": 0.1,
                "temperament_forward": 0.11,
            },
            "active_carry_channels": ["terrain_reweighting", "commitment_carry", "association_reweighting"],
            "dominant_carry_channel": "terrain_reweighting",
            "carry_alignment": {
                "memory_carry_visible": True,
                "commitment_carry_visible": True,
                "insight_carry_visible": True,
                "body_homeostasis_carry_visible": True,
                "relational_continuity_carry_visible": True,
                "group_thread_carry_visible": True,
                "group_thread_registry_visible": True,
                "identity_arc_visible": True,
            },
            "temporal_alignment": {
                "same_turn_mode": "reentry",
                "overnight_focus": "reentry",
                "focus_alignment": True,
                "same_to_overnight_reentry_delta": -0.39,
                "reentry_carry_visible": True,
                "reentry_carry_strength": 0.17,
            },
        },
        "inner_os_temporal_alignment": {
            "same_turn_mode": "reentry",
            "overnight_focus": "reentry",
            "focus_alignment": True,
            "same_to_overnight_reentry_delta": -0.39,
            "reentry_carry_visible": True,
            "reentry_carry_strength": 0.17,
        },
        "inner_os_sleep_memory_class_focus": "bond_protection",
        "inner_os_sleep_terrain_reweighting_bias": 0.44,
        "inner_os_sleep_agenda_focus": "repair",
        "inner_os_sleep_agenda_bias": 0.33,
        "inner_os_sleep_agenda_reason": "repair_trace",
        "inner_os_sleep_agenda_window_focus": "next_private_window",
        "inner_os_sleep_agenda_window_bias": 0.19,
        "inner_os_sleep_agenda_window_reason": "wait_for_private_window",
        "inner_os_sleep_agenda_window_carry_target": "private_window",
        "inner_os_sleep_learning_mode_focus": "repair_probe",
        "inner_os_sleep_learning_mode_carry_bias": 0.15,
        "inner_os_sleep_social_experiment_focus": "repair_signal_probe",
        "inner_os_sleep_social_experiment_carry_bias": 0.13,
        "inner_os_sleep_commitment_target_focus": "repair",
        "inner_os_sleep_commitment_state_focus": "commit",
        "inner_os_sleep_commitment_carry_bias": 0.37,
        "inner_os_sleep_commitment_followup_focus": "reopen_softly",
        "inner_os_sleep_commitment_mode_focus": "repair",
        "inner_os_sleep_commitment_carry_reason": "commit:repair",
        "inner_os_sleep_body_homeostasis_focus": "recovering",
        "inner_os_sleep_body_homeostasis_carry_bias": 0.14,
        "inner_os_sleep_homeostasis_budget_focus": "recovering",
        "inner_os_sleep_homeostasis_budget_bias": 0.09,
        "inner_os_sleep_relational_continuity_focus": "reopening",
        "inner_os_sleep_relational_continuity_carry_bias": 0.11,
        "inner_os_sleep_group_thread_focus": "threaded_group",
        "inner_os_sleep_group_thread_carry_bias": 0.1,
        "inner_os_sleep_autobiographical_thread_mode": "unfinished_thread",
        "inner_os_sleep_autobiographical_thread_anchor": "harbor promise",
        "inner_os_sleep_autobiographical_thread_focus": "unfinished promise",
        "inner_os_sleep_autobiographical_thread_strength": 0.41,
        "inner_os_sleep_temporal_membrane_focus": "reentry",
        "inner_os_sleep_temporal_timeline_bias": 0.12,
        "inner_os_sleep_temporal_reentry_bias": 0.17,
        "inner_os_sleep_temporal_supersession_bias": 0.04,
        "inner_os_sleep_temporal_continuity_bias": 0.11,
        "inner_os_sleep_temporal_relation_reentry_bias": 0.09,
        "inner_os_sleep_expressive_style_focus": "warm_companion",
        "inner_os_sleep_expressive_style_carry_bias": 0.1,
        "inner_os_sleep_expressive_style_history_focus": "warm_companion",
        "inner_os_sleep_expressive_style_history_bias": 0.08,
        "inner_os_sleep_banter_style_focus": "gentle_tease",
        "inner_os_sleep_lexical_variation_carry_bias": 0.11,
        "inner_os_sleep_growth_relational_trust": 0.64,
        "inner_os_sleep_growth_epistemic_maturity": 0.58,
        "inner_os_sleep_growth_expressive_range": 0.62,
        "inner_os_sleep_growth_residue_integration": 0.55,
        "inner_os_sleep_growth_playfulness_range": 0.49,
        "inner_os_sleep_growth_self_coherence": 0.67,
        "inner_os_sleep_growth_dominant_transition": "expressive_range",
        "inner_os_sleep_growth_bond_axis": 0.61,
        "inner_os_sleep_growth_stability_axis": 0.6,
        "inner_os_sleep_growth_curiosity_axis": 0.57,
        "inner_os_sleep_memory_dynamics_mode": "reconsolidate",
        "inner_os_sleep_memory_dominant_link": "bond:user|memory:harbor_thread",
        "inner_os_sleep_memory_monument_kind": "shared_ritual",
        "inner_os_sleep_memory_monument_salience": 0.66,
        "inner_os_sleep_memory_ignition_readiness": 0.52,
        "inner_os_sleep_memory_consolidation_pull": 0.64,
        "inner_os_sleep_memory_tension": 0.22,
        "inner_os_sleep_memory_topology_axis": 0.57,
        "inner_os_sleep_memory_salience_axis": 0.66,
        "inner_os_sleep_memory_ignition_axis": 0.52,
        "inner_os_sleep_memory_consolidation_axis": 0.64,
        "inner_os_sleep_memory_tension_axis": 0.22,
        "inner_os_sleep_insight_class_focus": "reframed_relation",
        "inner_os_sleep_insight_reframing_bias": 0.31,
        "inner_os_sleep_association_reweighting_bias": 0.38,
        "inner_os_sleep_association_reweighting_focus": "reframed_links",
        "inner_os_sleep_association_reweighting_reason": "reframed_relation",
        "inner_os_sleep_insight_terrain_shape_bias": 0.19,
        "inner_os_sleep_insight_terrain_shape_reason": "reframed_relation",
        "inner_os_sleep_insight_terrain_shape_target": "soft_relation",
        "inner_os_sleep_insight_anchor_center": [0.14, -0.02, 0.11],
        "inner_os_sleep_insight_anchor_dispersion": 0.24,
        "inner_os_sleep_temperament_focus": "forward",
        "inner_os_sleep_temperament_forward_bias": 0.11,
        "inner_os_sleep_temperament_guard_bias": 0.03,
        "inner_os_sleep_temperament_bond_bias": 0.05,
        "inner_os_sleep_temperament_recovery_bias": 0.02,
        "alerts": ["demo_alert"],
        "alerts_detail": [
            {"kind": "culture.high_abs_valence", "tag": "JP_basic", "value": 0.12, "threshold": 0.6}
        ],
        "run_seed": 20251027,
    }
    out_dir = tmp_path / "reports"
    json_path = nightly._write_json_summary(report, out_dir=str(out_dir))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(Path("schema/nightly.v1.json").read_text(encoding="utf-8"))

    jsonschema.validate(instance=payload, schema=schema)
    assert payload["schema"] == "nightly.v1"
    assert payload["plots"]["memory_graph"].endswith("memory_graph.png")
    assert payload["plots"]["affective_map"].endswith("affective_map.png")
    assert payload["plots"]["culture_resonance"].endswith("culture_resonance.png")
    assert payload["plots"]["culture_trend"].endswith("culture_trend.png")
    assert "valence" in payload.get("affective_stats", {})
    assert "vision_snapshot" in payload
    assert "resonance" in payload
    assert payload.get("resonance_history_path")
    assert payload.get("resonance_bayes_trace_path")
    assert "culture_stats" in payload
    assert payload.get("culture_history_path")
    assert payload.get("alerts_detail")
    assert payload.get("policy_feedback")
    assert payload.get("policy_feedback_history_path")
    assert payload.get("inner_os_sleep_snapshot_path")
    assert payload.get("inner_os_sleep_mode") == "reconsolidate"
    assert payload.get("inner_os_working_memory_snapshot_path")
    assert payload.get("inner_os_working_memory_focus") == "meaning"
    assert payload.get("inner_os_working_memory_readiness") == 0.64
    assert payload.get("inner_os_working_memory_replay_bias", {}).get("anchor") == "harbor slope"
    assert payload.get("inner_os_working_memory_replay_bias", {}).get("matched_events") == 2
    assert payload.get("inner_os_long_term_theme_summary", {}).get("summary") == "fragile harbor promise"
    assert payload.get("inner_os_identity_arc_summary", {}).get("arc_kind") == "repairing_bond"
    assert payload.get("inner_os_identity_arc_registry_summary", {}).get("dominant_arc_kind") == "repairing_bond"
    assert payload.get("inner_os_memory_class_summary", {}).get("dominant_class") == "bond_protection"
    assert payload.get("inner_os_agenda_summary", {}).get("dominant_agenda") == "repair"
    assert payload.get("inner_os_commitment_summary", {}).get("dominant_target") == "repair"
    assert payload.get("inner_os_insight_summary", {}).get("dominant_insight_class") == "reframed_relation"
    assert payload.get("inner_os_partner_relation_summary", {}).get("person_id") == "user"
    assert payload.get("inner_os_partner_relation_registry_summary", {}).get("total_people") == 2
    assert payload.get("inner_os_group_thread_registry_summary", {}).get("total_threads") == 1
    assert payload.get("inner_os_daily_carry_summary", {}).get("same_turn_focus", {}).get("memory_class") == "bond_protection"
    assert payload.get("inner_os_daily_carry_summary", {}).get("overnight_focus", {}).get("commitment_target_focus") == "repair"
    assert payload.get("inner_os_daily_carry_summary", {}).get("overnight_focus", {}).get("group_thread_focus") == "threaded_group"
    assert payload.get("inner_os_daily_carry_summary", {}).get("temporal_alignment", {}).get("focus_alignment") is True
    assert payload.get("inner_os_temporal_alignment", {}).get("same_turn_mode") == "reentry"
    assert payload.get("inner_os_temporal_alignment", {}).get("reentry_carry_strength") == 0.17
    assert payload.get("inner_os_sleep_memory_class_focus") == "bond_protection"
    assert payload.get("inner_os_sleep_terrain_reweighting_bias") == 0.44
    assert payload.get("inner_os_sleep_agenda_focus") == "repair"
    assert payload.get("inner_os_sleep_agenda_bias") == 0.33
    assert payload.get("inner_os_sleep_agenda_reason") == "repair_trace"
    assert payload.get("inner_os_sleep_agenda_window_focus") == "next_private_window"
    assert payload.get("inner_os_sleep_agenda_window_bias") == 0.19
    assert payload.get("inner_os_sleep_agenda_window_reason") == "wait_for_private_window"
    assert payload.get("inner_os_sleep_agenda_window_carry_target") == "private_window"
    assert payload.get("inner_os_sleep_learning_mode_focus") == "repair_probe"
    assert payload.get("inner_os_sleep_learning_mode_carry_bias") == 0.15
    assert payload.get("inner_os_sleep_social_experiment_focus") == "repair_signal_probe"
    assert payload.get("inner_os_sleep_social_experiment_carry_bias") == 0.13
    assert payload.get("inner_os_sleep_commitment_target_focus") == "repair"
    assert payload.get("inner_os_sleep_commitment_state_focus") == "commit"
    assert payload.get("inner_os_sleep_commitment_carry_bias") == 0.37
    assert payload.get("inner_os_sleep_commitment_followup_focus") == "reopen_softly"
    assert payload.get("inner_os_sleep_commitment_mode_focus") == "repair"
    assert payload.get("inner_os_sleep_commitment_carry_reason") == "commit:repair"
    assert payload.get("inner_os_sleep_body_homeostasis_focus") == "recovering"
    assert payload.get("inner_os_sleep_body_homeostasis_carry_bias") == 0.14
    assert payload.get("inner_os_sleep_homeostasis_budget_focus") == "recovering"
    assert payload.get("inner_os_sleep_homeostasis_budget_bias") == 0.09
    assert payload.get("inner_os_sleep_relational_continuity_focus") == "reopening"
    assert payload.get("inner_os_sleep_relational_continuity_carry_bias") == 0.11
    assert payload.get("inner_os_sleep_group_thread_focus") == "threaded_group"
    assert payload.get("inner_os_sleep_group_thread_carry_bias") == 0.1
    assert payload.get("inner_os_sleep_autobiographical_thread_mode") == "unfinished_thread"
    assert payload.get("inner_os_sleep_autobiographical_thread_anchor") == "harbor promise"
    assert payload.get("inner_os_sleep_autobiographical_thread_focus") == "unfinished promise"
    assert payload.get("inner_os_sleep_autobiographical_thread_strength") == 0.41
    assert payload.get("inner_os_sleep_temporal_membrane_focus") == "reentry"
    assert payload.get("inner_os_sleep_temporal_timeline_bias") == 0.12
    assert payload.get("inner_os_sleep_temporal_reentry_bias") == 0.17
    assert payload.get("inner_os_sleep_temporal_supersession_bias") == 0.04
    assert payload.get("inner_os_sleep_temporal_continuity_bias") == 0.11
    assert payload.get("inner_os_sleep_temporal_relation_reentry_bias") == 0.09
    assert payload.get("inner_os_sleep_expressive_style_focus") == "warm_companion"
    assert payload.get("inner_os_sleep_expressive_style_carry_bias") == 0.1
    assert payload.get("inner_os_sleep_expressive_style_history_focus") == "warm_companion"
    assert payload.get("inner_os_sleep_expressive_style_history_bias") == 0.08
    assert payload.get("inner_os_sleep_banter_style_focus") == "gentle_tease"
    assert payload.get("inner_os_sleep_lexical_variation_carry_bias") == 0.11
    assert payload.get("inner_os_sleep_growth_relational_trust") == 0.64
    assert payload.get("inner_os_sleep_growth_epistemic_maturity") == 0.58
    assert payload.get("inner_os_sleep_growth_expressive_range") == 0.62
    assert payload.get("inner_os_sleep_growth_residue_integration") == 0.55
    assert payload.get("inner_os_sleep_growth_playfulness_range") == 0.49
    assert payload.get("inner_os_sleep_growth_self_coherence") == 0.67
    assert payload.get("inner_os_sleep_growth_dominant_transition") == "expressive_range"
    assert payload.get("inner_os_sleep_growth_bond_axis") == 0.61
    assert payload.get("inner_os_sleep_growth_stability_axis") == 0.6
    assert payload.get("inner_os_sleep_growth_curiosity_axis") == 0.57
    assert payload.get("inner_os_sleep_memory_dynamics_mode") == "reconsolidate"
    assert payload.get("inner_os_sleep_memory_dominant_link") == "bond:user|memory:harbor_thread"
    assert payload.get("inner_os_sleep_memory_monument_kind") == "shared_ritual"
    assert payload.get("inner_os_sleep_memory_monument_salience") == 0.66
    assert payload.get("inner_os_sleep_memory_ignition_readiness") == 0.52
    assert payload.get("inner_os_sleep_memory_consolidation_pull") == 0.64
    assert payload.get("inner_os_sleep_memory_tension") == 0.22
    assert payload.get("inner_os_sleep_memory_topology_axis") == 0.57
    assert payload.get("inner_os_sleep_memory_salience_axis") == 0.66
    assert payload.get("inner_os_sleep_memory_ignition_axis") == 0.52
    assert payload.get("inner_os_sleep_memory_consolidation_axis") == 0.64
    assert payload.get("inner_os_sleep_memory_tension_axis") == 0.22
    assert payload.get("inner_os_sleep_insight_class_focus") == "reframed_relation"
    assert payload.get("inner_os_sleep_insight_reframing_bias") == 0.31
    assert payload.get("inner_os_sleep_association_reweighting_bias") == 0.38
    assert payload.get("inner_os_sleep_association_reweighting_focus") == "reframed_links"
    assert payload.get("inner_os_sleep_association_reweighting_reason") == "reframed_relation"
    assert payload.get("inner_os_sleep_insight_terrain_shape_bias") == 0.19
    assert payload.get("inner_os_sleep_insight_terrain_shape_reason") == "reframed_relation"
    assert payload.get("inner_os_sleep_insight_terrain_shape_target") == "soft_relation"
    assert payload.get("inner_os_sleep_insight_anchor_center") == [0.14, -0.02, 0.11]
    assert payload.get("inner_os_sleep_insight_anchor_dispersion") == 0.24
    assert payload.get("inner_os_sleep_temperament_focus") == "forward"
    assert payload.get("inner_os_sleep_temperament_forward_bias") == 0.11
    assert payload.get("inner_os_sleep_temperament_guard_bias") == 0.03
    assert payload.get("inner_os_sleep_temperament_bond_bias") == 0.05
    assert payload.get("inner_os_sleep_temperament_recovery_bias") == 0.02
    assert "vision_adjustment" in payload["policy_feedback"]
    normalisers = payload["policy_feedback"]["vision_adjustment"].get("normalisers", {})
    assert "events" in normalisers
