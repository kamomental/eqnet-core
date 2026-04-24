from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

from inner_os import SleepConsolidationCore
from inner_os.development_transition_policy import derive_growth_state
from inner_os.memory_dynamics import derive_memory_dynamics_state
from inner_os.schemas import INNER_OS_SLEEP_CONSOLIDATION_SCHEMA


def build_inner_os_sleep_snapshot(
    *,
    rest_state: Optional[Mapping[str, Any]] = None,
    latest_field_metrics: Optional[Mapping[str, Any]] = None,
    nightly_summary: Optional[Mapping[str, Any]] = None,
    memory_inventory: Optional[Mapping[str, Any]] = None,
    current_state: Optional[Mapping[str, Any]] = None,
    development_state: Optional[Mapping[str, Any]] = None,
    persistence_state: Optional[Mapping[str, Any]] = None,
    personality_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Wrap legacy nightly/rest signals into the reusable inner_os planner."""

    derived_current = _derive_current_state(
        rest_state=rest_state,
        latest_field_metrics=latest_field_metrics,
        nightly_summary=nightly_summary,
        memory_inventory=memory_inventory,
    )
    derived_forgetting = _derive_forgetting_snapshot(
        nightly_summary=nightly_summary,
        derived_current=derived_current,
    )
    derived_memory = _derive_memory_orchestration(
        nightly_summary=nightly_summary,
        memory_inventory=memory_inventory,
        latest_field_metrics=latest_field_metrics,
        forgetting_snapshot=derived_forgetting,
    )

    merged_current = dict(derived_current)
    merged_current.update(dict(current_state or {}))

    snapshot = SleepConsolidationCore().snapshot(
        current_state=merged_current,
        forgetting_snapshot=derived_forgetting,
        memory_orchestration=derived_memory,
        development_state=development_state,
        persistence_state=persistence_state,
        personality_state=personality_state,
    )
    previous_growth = (dict(current_state or {})).get("growth_state")
    previous_memory_dynamics = (dict(current_state or {})).get("memory_dynamics_state")
    growth_state = derive_growth_state(
        previous_growth=previous_growth if isinstance(previous_growth, Mapping) else None,
        development_state=development_state,
        forgetting_snapshot=derived_forgetting,
        sleep_consolidation=snapshot.to_dict(),
    )
    memory_dynamics_state = derive_memory_dynamics_state(
        previous_state=previous_memory_dynamics if isinstance(previous_memory_dynamics, Mapping) else None,
        memory_orchestration=derived_memory,
        association_graph=merged_current.get("association_graph_state"),
        forgetting_snapshot=derived_forgetting,
        sleep_consolidation=snapshot.to_dict(),
        recall_active=False,
    )
    snapshot = replace(
        snapshot,
        growth_state=growth_state.to_dict(),
        growth_replay_axes=growth_state.to_replay_axes(
            previous_growth if isinstance(previous_growth, Mapping) else None,
        ),
        memory_dynamics_state=memory_dynamics_state.to_dict(),
        memory_dynamics_axes=memory_dynamics_state.to_packet_axes(
            previous_memory_dynamics if isinstance(previous_memory_dynamics, Mapping) else None,
        ),
    )
    return {
        "schema": INNER_OS_SLEEP_CONSOLIDATION_SCHEMA,
        "snapshot": snapshot.to_dict(),
        "derived_inputs": {
            "current_state": merged_current,
            "forgetting_snapshot": derived_forgetting,
            "memory_orchestration": derived_memory,
        },
    }


def build_inner_os_sleep_snapshot_for_system(
    system: Any,
    *,
    nightly_summary: Optional[Mapping[str, Any]] = None,
    current_state: Optional[Mapping[str, Any]] = None,
    development_state: Optional[Mapping[str, Any]] = None,
    persistence_state: Optional[Mapping[str, Any]] = None,
    personality_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    rest_state = _call_mapping(system, "rest_state")
    latest_field_metrics = _latest_field_metrics(_call_value(system, "field_metrics_state"))
    memory_inventory = _memory_inventory_from_system(system)
    return build_inner_os_sleep_snapshot(
        rest_state=rest_state,
        latest_field_metrics=latest_field_metrics,
        nightly_summary=nightly_summary,
        memory_inventory=memory_inventory,
        current_state=current_state,
        development_state=development_state,
        persistence_state=persistence_state,
        personality_state=personality_state,
    )


def write_inner_os_sleep_snapshot_for_system(
    system: Any,
    *,
    out_path: str | Path,
    nightly_summary: Optional[Mapping[str, Any]] = None,
    current_state: Optional[Mapping[str, Any]] = None,
    development_state: Optional[Mapping[str, Any]] = None,
    persistence_state: Optional[Mapping[str, Any]] = None,
    personality_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = build_inner_os_sleep_snapshot_for_system(
        system,
        nightly_summary=nightly_summary,
        current_state=current_state,
        development_state=development_state,
        persistence_state=persistence_state,
        personality_state=personality_state,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _derive_current_state(
    *,
    rest_state: Optional[Mapping[str, Any]],
    latest_field_metrics: Optional[Mapping[str, Any]],
    nightly_summary: Optional[Mapping[str, Any]],
    memory_inventory: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    rest_active = bool((rest_state or {}).get("active"))
    fatigue_streak = _float_from(rest_state, "fatigue_streak", 0.0)
    history = list((rest_state or {}).get("history") or [])
    recent_history = history[-4:]

    fatigue_hits = 0.0
    loop_hits = 0.0
    overload_hits = 0.0
    for row in recent_history:
        triggers = row.get("triggers") if isinstance(row, Mapping) else {}
        if not isinstance(triggers, Mapping):
            triggers = {}
        fatigue_hits += 1.0 if triggers.get("fatigue") else 0.0
        loop_hits += 1.0 if triggers.get("loop") else 0.0
        overload_hits += 1.0 if triggers.get("overload") else 0.0

    entropy = _float_from(latest_field_metrics, "entropy", 0.0)
    enthalpy = _float_from(
        latest_field_metrics,
        "enthalpy",
        _float_from(latest_field_metrics, "enthalpy_mean", 0.0),
    )
    entropy_norm = _clamp01(entropy / 10.0)
    enthalpy_norm = _clamp01(enthalpy)

    l1_count = _float_from(memory_inventory, "l1_count", 0.0)
    l2_count = _float_from(memory_inventory, "l2_count", 0.0)
    l3_count = _float_from(memory_inventory, "l3_count", 0.0)
    recent_experiences = _float_from(memory_inventory, "recent_experiences", l1_count)

    culture_stats = (nightly_summary or {}).get("culture_stats") or {}
    partner_relation = (nightly_summary or {}).get("inner_os_partner_relation_summary") or {}
    partner_registry = (nightly_summary or {}).get("inner_os_partner_relation_registry_summary") or {}
    memory_class_summary = (nightly_summary or {}).get("inner_os_memory_class_summary") or {}
    commitment_summary = (nightly_summary or {}).get("inner_os_commitment_summary") or {}
    insight_summary = (nightly_summary or {}).get("inner_os_insight_summary") or {}
    culture_resonance = _culture_resonance_from_stats(culture_stats)
    policy_feedback = (nightly_summary or {}).get("policy_feedback") or {}
    institutional_memory = _clamp01(
        (0.5 if policy_feedback.get("enabled") else 0.0)
        + _float_from(policy_feedback, "intimacy_after", 0.0) * 0.2
    )
    ritual_memory = _clamp01(culture_resonance * 0.7 + (0.18 if culture_stats else 0.0))

    recovery_need = _clamp01(
        (0.34 if rest_active else 0.0)
        + _clamp01(fatigue_streak / 3.0) * 0.28
        + entropy_norm * 0.18
        + enthalpy_norm * 0.12
        + _clamp01(overload_hits / 4.0) * 0.08
    )
    roughness_dwell = _clamp01(
        _clamp01(fatigue_hits / 4.0) * 0.24
        + _clamp01(loop_hits / 4.0) * 0.28
        + _clamp01(overload_hits / 4.0) * 0.24
        + entropy_norm * 0.16
        + enthalpy_norm * 0.08
    )
    defensive_dwell = _clamp01(
        _clamp01(loop_hits / 4.0) * 0.28
        + _clamp01(overload_hits / 4.0) * 0.26
        + (0.16 if rest_active else 0.0)
        + enthalpy_norm * 0.14
        + _clamp01(fatigue_streak / 3.0) * 0.08
    )
    conscious_mosaic_density = _clamp01(
        _clamp01(recent_experiences / 24.0) * 0.48
        + _clamp01(l2_count / 24.0) * 0.22
        + _clamp01(l3_count / 16.0) * 0.1
        + _clamp01(len(recent_history) / 4.0) * 0.08
    )
    conscious_mosaic_recentness = _clamp01(
        _clamp01(recent_experiences / max(1.0, l1_count + 1.0)) * 0.66
        + (0.12 if recent_history else 0.0)
        + (0.08 if rest_active else 0.0)
    )
    registry_persons = dict(partner_registry.get("persons") or {}) if isinstance(partner_registry, Mapping) else {}
    registry_top_person_ids = [
        str(item).strip()
        for item in list(partner_registry.get("top_person_ids") or [])
        if str(item).strip()
    ] if isinstance(partner_registry, Mapping) else []
    if not registry_top_person_ids and isinstance(partner_registry, Mapping):
        dominant_person_id = _text_from(partner_registry, "dominant_person_id")
        if dominant_person_id:
            registry_top_person_ids = [dominant_person_id]
    registry_primary = {}
    if registry_top_person_ids:
        registry_primary = dict(registry_persons.get(registry_top_person_ids[0]) or {})
    registry_adaptive = dict(registry_primary.get("adaptive_traits") or {}) if isinstance(registry_primary, Mapping) else {}
    related_person_id = _text_from(partner_relation, "person_id") or _text_from(partner_registry, "dominant_person_id")
    attachment = _float_from(partner_relation, "attachment", _float_from(registry_adaptive, "attachment", 0.0))
    familiarity = _float_from(partner_relation, "familiarity", _float_from(registry_adaptive, "familiarity", 0.0))
    trust_memory = _float_from(partner_relation, "trust_memory", _float_from(registry_adaptive, "trust_memory", 0.0))
    relation_seed_strength = _float_from(partner_relation, "strength", _float_from(registry_primary, "strength", 0.0))
    partner_address_hint = _text_from(partner_relation, "address_hint") or _text_from(registry_primary, "address_hint")
    partner_timing_hint = _text_from(partner_relation, "timing_hint") or _text_from(registry_primary, "timing_hint")
    partner_stance_hint = _text_from(partner_relation, "stance_hint") or _text_from(registry_primary, "stance_hint")
    partner_social_interpretation = _text_from(partner_relation, "social_interpretation") or _text_from(registry_primary, "social_interpretation")
    memory_write_class = _text_from(memory_class_summary, "dominant_class") or _text_from(nightly_summary, "inner_os_sleep_memory_class_focus")
    memory_write_class_counts = (
        dict(memory_class_summary.get("counts") or {})
        if isinstance(memory_class_summary, Mapping)
        else {}
    )
    agenda_summary = (nightly_summary or {}).get("inner_os_agenda_summary") or {}
    agenda_focus = _text_from(nightly_summary, "inner_os_sleep_agenda_focus") or _text_from(
        agenda_summary,
        "dominant_agenda",
    )
    agenda_bias = _float_from(
        nightly_summary,
        "inner_os_sleep_agenda_bias",
        _float_from(agenda_summary, "agenda_carry_bias", 0.0),
    )
    agenda_reason = _text_from(nightly_summary, "inner_os_sleep_agenda_reason") or _text_from(
        agenda_summary,
        "dominant_reason",
    )
    agenda_window_focus = _text_from(nightly_summary, "inner_os_sleep_agenda_window_focus")
    agenda_window_bias = _float_from(nightly_summary, "inner_os_sleep_agenda_window_bias", 0.0)
    agenda_window_reason = _text_from(nightly_summary, "inner_os_sleep_agenda_window_reason")
    agenda_window_carry_target = _text_from(nightly_summary, "inner_os_sleep_agenda_window_carry_target")
    learning_mode_focus = _text_from(nightly_summary, "inner_os_sleep_learning_mode_focus")
    learning_mode_carry_bias = _float_from(nightly_summary, "inner_os_sleep_learning_mode_carry_bias", 0.0)
    social_experiment_focus = _text_from(nightly_summary, "inner_os_sleep_social_experiment_focus")
    social_experiment_carry_bias = _float_from(nightly_summary, "inner_os_sleep_social_experiment_carry_bias", 0.0)
    commitment_target_focus = _text_from(nightly_summary, "inner_os_sleep_commitment_target_focus") or _text_from(
        commitment_summary,
        "dominant_target",
    )
    commitment_state_focus = _text_from(nightly_summary, "inner_os_sleep_commitment_state_focus") or _text_from(
        commitment_summary,
        "dominant_state",
    )
    commitment_target_counts = (
        dict(commitment_summary.get("target_counts") or {})
        if isinstance(commitment_summary, Mapping)
        else {}
    )
    commitment_carry_bias = _float_from(
        nightly_summary,
        "inner_os_sleep_commitment_carry_bias",
        _float_from(commitment_summary, "commitment_carry_bias", 0.0),
    )
    commitment_followup_focus = _text_from(nightly_summary, "inner_os_sleep_commitment_followup_focus") or _text_from(
        commitment_summary,
        "commitment_followup_focus",
    )
    commitment_mode_focus = _text_from(nightly_summary, "inner_os_sleep_commitment_mode_focus") or _text_from(
        commitment_summary,
        "commitment_mode_focus",
    )
    commitment_carry_reason = _text_from(nightly_summary, "inner_os_sleep_commitment_carry_reason") or _text_from(
        commitment_summary,
        "dominant_reason",
    )
    terrain_reweighting_bias = _float_from(nightly_summary, "inner_os_sleep_terrain_reweighting_bias", 0.0)
    insight_class_focus = _text_from(insight_summary, "dominant_insight_class") or _text_from(nightly_summary, "inner_os_sleep_insight_class_focus")
    insight_link_counts = (
        dict(insight_summary.get("insight_link_counts") or {})
        if isinstance(insight_summary, Mapping)
        else {}
    )
    insight_class_counts = (
        dict(insight_summary.get("insight_class_counts") or {})
        if isinstance(insight_summary, Mapping)
        else {}
    )
    insight_reframing_bias = _float_from(nightly_summary, "inner_os_sleep_insight_reframing_bias", _float_from(insight_summary, "insight_reframing_bias", 0.0))
    association_reweighting_bias = _float_from(nightly_summary, "inner_os_sleep_association_reweighting_bias", _float_from(insight_summary, "association_reweighting_bias", 0.0))
    association_reweighting_focus = _text_from(nightly_summary, "inner_os_sleep_association_reweighting_focus") or _text_from(
        insight_summary,
        "association_reweighting_focus",
    )
    association_reweighting_reason = _text_from(nightly_summary, "inner_os_sleep_association_reweighting_reason") or _text_from(
        insight_summary,
        "association_reweighting_reason",
    )
    insight_terrain_shape_bias = _float_from(
        nightly_summary,
        "inner_os_sleep_insight_terrain_shape_bias",
        _float_from(insight_summary, "insight_terrain_shape_bias", 0.0),
    )
    insight_terrain_shape_reason = _text_from(nightly_summary, "inner_os_sleep_insight_terrain_shape_reason") or _text_from(
        insight_summary,
        "insight_terrain_shape_reason",
    )
    insight_terrain_shape_target = _text_from(nightly_summary, "inner_os_sleep_insight_terrain_shape_target") or _text_from(
        insight_summary,
        "insight_terrain_shape_target",
    )
    insight_anchor_center = _vector_from(
        (nightly_summary or {}).get("inner_os_sleep_insight_anchor_center")
        or (insight_summary.get("insight_anchor_center") if isinstance(insight_summary, Mapping) else None)
    )
    insight_anchor_dispersion = _float_from(
        nightly_summary,
        "inner_os_sleep_insight_anchor_dispersion",
        _float_from(insight_summary, "insight_anchor_dispersion", 0.0),
    )
    temperament_focus = _text_from(nightly_summary, "inner_os_sleep_temperament_focus")
    temperament_forward_bias = _float_from(nightly_summary, "inner_os_sleep_temperament_forward_bias", 0.0)
    temperament_guard_bias = _float_from(nightly_summary, "inner_os_sleep_temperament_guard_bias", 0.0)
    temperament_bond_bias = _float_from(nightly_summary, "inner_os_sleep_temperament_bond_bias", 0.0)
    temperament_recovery_bias = _float_from(nightly_summary, "inner_os_sleep_temperament_recovery_bias", 0.0)
    homeostasis_budget_focus = _text_from(nightly_summary, "inner_os_sleep_homeostasis_budget_focus")
    homeostasis_budget_bias = _float_from(nightly_summary, "inner_os_sleep_homeostasis_budget_bias", 0.0)
    body_homeostasis_focus = _text_from(nightly_summary, "inner_os_sleep_body_homeostasis_focus")
    body_homeostasis_carry_bias = _float_from(nightly_summary, "inner_os_sleep_body_homeostasis_carry_bias", 0.0)
    relational_continuity_focus = _text_from(nightly_summary, "inner_os_sleep_relational_continuity_focus")
    relational_continuity_carry_bias = _float_from(nightly_summary, "inner_os_sleep_relational_continuity_carry_bias", 0.0)
    group_thread_registry = (nightly_summary or {}).get("inner_os_group_thread_registry_summary") or {}
    discussion_thread_registry = (nightly_summary or {}).get("inner_os_discussion_thread_registry_summary") or {}
    autobiographical_thread_mode = _text_from(nightly_summary, "inner_os_sleep_autobiographical_thread_mode")
    autobiographical_thread_anchor = _text_from(nightly_summary, "inner_os_sleep_autobiographical_thread_anchor")
    autobiographical_thread_focus = _text_from(nightly_summary, "inner_os_sleep_autobiographical_thread_focus")
    autobiographical_thread_strength = _float_from(nightly_summary, "inner_os_sleep_autobiographical_thread_strength", 0.0)
    group_thread_focus = _text_from(nightly_summary, "inner_os_sleep_group_thread_focus")
    group_thread_carry_bias = _float_from(nightly_summary, "inner_os_sleep_group_thread_carry_bias", 0.0)
    temporal_membrane_focus = _text_from(nightly_summary, "inner_os_sleep_temporal_membrane_focus")
    temporal_timeline_bias = _float_from(nightly_summary, "inner_os_sleep_temporal_timeline_bias", 0.0)
    temporal_reentry_bias = _float_from(nightly_summary, "inner_os_sleep_temporal_reentry_bias", 0.0)
    temporal_supersession_bias = _float_from(nightly_summary, "inner_os_sleep_temporal_supersession_bias", 0.0)
    temporal_continuity_bias = _float_from(nightly_summary, "inner_os_sleep_temporal_continuity_bias", 0.0)
    temporal_relation_reentry_bias = _float_from(nightly_summary, "inner_os_sleep_temporal_relation_reentry_bias", 0.0)
    expressive_style_focus = _text_from(nightly_summary, "inner_os_sleep_expressive_style_focus")
    expressive_style_carry_bias = _float_from(nightly_summary, "inner_os_sleep_expressive_style_carry_bias", 0.0)
    expressive_style_history_focus = _text_from(nightly_summary, "inner_os_sleep_expressive_style_history_focus")
    expressive_style_history_bias = _float_from(nightly_summary, "inner_os_sleep_expressive_style_history_bias", 0.0)
    banter_style_focus = _text_from(nightly_summary, "inner_os_sleep_banter_style_focus")
    lexical_variation_carry_bias = _float_from(nightly_summary, "inner_os_sleep_lexical_variation_carry_bias", 0.0)

    return {
        "recovery_need": recovery_need,
        "roughness_dwell": roughness_dwell,
        "defensive_dwell": defensive_dwell,
        "ritual_memory": ritual_memory,
        "institutional_memory": institutional_memory,
        "conscious_mosaic_density": conscious_mosaic_density,
        "conscious_mosaic_recentness": conscious_mosaic_recentness,
        "culture_resonance": culture_resonance,
        "community_resonance": culture_resonance,
        "person_registry_snapshot": dict(partner_registry) if isinstance(partner_registry, Mapping) else {},
        "related_person_id": related_person_id,
        "related_person_ids": registry_top_person_ids,
        "attachment": attachment,
        "familiarity": familiarity,
        "trust_memory": trust_memory,
        "relation_seed_strength": relation_seed_strength,
        "partner_address_hint": partner_address_hint,
        "partner_timing_hint": partner_timing_hint,
        "partner_stance_hint": partner_stance_hint,
        "partner_social_interpretation": partner_social_interpretation,
        "memory_write_class": memory_write_class,
        "memory_write_class_counts": memory_write_class_counts,
        "agenda_focus": agenda_focus,
        "agenda_bias": agenda_bias,
        "agenda_reason": agenda_reason,
        "agenda_window_focus": agenda_window_focus,
        "agenda_window_bias": agenda_window_bias,
        "agenda_window_reason": agenda_window_reason,
        "agenda_window_carry_target": agenda_window_carry_target,
        "learning_mode_focus": learning_mode_focus,
        "learning_mode_carry_bias": learning_mode_carry_bias,
        "social_experiment_focus": social_experiment_focus,
        "social_experiment_carry_bias": social_experiment_carry_bias,
        "commitment_target_focus": commitment_target_focus,
        "commitment_state_focus": commitment_state_focus,
        "commitment_target_counts": commitment_target_counts,
        "commitment_carry_bias": commitment_carry_bias,
        "commitment_followup_focus": commitment_followup_focus,
        "commitment_mode_focus": commitment_mode_focus,
        "commitment_carry_reason": commitment_carry_reason,
        "terrain_reweighting_bias": terrain_reweighting_bias,
        "insight_class_focus": insight_class_focus,
        "insight_link_counts": insight_link_counts,
        "insight_class_counts": insight_class_counts,
        "insight_reframing_bias": insight_reframing_bias,
        "association_reweighting_bias": association_reweighting_bias,
        "association_reweighting_focus": association_reweighting_focus,
        "association_reweighting_reason": association_reweighting_reason,
        "insight_terrain_shape_bias": insight_terrain_shape_bias,
        "insight_terrain_shape_reason": insight_terrain_shape_reason,
        "insight_terrain_shape_target": insight_terrain_shape_target,
        "insight_anchor_center": insight_anchor_center,
        "insight_anchor_dispersion": insight_anchor_dispersion,
        "temperament_focus": temperament_focus,
        "temperament_forward_bias": temperament_forward_bias,
        "temperament_guard_bias": temperament_guard_bias,
        "temperament_bond_bias": temperament_bond_bias,
        "temperament_recovery_bias": temperament_recovery_bias,
        "homeostasis_budget_focus": homeostasis_budget_focus,
        "homeostasis_budget_bias": homeostasis_budget_bias,
        "body_homeostasis_focus": body_homeostasis_focus,
        "body_homeostasis_carry_bias": body_homeostasis_carry_bias,
        "relational_continuity_focus": relational_continuity_focus,
        "relational_continuity_carry_bias": relational_continuity_carry_bias,
        "group_thread_registry_snapshot": dict(group_thread_registry) if isinstance(group_thread_registry, Mapping) else {},
        "discussion_thread_registry_snapshot": dict(discussion_thread_registry) if isinstance(discussion_thread_registry, Mapping) else {},
        "autobiographical_thread_mode": autobiographical_thread_mode,
        "autobiographical_thread_anchor": autobiographical_thread_anchor,
        "autobiographical_thread_focus": autobiographical_thread_focus,
        "autobiographical_thread_strength": autobiographical_thread_strength,
        "group_thread_focus": group_thread_focus,
        "group_thread_carry_bias": group_thread_carry_bias,
        "temporal_membrane_focus": temporal_membrane_focus,
        "temporal_timeline_bias": temporal_timeline_bias,
        "temporal_reentry_bias": temporal_reentry_bias,
        "temporal_supersession_bias": temporal_supersession_bias,
        "temporal_continuity_bias": temporal_continuity_bias,
        "temporal_relation_reentry_bias": temporal_relation_reentry_bias,
        "expressive_style_focus": expressive_style_focus,
        "expressive_style_carry_bias": expressive_style_carry_bias,
        "expressive_style_history_focus": expressive_style_history_focus,
        "expressive_style_history_bias": expressive_style_history_bias,
        "banter_style_focus": banter_style_focus,
        "lexical_variation_carry_bias": lexical_variation_carry_bias,
    }


def _derive_forgetting_snapshot(
    *,
    nightly_summary: Optional[Mapping[str, Any]],
    derived_current: Mapping[str, Any],
) -> dict[str, Any]:
    forgetting = (nightly_summary or {}).get("forgetting") or {}
    replay_horizon = int(_float_from(forgetting, "replay_horizon", _float_from(forgetting, "horizon", 2.0)))
    forgetting_pressure = _float_from(
        forgetting,
        "forgetting_pressure",
        _clamp01(
            _float_from(derived_current, "recovery_need", 0.0) * 0.34
            + _float_from(derived_current, "roughness_dwell", 0.0) * 0.24
            + _float_from(derived_current, "defensive_dwell", 0.0) * 0.16
        ),
    )
    persona_halflife_tau = _float_from(forgetting, "persona_halflife_tau", 24.0)
    return {
        "forgetting_pressure": _clamp01(forgetting_pressure),
        "replay_horizon": max(1, replay_horizon),
        "persona_halflife_tau": max(6.0, persona_halflife_tau),
    }


def _derive_memory_orchestration(
    *,
    nightly_summary: Optional[Mapping[str, Any]],
    memory_inventory: Optional[Mapping[str, Any]],
    latest_field_metrics: Optional[Mapping[str, Any]],
    forgetting_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    l1_count = _float_from(memory_inventory, "l1_count", 0.0)
    l2_count = _float_from(memory_inventory, "l2_count", 0.0)
    l3_count = _float_from(memory_inventory, "l3_count", 0.0)
    recent_experiences = _float_from(memory_inventory, "recent_experiences", l1_count)

    roughness = _clamp01(_float_from(latest_field_metrics, "entropy", 0.0) / 10.0)
    forgetting_pressure = _float_from(forgetting_snapshot, "forgetting_pressure", 0.0)
    memory_match = (nightly_summary or {}).get("memory_match") or {}
    resonance = (nightly_summary or {}).get("resonance") or {}
    resonance_summary = resonance.get("summary") if isinstance(resonance, Mapping) else {}
    monument_salience = _clamp01(
        _float_from(memory_match, "match_ratio", 0.0) * 0.6
        + _float_from(resonance_summary, "energy", 0.0) * 0.18
    )
    reuse_trajectory = _clamp01(
        _clamp01(l2_count / max(1.0, l1_count)) * 0.36
        + _clamp01(recent_experiences / 24.0) * 0.22
        + monument_salience * 0.12
    )
    interference_pressure = _clamp01(
        forgetting_pressure * 0.42
        + _clamp01(l1_count / 64.0) * 0.18
        + roughness * 0.14
    )
    consolidation_priority = _clamp01(
        _clamp01(recent_experiences / 24.0) * 0.24
        + _clamp01(l2_count / 32.0) * 0.2
        + max(0.0, 1.0 - interference_pressure) * 0.16
        + monument_salience * 0.18
    )
    prospective_memory_pull = _clamp01(
        _float_from(resonance_summary, "objective", 0.0) * 0.2
        + _float_from(resonance_summary, "corr", 0.0) * 0.12
        + _clamp01(l3_count / 24.0) * 0.12
    )
    return {
        "monument_salience": monument_salience,
        "conscious_mosaic_density": _clamp01(
            _clamp01(recent_experiences / 24.0) * 0.5
            + _clamp01(l2_count / 32.0) * 0.22
            + _clamp01(l3_count / 24.0) * 0.12
        ),
        "conscious_mosaic_recentness": _clamp01(
            _clamp01(recent_experiences / max(1.0, l1_count + 1.0)) * 0.7
            + monument_salience * 0.08
        ),
        "reuse_trajectory": reuse_trajectory,
        "interference_pressure": interference_pressure,
        "consolidation_priority": consolidation_priority,
        "prospective_memory_pull": prospective_memory_pull,
    }


def _memory_inventory_from_system(system: Any) -> dict[str, Any]:
    l1 = getattr(getattr(system, "l1", None), "experiences", None) or []
    l2 = getattr(getattr(system, "l2", None), "episodes", None) or []
    l3 = getattr(getattr(system, "l3", None), "patterns", None) or []
    return {
        "l1_count": float(len(l1)),
        "l2_count": float(len(l2)),
        "l3_count": float(len(l3)),
        "recent_experiences": float(min(len(l1), 24)),
    }


def _latest_field_metrics(field_metrics_state: Any) -> dict[str, Any]:
    if isinstance(field_metrics_state, list) and field_metrics_state:
        latest = field_metrics_state[-1]
        if isinstance(latest, Mapping):
            return dict(latest)
    if isinstance(field_metrics_state, Mapping):
        return dict(field_metrics_state)
    return {}


def _call_mapping(obj: Any, attr: str) -> dict[str, Any]:
    value = _call_value(obj, attr)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _call_value(obj: Any, attr: str) -> Any:
    target = getattr(obj, attr, None)
    if callable(target):
        try:
            return target()
        except Exception:
            return None
    return target


def _culture_resonance_from_stats(stats: Mapping[str, Any]) -> float:
    if not isinstance(stats, Mapping) or not stats:
        return 0.0
    totals = []
    for value in stats.values():
        if not isinstance(value, Mapping):
            continue
        count = _float_from(value, "count", 0.0)
        politeness = _float_from(value, "mean_politeness", 0.0)
        intimacy = _float_from(value, "mean_intimacy", 0.0)
        totals.append(_clamp01(count / 12.0) * 0.5 + politeness * 0.25 + intimacy * 0.25)
    if not totals:
        return 0.0
    return _clamp01(sum(totals) / len(totals))


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _text_from(mapping: Optional[Mapping[str, Any]], key: str) -> str:
    if not isinstance(mapping, Mapping):
        return ""
    value = mapping.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _vector_from(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result
