from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)


@dataclass(frozen=True)
class DailyCarrySummary:
    summary_version: str = "v1"
    same_turn_focus: dict[str, Any] = field(default_factory=dict)
    overnight_focus: dict[str, Any] = field(default_factory=dict)
    carry_strengths: dict[str, float] = field(default_factory=dict)
    active_carry_channels: tuple[str, ...] = ()
    dominant_carry_channel: str = ""
    carry_alignment: dict[str, bool] = field(default_factory=dict)
    temporal_alignment: dict[str, Any] = field(default_factory=dict)
    boundary_alignment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "same_turn_focus": dict(self.same_turn_focus),
            "overnight_focus": dict(self.overnight_focus),
            "carry_strengths": {str(key): round(float(value), 4) for key, value in self.carry_strengths.items()},
            "active_carry_channels": list(self.active_carry_channels),
            "dominant_carry_channel": self.dominant_carry_channel,
            "carry_alignment": dict(self.carry_alignment),
            "temporal_alignment": dict(self.temporal_alignment),
            "boundary_alignment": dict(self.boundary_alignment),
        }


class DailyCarrySummaryBuilder:
    def build(self, report: Mapping[str, Any] | None) -> DailyCarrySummary:
        payload = dict(report or {})
        memory_summary = dict(payload.get("inner_os_memory_class_summary") or {})
        agenda_summary = dict(payload.get("inner_os_agenda_summary") or {})
        commitment_summary = dict(payload.get("inner_os_commitment_summary") or {})
        insight_summary = dict(payload.get("inner_os_insight_summary") or {})
        partner_summary = dict(payload.get("inner_os_partner_relation_summary") or {})
        partner_registry_summary = dict(payload.get("inner_os_partner_relation_registry_summary") or {})
        group_thread_registry_summary = dict(payload.get("inner_os_group_thread_registry_summary") or {})
        discussion_thread_registry_summary = dict(payload.get("inner_os_discussion_thread_registry_summary") or {})
        relation_arc_summary = dict(payload.get("inner_os_relation_arc_summary") or {})
        group_relation_arc_summary = dict(payload.get("inner_os_group_relation_arc_summary") or {})
        relation_arc_registry_summary = dict(payload.get("inner_os_relation_arc_registry_summary") or {})
        identity_arc_summary = dict(payload.get("inner_os_identity_arc_summary") or {})
        identity_arc_registry_summary = dict(payload.get("inner_os_identity_arc_registry_summary") or {})

        same_turn_focus = {
            "memory_class": _text(memory_summary.get("dominant_class")),
            "memory_reason": _text(memory_summary.get("dominant_reason")),
            "agenda_state": _text(payload.get("inner_os_same_turn_agenda_state")) or _text(agenda_summary.get("dominant_agenda")),
            "agenda_reason": _text(payload.get("inner_os_same_turn_agenda_reason")) or _text(agenda_summary.get("dominant_reason")),
            "agenda_window_state": _text(payload.get("inner_os_same_turn_agenda_window_state")),
            "agenda_window_carry_target": _text(payload.get("inner_os_same_turn_agenda_window_carry_target")),
            "learning_mode_state": _text(payload.get("inner_os_same_turn_learning_mode_state")),
            "social_experiment_state": _text(payload.get("inner_os_same_turn_social_experiment_state")),
            "commitment_target": _text(commitment_summary.get("dominant_target")),
            "commitment_state": _text(commitment_summary.get("dominant_state")),
            "insight_class": _text(insight_summary.get("dominant_insight_class")),
            "insight_topic": _text(insight_summary.get("dominant_reframed_topic")),
            "partner_person_id": _text(partner_summary.get("person_id")),
            "partner_role": _text(partner_summary.get("social_role")),
            "partner_registry_dominant_person": _text(partner_registry_summary.get("dominant_person_id")),
            "partner_registry_total_people": int(partner_registry_summary.get("total_people") or 0),
            "group_thread_dominant_thread": _text(group_thread_registry_summary.get("dominant_thread_id")),
            "group_thread_total_threads": int(group_thread_registry_summary.get("total_threads") or 0),
            "discussion_registry_dominant_thread": _text(discussion_thread_registry_summary.get("dominant_thread_id")),
            "discussion_registry_dominant_anchor": _text(discussion_thread_registry_summary.get("dominant_anchor")),
            "discussion_registry_dominant_issue_state": _text(discussion_thread_registry_summary.get("dominant_issue_state")),
            "discussion_registry_total_threads": int(discussion_thread_registry_summary.get("total_threads") or 0),
            "contact_reflection_state": _text(payload.get("inner_os_same_turn_contact_reflection_state")),
            "contact_reflection_style": _text(payload.get("inner_os_same_turn_contact_reflection_style")),
            "contact_transmit_share": round(_float01(payload.get("inner_os_same_turn_contact_transmit_share")), 4),
            "contact_reflect_share": round(_float01(payload.get("inner_os_same_turn_contact_reflect_share")), 4),
            "contact_absorb_share": round(_float01(payload.get("inner_os_same_turn_contact_absorb_share")), 4),
            "contact_block_share": round(_float01(payload.get("inner_os_same_turn_contact_block_share")), 4),
            "autobiographical_thread_mode": _text(payload.get("inner_os_same_turn_autobiographical_thread_mode")),
            "autobiographical_thread_anchor": _text(payload.get("inner_os_same_turn_autobiographical_thread_anchor")),
            "autobiographical_thread_focus": _text(payload.get("inner_os_same_turn_autobiographical_thread_focus")),
            "autobiographical_thread_strength": round(_float01(payload.get("inner_os_same_turn_autobiographical_thread_strength")), 4),
            "relation_arc_kind": _text(relation_arc_summary.get("arc_kind")),
            "relation_arc_phase": _text(relation_arc_summary.get("phase")),
            "group_relation_arc_kind": _text(group_relation_arc_summary.get("arc_kind")),
            "group_relation_boundary_mode": _text(group_relation_arc_summary.get("boundary_mode")),
            "temporal_membrane_mode": _text(payload.get("inner_os_same_turn_temporal_membrane_mode")),
            "temporal_timeline_coherence": round(_float01(payload.get("inner_os_same_turn_temporal_timeline_coherence")), 4),
            "temporal_reentry_pull": round(_float01(payload.get("inner_os_same_turn_temporal_reentry_pull")), 4),
            "temporal_supersession_pressure": round(_float01(payload.get("inner_os_same_turn_temporal_supersession_pressure")), 4),
            "temporal_continuity_pressure": round(_float01(payload.get("inner_os_same_turn_temporal_continuity_pressure")), 4),
            "temporal_relation_reentry_pull": round(_float01(payload.get("inner_os_same_turn_temporal_relation_reentry_pull")), 4),
            "boundary_gate_mode": _text(payload.get("inner_os_same_turn_boundary_gate_mode")),
            "boundary_transform_mode": _text(payload.get("inner_os_same_turn_boundary_transform_mode")),
            "boundary_softened_count": int(payload.get("inner_os_same_turn_boundary_softened_count") or 0),
            "boundary_withheld_count": int(payload.get("inner_os_same_turn_boundary_withheld_count") or 0),
            "boundary_deferred_count": int(payload.get("inner_os_same_turn_boundary_deferred_count") or 0),
            "boundary_residual_pressure": round(_float01(payload.get("inner_os_same_turn_boundary_residual_pressure")), 4),
            "residual_reflection_mode": _text(payload.get("inner_os_same_turn_residual_reflection_mode")),
            "residual_reflection_focus": _text(payload.get("inner_os_same_turn_residual_reflection_focus")),
            "residual_reflection_strength": round(_float01(payload.get("inner_os_same_turn_residual_reflection_strength")), 4),
            "relation_arc_registry_dominant_kind": _text(relation_arc_registry_summary.get("dominant_arc_kind")),
            "relation_arc_registry_active_count": int(relation_arc_registry_summary.get("active_arc_count") or 0),
            "identity_arc_kind": _text(identity_arc_summary.get("arc_kind")),
            "identity_arc_phase": _text(identity_arc_summary.get("phase")),
            "identity_arc_registry_dominant_kind": _text(identity_arc_registry_summary.get("dominant_arc_kind")),
            "identity_arc_registry_active_count": int(identity_arc_registry_summary.get("active_arc_count") or 0),
            "expressive_style": _text(payload.get("inner_os_same_turn_expressive_style_state")),
            "relational_banter_style": _text(payload.get("inner_os_same_turn_relational_style_banter_style")),
            "growth_relational_trust": round(_float01(payload.get("inner_os_same_turn_growth_relational_trust")), 4),
            "growth_epistemic_maturity": round(_float01(payload.get("inner_os_same_turn_growth_epistemic_maturity")), 4),
            "growth_expressive_range": round(_float01(payload.get("inner_os_same_turn_growth_expressive_range")), 4),
            "growth_residue_integration": round(_float01(payload.get("inner_os_same_turn_growth_residue_integration")), 4),
            "growth_playfulness_range": round(_float01(payload.get("inner_os_same_turn_growth_playfulness_range")), 4),
            "growth_self_coherence": round(_float01(payload.get("inner_os_same_turn_growth_self_coherence")), 4),
            "growth_dominant_transition": _text(payload.get("inner_os_same_turn_growth_dominant_transition")),
            "growth_bond_axis": round(_float01(payload.get("inner_os_same_turn_growth_bond_axis")), 4),
            "growth_stability_axis": round(_float01(payload.get("inner_os_same_turn_growth_stability_axis")), 4),
            "growth_curiosity_axis": round(_float01(payload.get("inner_os_same_turn_growth_curiosity_axis")), 4),
            "memory_dynamics_mode": _text(payload.get("inner_os_same_turn_memory_dynamics_mode")),
            "memory_dominant_link": _text(payload.get("inner_os_same_turn_memory_dominant_link")),
            "memory_monument_kind": _text(payload.get("inner_os_same_turn_memory_monument_kind")),
            "memory_monument_salience": round(_float01(payload.get("inner_os_same_turn_memory_monument_salience")), 4),
            "memory_ignition_readiness": round(_float01(payload.get("inner_os_same_turn_memory_ignition_readiness")), 4),
            "memory_consolidation_pull": round(_float01(payload.get("inner_os_same_turn_memory_consolidation_pull")), 4),
            "memory_tension": round(_float01(payload.get("inner_os_same_turn_memory_tension")), 4),
            "memory_topology_axis": round(_float01(payload.get("inner_os_same_turn_memory_topology_axis")), 4),
            "memory_salience_axis": round(_float01(payload.get("inner_os_same_turn_memory_salience_axis")), 4),
            "memory_ignition_axis": round(_float01(payload.get("inner_os_same_turn_memory_ignition_axis")), 4),
            "memory_consolidation_axis": round(_float01(payload.get("inner_os_same_turn_memory_consolidation_axis")), 4),
            "memory_tension_axis": round(_float01(payload.get("inner_os_same_turn_memory_tension_axis")), 4),
        }

        overnight_focus = {
            "memory_class_focus": _text(payload.get("inner_os_sleep_memory_class_focus")),
            "agenda_focus": _text(payload.get("inner_os_sleep_agenda_focus")),
            "agenda_reason": _text(payload.get("inner_os_sleep_agenda_reason")),
            "agenda_window_focus": _text(payload.get("inner_os_sleep_agenda_window_focus")),
            "agenda_window_reason": _text(payload.get("inner_os_sleep_agenda_window_reason")),
            "agenda_window_carry_target": _text(payload.get("inner_os_sleep_agenda_window_carry_target")),
            "learning_mode_focus": _text(payload.get("inner_os_sleep_learning_mode_focus")),
            "social_experiment_focus": _text(payload.get("inner_os_sleep_social_experiment_focus")),
            "commitment_target_focus": _text(payload.get("inner_os_sleep_commitment_target_focus")),
            "commitment_state_focus": _text(payload.get("inner_os_sleep_commitment_state_focus")),
            "commitment_followup_focus": _text(payload.get("inner_os_sleep_commitment_followup_focus")),
            "commitment_mode_focus": _text(payload.get("inner_os_sleep_commitment_mode_focus")),
            "association_focus": _text(payload.get("inner_os_sleep_association_reweighting_focus")),
            "association_reason": _text(payload.get("inner_os_sleep_association_reweighting_reason")),
            "insight_class_focus": _text(payload.get("inner_os_sleep_insight_class_focus")),
            "terrain_shape_target": _text(payload.get("inner_os_sleep_insight_terrain_shape_target")),
            "terrain_shape_reason": _text(payload.get("inner_os_sleep_insight_terrain_shape_reason")),
            "temperament_focus": _text(payload.get("inner_os_sleep_temperament_focus")),
            "homeostasis_budget_focus": _text(payload.get("inner_os_sleep_homeostasis_budget_focus")),
            "body_homeostasis_focus": _text(payload.get("inner_os_sleep_body_homeostasis_focus")),
            "relational_continuity_focus": _text(payload.get("inner_os_sleep_relational_continuity_focus")),
            "group_thread_focus": _text(payload.get("inner_os_sleep_group_thread_focus")),
            "discussion_registry_dominant_thread": _text(discussion_thread_registry_summary.get("dominant_thread_id")),
            "discussion_registry_dominant_anchor": _text(discussion_thread_registry_summary.get("dominant_anchor")),
            "discussion_registry_dominant_issue_state": _text(discussion_thread_registry_summary.get("dominant_issue_state")),
            "discussion_registry_total_threads": int(discussion_thread_registry_summary.get("total_threads") or 0),
            "autobiographical_thread_mode": _text(payload.get("inner_os_sleep_autobiographical_thread_mode")),
            "autobiographical_thread_anchor": _text(payload.get("inner_os_sleep_autobiographical_thread_anchor")),
            "autobiographical_thread_focus": _text(payload.get("inner_os_sleep_autobiographical_thread_focus")),
            "autobiographical_thread_strength": round(_float01(payload.get("inner_os_sleep_autobiographical_thread_strength")), 4),
            "temporal_membrane_focus": _text(payload.get("inner_os_sleep_temporal_membrane_focus")),
            "temporal_timeline_bias": round(_float01(payload.get("inner_os_sleep_temporal_timeline_bias")), 4),
            "temporal_reentry_bias": round(_float01(payload.get("inner_os_sleep_temporal_reentry_bias")), 4),
            "temporal_supersession_bias": round(_float01(payload.get("inner_os_sleep_temporal_supersession_bias")), 4),
            "temporal_continuity_bias": round(_float01(payload.get("inner_os_sleep_temporal_continuity_bias")), 4),
            "temporal_relation_reentry_bias": round(_float01(payload.get("inner_os_sleep_temporal_relation_reentry_bias")), 4),
            "relation_arc_kind": _text(relation_arc_summary.get("arc_kind")),
            "relation_arc_phase": _text(relation_arc_summary.get("phase")),
            "group_relation_arc_kind": _text(group_relation_arc_summary.get("arc_kind")),
            "group_relation_boundary_mode": _text(group_relation_arc_summary.get("boundary_mode")),
            "group_relation_reentry_window_focus": _text(group_relation_arc_summary.get("reentry_window_focus")),
            "relation_arc_registry_dominant_kind": _text(relation_arc_registry_summary.get("dominant_arc_kind")),
            "relation_arc_registry_active_count": int(relation_arc_registry_summary.get("active_arc_count") or 0),
            "identity_arc_kind": _text(identity_arc_summary.get("arc_kind")),
            "identity_arc_phase": _text(identity_arc_summary.get("phase")),
            "identity_arc_summary": _text(identity_arc_summary.get("summary")),
            "identity_arc_registry_dominant_kind": _text(identity_arc_registry_summary.get("dominant_arc_kind")),
            "identity_arc_registry_active_count": int(identity_arc_registry_summary.get("active_arc_count") or 0),
            "expressive_style_focus": _text(payload.get("inner_os_sleep_expressive_style_focus")),
            "expressive_style_history_focus": _text(payload.get("inner_os_sleep_expressive_style_history_focus")),
            "banter_style_focus": _text(payload.get("inner_os_sleep_banter_style_focus")),
            "growth_relational_trust": round(_float01(payload.get("inner_os_sleep_growth_relational_trust")), 4),
            "growth_epistemic_maturity": round(_float01(payload.get("inner_os_sleep_growth_epistemic_maturity")), 4),
            "growth_expressive_range": round(_float01(payload.get("inner_os_sleep_growth_expressive_range")), 4),
            "growth_residue_integration": round(_float01(payload.get("inner_os_sleep_growth_residue_integration")), 4),
            "growth_playfulness_range": round(_float01(payload.get("inner_os_sleep_growth_playfulness_range")), 4),
            "growth_self_coherence": round(_float01(payload.get("inner_os_sleep_growth_self_coherence")), 4),
            "growth_dominant_transition": _text(payload.get("inner_os_sleep_growth_dominant_transition")),
            "growth_bond_axis": round(_float01(payload.get("inner_os_sleep_growth_bond_axis")), 4),
            "growth_stability_axis": round(_float01(payload.get("inner_os_sleep_growth_stability_axis")), 4),
            "growth_curiosity_axis": round(_float01(payload.get("inner_os_sleep_growth_curiosity_axis")), 4),
            "memory_dynamics_mode": _text(payload.get("inner_os_sleep_memory_dynamics_mode")),
            "memory_dominant_link": _text(payload.get("inner_os_sleep_memory_dominant_link")),
            "memory_monument_kind": _text(payload.get("inner_os_sleep_memory_monument_kind")),
            "memory_monument_salience": round(_float01(payload.get("inner_os_sleep_memory_monument_salience")), 4),
            "memory_ignition_readiness": round(_float01(payload.get("inner_os_sleep_memory_ignition_readiness")), 4),
            "memory_consolidation_pull": round(_float01(payload.get("inner_os_sleep_memory_consolidation_pull")), 4),
            "memory_tension": round(_float01(payload.get("inner_os_sleep_memory_tension")), 4),
            "memory_topology_axis": round(_float01(payload.get("inner_os_sleep_memory_topology_axis")), 4),
            "memory_salience_axis": round(_float01(payload.get("inner_os_sleep_memory_salience_axis")), 4),
            "memory_ignition_axis": round(_float01(payload.get("inner_os_sleep_memory_ignition_axis")), 4),
            "memory_consolidation_axis": round(_float01(payload.get("inner_os_sleep_memory_consolidation_axis")), 4),
            "memory_tension_axis": round(_float01(payload.get("inner_os_sleep_memory_tension_axis")), 4),
        }

        carry_strengths = {
            "terrain_reweighting": _float01(payload.get("inner_os_sleep_terrain_reweighting_bias")),
            "agenda": _float01(payload.get("inner_os_sleep_agenda_bias")),
            "agenda_window": _float01(payload.get("inner_os_sleep_agenda_window_bias")),
            "learning_mode": _float01(payload.get("inner_os_sleep_learning_mode_carry_bias")),
            "social_experiment": _float01(payload.get("inner_os_sleep_social_experiment_carry_bias")),
            "commitment_carry": _float01(payload.get("inner_os_sleep_commitment_carry_bias")),
            "association_reweighting": _float01(payload.get("inner_os_sleep_association_reweighting_bias")),
            "insight_reframing": _float01(payload.get("inner_os_sleep_insight_reframing_bias")),
            "insight_terrain_shape": _float01(payload.get("inner_os_sleep_insight_terrain_shape_bias")),
            "temperament_forward": _float01(payload.get("inner_os_sleep_temperament_forward_bias")),
            "temperament_guard": _float01(payload.get("inner_os_sleep_temperament_guard_bias")),
            "temperament_bond": _float01(payload.get("inner_os_sleep_temperament_bond_bias")),
            "temperament_recovery": _float01(payload.get("inner_os_sleep_temperament_recovery_bias")),
            "homeostasis_budget": _float01(payload.get("inner_os_sleep_homeostasis_budget_bias")),
            "body_homeostasis_carry": _float01(payload.get("inner_os_sleep_body_homeostasis_carry_bias")),
            "relational_continuity_carry": _float01(payload.get("inner_os_sleep_relational_continuity_carry_bias")),
            "group_thread_carry": _float01(payload.get("inner_os_sleep_group_thread_carry_bias")),
            "autobiographical_thread": _float01(payload.get("inner_os_sleep_autobiographical_thread_strength")),
            "temporal_timeline": _float01(payload.get("inner_os_sleep_temporal_timeline_bias")),
            "temporal_reentry": _float01(payload.get("inner_os_sleep_temporal_reentry_bias")),
            "temporal_supersession": _float01(payload.get("inner_os_sleep_temporal_supersession_bias")),
            "temporal_continuity": _float01(payload.get("inner_os_sleep_temporal_continuity_bias")),
            "temporal_relation_reentry": _float01(payload.get("inner_os_sleep_temporal_relation_reentry_bias")),
            "expressive_style_carry": _float01(payload.get("inner_os_sleep_expressive_style_carry_bias")),
            "expressive_style_history_carry": _float01(payload.get("inner_os_sleep_expressive_style_history_bias")),
            "lexical_variation_carry": _float01(payload.get("inner_os_sleep_lexical_variation_carry_bias")),
        }

        active_carry_channels = tuple(
            key for key, value in carry_strengths.items()
            if value >= 0.08
        )
        dominant_carry_channel = ""
        if carry_strengths:
            dominant_carry_channel = max(
                carry_strengths.keys(),
                key=lambda key: (carry_strengths[key], key),
            )
            if carry_strengths.get(dominant_carry_channel, 0.0) <= 0.0:
                dominant_carry_channel = ""

        same_target = same_turn_focus["commitment_target"]
        overnight_target = overnight_focus["commitment_target_focus"]
        same_agenda = same_turn_focus["agenda_state"]
        overnight_agenda = overnight_focus["agenda_focus"]
        same_agenda_window = same_turn_focus["agenda_window_state"]
        overnight_agenda_window = overnight_focus["agenda_window_focus"]
        same_learning_mode = same_turn_focus["learning_mode_state"]
        overnight_learning_mode = overnight_focus["learning_mode_focus"]
        same_social_experiment = same_turn_focus["social_experiment_state"]
        overnight_social_experiment = overnight_focus["social_experiment_focus"]
        same_insight = same_turn_focus["insight_class"]
        overnight_insight = overnight_focus["insight_class_focus"]
        same_memory = same_turn_focus["memory_class"]
        memory_focus = overnight_focus["memory_class_focus"]
        alignment = {
            "memory_carry_visible": bool(same_memory and memory_focus and same_memory == memory_focus),
            "agenda_carry_visible": bool(same_agenda and overnight_agenda and same_agenda == overnight_agenda),
            "agenda_window_carry_visible": bool(
                same_agenda_window and overnight_agenda_window and same_agenda_window == overnight_agenda_window
            ),
            "learning_mode_carry_visible": bool(
                same_learning_mode and overnight_learning_mode and same_learning_mode == overnight_learning_mode
            ),
            "social_experiment_carry_visible": bool(
                same_social_experiment and overnight_social_experiment and same_social_experiment == overnight_social_experiment
            ),
            "commitment_carry_visible": bool(same_target and overnight_target and same_target == overnight_target),
            "insight_carry_visible": bool(same_insight and overnight_insight and same_insight == overnight_insight),
            "association_carry_visible": bool(
                overnight_focus["association_focus"]
                and carry_strengths["association_reweighting"] > 0.0
            ),
            "terrain_shape_carry_visible": bool(
                overnight_focus["terrain_shape_target"]
                and carry_strengths["insight_terrain_shape"] > 0.0
            ),
            "temperament_carry_visible": bool(
                overnight_focus["temperament_focus"]
                and (
                    carry_strengths["temperament_forward"] > 0.0
                    or carry_strengths["temperament_guard"] > 0.0
                    or carry_strengths["temperament_bond"] > 0.0
                    or carry_strengths["temperament_recovery"] > 0.0
                )
            ),
            "homeostasis_budget_visible": bool(
                overnight_focus["homeostasis_budget_focus"]
                and carry_strengths["homeostasis_budget"] > 0.0
            ),
            "body_homeostasis_carry_visible": bool(
                overnight_focus["body_homeostasis_focus"]
                and carry_strengths["body_homeostasis_carry"] > 0.0
            ),
            "relational_continuity_carry_visible": bool(
                overnight_focus["relational_continuity_focus"]
                and carry_strengths["relational_continuity_carry"] > 0.0
            ),
            "group_thread_carry_visible": bool(
                overnight_focus["group_thread_focus"]
                and carry_strengths["group_thread_carry"] > 0.0
            ),
            "temporal_membrane_visible": bool(
                overnight_focus["temporal_membrane_focus"]
                or carry_strengths["temporal_timeline"] > 0.0
                or carry_strengths["temporal_reentry"] > 0.0
                or carry_strengths["temporal_supersession"] > 0.0
                or carry_strengths["temporal_continuity"] > 0.0
                or carry_strengths["temporal_relation_reentry"] > 0.0
            ),
            "boundary_visible": bool(
                same_turn_focus["boundary_gate_mode"]
                or same_turn_focus["boundary_transform_mode"]
                or int(same_turn_focus["boundary_softened_count"] or 0) > 0
                or int(same_turn_focus["boundary_withheld_count"] or 0) > 0
                or int(same_turn_focus["boundary_deferred_count"] or 0) > 0
            ),
            "contact_reflection_visible": bool(
                same_turn_focus["contact_reflection_state"]
                or same_turn_focus["contact_reflection_style"]
                or float(same_turn_focus["contact_reflect_share"] or 0.0) > 0.0
                or float(same_turn_focus["contact_absorb_share"] or 0.0) > 0.0
                or float(same_turn_focus["contact_block_share"] or 0.0) > 0.0
            ),
            "residual_visible": bool(
                same_turn_focus["residual_reflection_mode"]
                or same_turn_focus["residual_reflection_focus"]
                or float(same_turn_focus["residual_reflection_strength"] or 0.0) > 0.0
            ),
            "identity_arc_visible": bool(
                overnight_focus["identity_arc_kind"]
                or overnight_focus["identity_arc_phase"]
                or overnight_focus["identity_arc_summary"]
            ),
            "identity_arc_registry_visible": bool(
                overnight_focus["identity_arc_registry_dominant_kind"]
                or int(overnight_focus["identity_arc_registry_active_count"] or 0) > 0
            ),
            "relation_arc_visible": bool(
                overnight_focus["relation_arc_kind"]
                or overnight_focus["relation_arc_phase"]
            ),
            "group_relation_arc_visible": bool(
                overnight_focus["group_relation_arc_kind"]
                or overnight_focus["group_relation_boundary_mode"]
                or overnight_focus["group_relation_reentry_window_focus"]
            ),
            "relation_arc_registry_visible": bool(
                overnight_focus["relation_arc_registry_dominant_kind"]
                or int(overnight_focus["relation_arc_registry_active_count"] or 0) > 0
            ),
            "expressive_style_carry_visible": bool(
                overnight_focus["expressive_style_focus"]
                and carry_strengths["expressive_style_carry"] > 0.0
            ),
            "expressive_style_history_visible": bool(
                overnight_focus["expressive_style_history_focus"]
                and carry_strengths["expressive_style_history_carry"] > 0.0
            ),
            "banter_style_carry_visible": bool(
                overnight_focus["banter_style_focus"]
                and carry_strengths["lexical_variation_carry"] > 0.0
            ),
            "memory_dynamics_carry_visible": bool(
                overnight_focus["memory_dynamics_mode"]
                or overnight_focus["memory_dominant_link"]
                or overnight_focus["memory_monument_kind"]
                or float(overnight_focus["memory_ignition_readiness"] or 0.0) > 0.0
            ),
            "partner_registry_visible": bool(
                same_turn_focus["partner_registry_dominant_person"]
                or int(same_turn_focus["partner_registry_total_people"] or 0) > 1
            ),
            "group_thread_registry_visible": bool(
                same_turn_focus["group_thread_dominant_thread"]
                or int(same_turn_focus["group_thread_total_threads"] or 0) > 0
            ),
            "discussion_registry_visible": bool(
                same_turn_focus["discussion_registry_dominant_thread"]
                or same_turn_focus["discussion_registry_dominant_anchor"]
                or int(same_turn_focus["discussion_registry_total_threads"] or 0) > 0
            ),
            "autobiographical_thread_visible": bool(
                same_turn_focus["autobiographical_thread_mode"]
                or same_turn_focus["autobiographical_thread_anchor"]
                or overnight_focus["autobiographical_thread_mode"]
                or overnight_focus["autobiographical_thread_anchor"]
                or carry_strengths["autobiographical_thread"] > 0.0
            ),
        }

        same_turn_temporal_mode = same_turn_focus["temporal_membrane_mode"]
        overnight_temporal_focus = overnight_focus["temporal_membrane_focus"]
        same_turn_temporal_reentry_pull = float(same_turn_focus["temporal_reentry_pull"] or 0.0)
        overnight_temporal_reentry_bias = float(overnight_focus["temporal_reentry_bias"] or 0.0)
        temporal_alignment = {
            "same_turn_mode": same_turn_temporal_mode,
            "overnight_focus": overnight_temporal_focus,
            "focus_alignment": bool(
                same_turn_temporal_mode
                and overnight_temporal_focus
                and same_turn_temporal_mode == overnight_temporal_focus
            ),
            "same_to_overnight_reentry_delta": round(
                overnight_temporal_reentry_bias - same_turn_temporal_reentry_pull,
                4,
            ),
            "reentry_carry_visible": bool(carry_strengths["temporal_reentry"] > 0.0),
            "reentry_carry_strength": round(float(carry_strengths["temporal_reentry"]), 4),
        }
        boundary_alignment = {
            "gate_mode": same_turn_focus["boundary_gate_mode"],
            "transform_mode": same_turn_focus["boundary_transform_mode"],
            "softened_count": int(same_turn_focus["boundary_softened_count"] or 0),
            "withheld_count": int(same_turn_focus["boundary_withheld_count"] or 0),
            "deferred_count": int(same_turn_focus["boundary_deferred_count"] or 0),
            "residual_pressure": round(float(same_turn_focus["boundary_residual_pressure"] or 0.0), 4),
            "residual_mode": same_turn_focus["residual_reflection_mode"],
            "residual_focus": same_turn_focus["residual_reflection_focus"],
            "residual_strength": round(float(same_turn_focus["residual_reflection_strength"] or 0.0), 4),
            "unsaid_pressure_visible": bool(
                same_turn_focus["boundary_gate_mode"]
                or int(same_turn_focus["boundary_softened_count"] or 0) > 0
                or int(same_turn_focus["boundary_withheld_count"] or 0) > 0
                or int(same_turn_focus["boundary_deferred_count"] or 0) > 0
                or float(same_turn_focus["residual_reflection_strength"] or 0.0) > 0.0
            ),
            "contact_state": same_turn_focus["contact_reflection_state"],
            "contact_style": same_turn_focus["contact_reflection_style"],
            "contact_reflect_share": round(float(same_turn_focus["contact_reflect_share"] or 0.0), 4),
            "contact_absorb_share": round(float(same_turn_focus["contact_absorb_share"] or 0.0), 4),
            "contact_block_share": round(float(same_turn_focus["contact_block_share"] or 0.0), 4),
            "contact_reflection_visible": bool(
                same_turn_focus["contact_reflection_state"]
                or same_turn_focus["contact_reflection_style"]
                or float(same_turn_focus["contact_reflect_share"] or 0.0) > 0.0
                or float(same_turn_focus["contact_absorb_share"] or 0.0) > 0.0
                or float(same_turn_focus["contact_block_share"] or 0.0) > 0.0
            ),
        }

        return DailyCarrySummary(
            same_turn_focus=same_turn_focus,
            overnight_focus=overnight_focus,
            carry_strengths=carry_strengths,
            active_carry_channels=active_carry_channels,
            dominant_carry_channel=dominant_carry_channel,
            carry_alignment=alignment,
            temporal_alignment=temporal_alignment,
            boundary_alignment=boundary_alignment,
        )
