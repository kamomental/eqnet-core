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

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "same_turn_focus": dict(self.same_turn_focus),
            "overnight_focus": dict(self.overnight_focus),
            "carry_strengths": {str(key): round(float(value), 4) for key, value in self.carry_strengths.items()},
            "active_carry_channels": list(self.active_carry_channels),
            "dominant_carry_channel": self.dominant_carry_channel,
            "carry_alignment": dict(self.carry_alignment),
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

        same_turn_focus = {
            "memory_class": _text(memory_summary.get("dominant_class")),
            "memory_reason": _text(memory_summary.get("dominant_reason")),
            "agenda_state": _text(payload.get("inner_os_same_turn_agenda_state")) or _text(agenda_summary.get("dominant_agenda")),
            "agenda_reason": _text(payload.get("inner_os_same_turn_agenda_reason")) or _text(agenda_summary.get("dominant_reason")),
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
            "expressive_style": _text(payload.get("inner_os_same_turn_expressive_style_state")),
            "relational_banter_style": _text(payload.get("inner_os_same_turn_relational_style_banter_style")),
        }

        overnight_focus = {
            "memory_class_focus": _text(payload.get("inner_os_sleep_memory_class_focus")),
            "agenda_focus": _text(payload.get("inner_os_sleep_agenda_focus")),
            "agenda_reason": _text(payload.get("inner_os_sleep_agenda_reason")),
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
            "expressive_style_focus": _text(payload.get("inner_os_sleep_expressive_style_focus")),
            "expressive_style_history_focus": _text(payload.get("inner_os_sleep_expressive_style_history_focus")),
            "banter_style_focus": _text(payload.get("inner_os_sleep_banter_style_focus")),
        }

        carry_strengths = {
            "terrain_reweighting": _float01(payload.get("inner_os_sleep_terrain_reweighting_bias")),
            "agenda": _float01(payload.get("inner_os_sleep_agenda_bias")),
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
        same_insight = same_turn_focus["insight_class"]
        overnight_insight = overnight_focus["insight_class_focus"]
        same_memory = same_turn_focus["memory_class"]
        memory_focus = overnight_focus["memory_class_focus"]
        alignment = {
            "memory_carry_visible": bool(same_memory and memory_focus and same_memory == memory_focus),
            "agenda_carry_visible": bool(same_agenda and overnight_agenda and same_agenda == overnight_agenda),
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
            "partner_registry_visible": bool(
                same_turn_focus["partner_registry_dominant_person"]
                or int(same_turn_focus["partner_registry_total_people"] or 0) > 1
            ),
            "group_thread_registry_visible": bool(
                same_turn_focus["group_thread_dominant_thread"]
                or int(same_turn_focus["group_thread_total_threads"] or 0) > 0
            ),
        }

        return DailyCarrySummary(
            same_turn_focus=same_turn_focus,
            overnight_focus=overnight_focus,
            carry_strengths=carry_strengths,
            active_carry_channels=active_carry_channels,
            dominant_carry_channel=dominant_carry_channel,
            carry_alignment=alignment,
        )
