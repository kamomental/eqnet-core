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
class ContinuitySummary:
    summary_version: str = "v1"
    same_turn: dict[str, Any] = field(default_factory=dict)
    overnight: dict[str, Any] = field(default_factory=dict)
    carry_strengths: dict[str, float] = field(default_factory=dict)
    dominant_carry_channel: str = ""
    transfer: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "same_turn": dict(self.same_turn),
            "overnight": dict(self.overnight),
            "carry_strengths": {str(key): round(float(value), 4) for key, value in self.carry_strengths.items()},
            "dominant_carry_channel": self.dominant_carry_channel,
            "transfer": dict(self.transfer),
        }


class ContinuitySummaryBuilder:
    def build(
        self,
        *,
        interaction_policy_packet: Mapping[str, Any] | None,
        current_state: Mapping[str, Any] | None,
        transfer_summary: Mapping[str, Any] | None = None,
    ) -> ContinuitySummary:
        packet = dict(interaction_policy_packet or {})
        state = dict(current_state or {})
        reaction = dict(packet.get("reaction_vs_overnight_bias") or {})
        same_turn_packet = dict(reaction.get("same_turn") or {})
        overnight_packet = dict(reaction.get("overnight") or {})
        relation_competition = dict(packet.get("relation_competition_state") or {})
        active_relation_table = dict(packet.get("active_relation_table") or {})
        person_registry = dict(state.get("person_registry_snapshot") or {})
        group_thread_registry = dict(state.get("group_thread_registry_snapshot") or {})
        transfer = dict(transfer_summary or {})

        same_turn = {
            "protection_mode": _text(same_turn_packet.get("protection_mode") or packet.get("protection_mode")),
            "memory_write_class": _text(same_turn_packet.get("memory_write_class") or packet.get("memory_write_class")),
            "agenda_state": _text(same_turn_packet.get("agenda_state") or (packet.get("agenda_state") or {}).get("state")),
            "agenda_reason": _text(same_turn_packet.get("agenda_reason") or (packet.get("agenda_state") or {}).get("reason")),
            "agenda_window_state": _text(same_turn_packet.get("agenda_window_state") or (packet.get("agenda_window_state") or {}).get("state")),
            "agenda_window_reason": _text(same_turn_packet.get("agenda_window_reason") or (packet.get("agenda_window_state") or {}).get("reason")),
            "agenda_window_carry_target": _text(same_turn_packet.get("agenda_window_carry_target") or (packet.get("agenda_window_state") or {}).get("carry_target")),
            "agenda_window_deferral_budget": round(_float01(same_turn_packet.get("agenda_window_deferral_budget") or (packet.get("agenda_window_state") or {}).get("deferral_budget")), 4),
            "commitment_target": _text(same_turn_packet.get("commitment_target") or (packet.get("commitment_state") or {}).get("target")),
            "attention_regulation_state": _text((packet.get("attention_regulation_state") or {}).get("state")),
            "grice_guard_state": _text((packet.get("grice_guard_state") or {}).get("state")),
            "relational_style_memory_state": _text((packet.get("relational_style_memory_state") or {}).get("state")),
            "relational_style_playful_ceiling": round(_float01((packet.get("relational_style_memory_state") or {}).get("playful_ceiling")), 4),
            "relational_style_advice_tolerance": round(_float01((packet.get("relational_style_memory_state") or {}).get("advice_tolerance")), 4),
            "relational_banter_style": _text((packet.get("relational_style_memory_state") or {}).get("banter_style")),
            "relational_lexical_variation_bias": round(_float01((packet.get("relational_style_memory_state") or {}).get("lexical_variation_bias")), 4),
            "cultural_conversation_state": _text((packet.get("cultural_conversation_state") or {}).get("state")),
            "cultural_directness_ceiling": round(_float01((packet.get("cultural_conversation_state") or {}).get("directness_ceiling")), 4),
            "cultural_joke_ratio_ceiling": round(_float01((packet.get("cultural_conversation_state") or {}).get("joke_ratio_ceiling")), 4),
            "expressive_style_state": _text(same_turn_packet.get("expressive_style_state") or (packet.get("expressive_style_state") or {}).get("state")),
            "expressive_lightness_room": round(_float01(same_turn_packet.get("expressive_lightness_room") or (packet.get("expressive_style_state") or {}).get("lightness_room")), 4),
            "expressive_continuity_weight": round(_float01(same_turn_packet.get("expressive_continuity_weight") or (packet.get("expressive_style_state") or {}).get("continuity_weight")), 4),
            "expressive_style_history_focus": _text(same_turn_packet.get("expressive_style_history_focus") or packet.get("expressive_style_history_focus")),
            "lightness_budget_state": _text(same_turn_packet.get("lightness_budget_state") or (packet.get("lightness_budget_state") or {}).get("state")),
            "lightness_banter_room": round(_float01(same_turn_packet.get("lightness_budget_banter_room") or (packet.get("lightness_budget_state") or {}).get("banter_room")), 4),
            "banter_style_focus": _text(same_turn_packet.get("banter_style_focus") or packet.get("banter_style_focus")),
            "body_homeostasis_state": _text(same_turn_packet.get("body_homeostasis_state") or (packet.get("body_homeostasis_state") or {}).get("state")),
            "homeostasis_budget_state": _text(same_turn_packet.get("homeostasis_budget_state") or (packet.get("homeostasis_budget_state") or {}).get("state")),
            "relational_continuity_state": _text(same_turn_packet.get("relational_continuity_state") or (packet.get("relational_continuity_state") or {}).get("state")),
            "relation_competition_state": _text(same_turn_packet.get("relation_competition_state") or relation_competition.get("state")),
            "social_topology": _text(same_turn_packet.get("social_topology") or packet.get("social_topology")),
            "social_topology_state": _text(same_turn_packet.get("social_topology_state") or (packet.get("social_topology_state") or {}).get("state")),
            "dominant_person_id": _text(
                same_turn_packet.get("relation_competition_dominant_person_id")
                or relation_competition.get("dominant_person_id")
                or person_registry.get("dominant_person_id")
            ),
            "dominant_group_thread_id": _text(
                same_turn_packet.get("group_thread_dominant_thread_id")
                or group_thread_registry.get("dominant_thread_id")
            ),
            "active_relation_total_people": int(
                active_relation_table.get("total_people")
                or relation_competition.get("total_people")
                or person_registry.get("total_people")
                or 0
            ),
            "active_group_thread_total": int(group_thread_registry.get("total_threads") or 0),
        }
        overnight = {
            "association_focus": _text(overnight_packet.get("association_reweighting_focus") or state.get("association_reweighting_focus")),
            "agenda_focus": _text(overnight_packet.get("agenda_focus") or state.get("agenda_focus")),
            "agenda_reason": _text(overnight_packet.get("agenda_reason") or state.get("agenda_reason")),
            "agenda_window_focus": _text(overnight_packet.get("agenda_window_focus") or state.get("agenda_window_focus")),
            "agenda_window_reason": _text(overnight_packet.get("agenda_window_reason") or state.get("agenda_window_reason")),
            "insight_class_focus": _text(overnight_packet.get("insight_class_focus") or state.get("insight_class_focus")),
            "insight_terrain_shape_target": _text(overnight_packet.get("insight_terrain_shape_target") or state.get("insight_terrain_shape_target")),
            "commitment_target_focus": _text(overnight_packet.get("commitment_target_focus") or state.get("commitment_target_focus")),
            "commitment_followup_focus": _text(overnight_packet.get("commitment_followup_focus") or state.get("commitment_followup_focus")),
            "body_homeostasis_focus": _text(overnight_packet.get("body_homeostasis_focus") or state.get("body_homeostasis_focus")),
            "homeostasis_budget_focus": _text(overnight_packet.get("homeostasis_budget_focus") or state.get("homeostasis_budget_focus")),
            "relational_continuity_focus": _text(overnight_packet.get("relational_continuity_focus") or state.get("relational_continuity_focus")),
            "group_thread_focus": _text(overnight_packet.get("group_thread_focus") or state.get("group_thread_focus")),
            "temperament_focus": _text(overnight_packet.get("temperament_focus") or state.get("temperament_focus")),
            "expressive_style_focus": _text(overnight_packet.get("expressive_style_focus") or state.get("expressive_style_focus")),
            "expressive_style_history_focus": _text(overnight_packet.get("expressive_style_history_focus") or state.get("expressive_style_history_focus")),
            "banter_style_focus": _text(overnight_packet.get("banter_style_focus") or state.get("banter_style_focus")),
        }
        carry_strengths = {
            "initiative_followup": _float01(overnight_packet.get("initiative_followup_bias") or state.get("initiative_followup_bias")),
            "agenda": _float01(overnight_packet.get("agenda_bias") or state.get("agenda_bias")),
            "agenda_window": _float01(overnight_packet.get("agenda_window_bias") or state.get("agenda_window_bias")),
            "commitment": _float01(overnight_packet.get("commitment_carry_bias") or state.get("commitment_carry_bias")),
            "body_homeostasis": _float01(overnight_packet.get("body_homeostasis_carry_bias") or state.get("body_homeostasis_carry_bias")),
            "homeostasis_budget": _float01(overnight_packet.get("homeostasis_budget_bias") or state.get("homeostasis_budget_bias")),
            "relational_continuity": _float01(overnight_packet.get("relational_continuity_carry_bias") or state.get("relational_continuity_carry_bias")),
            "group_thread": _float01(overnight_packet.get("group_thread_carry_bias") or state.get("group_thread_carry_bias")),
            "expressive_style": _float01(overnight_packet.get("expressive_style_carry_bias") or state.get("expressive_style_carry_bias")),
            "expressive_style_history": _float01(overnight_packet.get("expressive_style_history_bias") or state.get("expressive_style_history_bias")),
            "lexical_variation": _float01(overnight_packet.get("lexical_variation_carry_bias") or state.get("lexical_variation_carry_bias")),
            "association": _float01(state.get("association_reweighting_bias")),
            "insight_reframing": _float01(state.get("insight_reframing_bias")),
            "terrain": _float01(state.get("terrain_reweighting_bias")),
            "temperament_forward": _float01(state.get("temperament_forward_bias")),
            "temperament_guard": _float01(state.get("temperament_guard_bias")),
            "temperament_bond": _float01(state.get("temperament_bond_bias")),
            "temperament_recovery": _float01(state.get("temperament_recovery_bias")),
        }
        dominant_carry_channel = ""
        if carry_strengths:
            dominant_carry_channel = max(carry_strengths, key=lambda key: (carry_strengths[key], key))
            if carry_strengths.get(dominant_carry_channel, 0.0) <= 0.0:
                dominant_carry_channel = ""

        transfer_view = {
            "migration_active": bool(transfer.get("migration_active", False)),
            "from_legacy": bool(transfer.get("from_legacy", False)),
            "semantic_seed_visible": bool(transfer.get("semantic_seed_visible", False)),
            "commitment_carry_visible": bool(transfer.get("commitment_carry_visible", False)),
            "target_model_requested": _text(transfer.get("target_model_requested")),
        }
        return ContinuitySummary(
            same_turn=same_turn,
            overnight=overnight,
            carry_strengths=carry_strengths,
            dominant_carry_channel=dominant_carry_channel,
            transfer=transfer_view,
        )
