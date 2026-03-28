from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping

from .schemas import INNER_OS_DISTILLATION_RECORD_SCHEMA


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _hash_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compact_mapping(
    payload: Mapping[str, Any] | None,
    allowed_keys: tuple[str, ...],
) -> dict[str, Any]:
    source = dict(payload or {})
    result: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in source:
            continue
        value = source.get(key)
        if isinstance(value, Mapping):
            result[key] = dict(value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class InnerOSDistillationRecord:
    schema: str = INNER_OS_DISTILLATION_RECORD_SCHEMA
    turn_id: str = ""
    session_id: str = ""
    timestamp_ms: int = 0
    model: dict[str, Any] = field(default_factory=dict)
    input_fingerprint: dict[str, Any] = field(default_factory=dict)
    decision_snapshot: dict[str, Any] = field(default_factory=dict)
    carry_snapshot: dict[str, Any] = field(default_factory=dict)
    output_fingerprint: dict[str, Any] = field(default_factory=dict)
    text_payload: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp_ms": int(self.timestamp_ms),
            "model": dict(self.model),
            "input_fingerprint": dict(self.input_fingerprint),
            "decision_snapshot": dict(self.decision_snapshot),
            "carry_snapshot": dict(self.carry_snapshot),
            "output_fingerprint": dict(self.output_fingerprint),
            "text_payload": dict(self.text_payload),
        }


class InnerOSDistillationRecordBuilder:
    def build(
        self,
        *,
        turn_id: str,
        session_id: str,
        timestamp_ms: int,
        user_text: str | None,
        context_text: str | None,
        response_text: str | None,
        response_meta: Mapping[str, Any] | None,
        interaction_policy_packet: Mapping[str, Any] | None,
        persona_meta_inner_os: Mapping[str, Any] | None,
        include_text: bool = False,
    ) -> InnerOSDistillationRecord:
        response_payload = dict(response_meta or {})
        packet = dict(interaction_policy_packet or {})
        inner_os_meta = dict(persona_meta_inner_os or {})

        model = {
            "name": _text(response_payload.get("model")) or _text(inner_os_meta.get("llm_model")),
            "source": _text(response_payload.get("model_source")) or _text(inner_os_meta.get("llm_model_source")),
            "trace_id": _text(response_payload.get("trace_id")),
            "latency_ms": round(_float(response_payload.get("latency_ms")), 4),
            "confidence": round(_float(response_payload.get("confidence")), 4),
        }

        input_fingerprint = {
            "user_text_sha256": _hash_text(user_text),
            "user_text_length": len(_text(user_text)),
            "context_text_sha256": _hash_text(context_text),
            "context_text_length": len(_text(context_text)),
        }

        protection_snapshot = _compact_mapping(
            packet.get("protection_mode_decision"),
            ("mode", "winner_margin", "scores", "dominant_inputs"),
        )
        if "mode" not in protection_snapshot:
            protection_snapshot["mode"] = _text(packet.get("protection_mode"))

        memory_snapshot = _compact_mapping(
            packet.get("memory_write_class_bias"),
            (
                "selected_class",
                "winner_margin",
                "same_turn_scores",
                "mode_prior",
                "insight_prior",
                "combined_scores",
                "dominant_inputs",
                "protective_lock",
            ),
        )
        if "selected_class" not in memory_snapshot:
            memory_snapshot["selected_class"] = _text(packet.get("memory_write_class"))

        decision_snapshot = {
            "workspace": _compact_mapping(
                inner_os_meta.get("workspace_decision"),
                ("workspace_mode", "winner_margin", "slot_scores", "dominant_inputs"),
            ),
            "protection_mode": protection_snapshot,
            "memory_write_class": memory_snapshot,
            "body_recovery_guard": _compact_mapping(
                packet.get("body_recovery_guard"),
                ("state", "score", "winner_margin", "scores", "dominant_inputs"),
            ),
            "body_homeostasis_state": _compact_mapping(
                packet.get("body_homeostasis_state"),
                ("state", "score", "winner_margin", "scores", "dominant_inputs"),
            ),
            "homeostasis_budget_state": _compact_mapping(
                packet.get("homeostasis_budget_state"),
                ("state", "score", "winner_margin", "scores", "reserve_level", "debt_level", "restoration_bias", "dominant_inputs"),
            ),
            "initiative_readiness": _compact_mapping(
                packet.get("initiative_readiness"),
                ("state", "score", "winner_margin", "scores", "dominant_inputs"),
            ),
            "agenda_state": _compact_mapping(
                packet.get("agenda_state"),
                ("state", "reason", "score", "winner_margin", "dominant_inputs"),
            ),
            "agenda_window_state": _compact_mapping(
                packet.get("agenda_window_state"),
                ("state", "reason", "score", "winner_margin", "deferral_budget", "carry_target", "opportunistic_ok", "dominant_inputs"),
            ),
            "commitment_state": _compact_mapping(
                packet.get("commitment_state"),
                ("state", "target", "accepted_cost", "winner_margin", "dominant_inputs"),
            ),
            "relational_style_memory_state": _compact_mapping(
                packet.get("relational_style_memory_state"),
                (
                    "state",
                    "score",
                    "winner_margin",
                    "warmth_bias",
                    "playful_ceiling",
                    "advice_tolerance",
                    "lexical_familiarity",
                    "lexical_variation_bias",
                    "banter_room",
                    "banter_style",
                    "dominant_person_id",
                    "dominant_inputs",
                ),
            ),
            "cultural_conversation_state": _compact_mapping(
                packet.get("cultural_conversation_state"),
                (
                    "state",
                    "score",
                    "winner_margin",
                    "tone",
                    "directness_ceiling",
                    "joke_ratio_ceiling",
                    "politeness_pressure",
                    "group_attunement",
                    "compaction_bias",
                    "dominant_inputs",
                ),
            ),
            "expressive_style_state": _compact_mapping(
                packet.get("expressive_style_state"),
                ("state", "score", "winner_margin", "lightness_room", "continuity_weight", "dominant_inputs"),
            ),
            "lightness_budget_state": _compact_mapping(
                packet.get("lightness_budget_state"),
                (
                    "state",
                    "score",
                    "winner_margin",
                    "banter_room",
                    "playful_ceiling",
                    "suppression",
                    "advice_tolerance",
                    "dominant_inputs",
                ),
            ),
            "relational_continuity_state": _compact_mapping(
                packet.get("relational_continuity_state"),
                ("state", "score", "winner_margin", "scores", "dominant_inputs"),
            ),
            "social_topology_state": _compact_mapping(
                packet.get("social_topology_state"),
                ("state", "score", "winner_margin", "scores", "visibility_pressure", "threading_pressure", "hierarchy_pressure", "total_people", "dominant_inputs"),
            ),
            "boundary_transform": _compact_mapping(
                inner_os_meta.get("boundary_transform"),
                (
                    "gate_mode",
                    "authority_scope",
                    "transformation_mode",
                    "allowed_acts",
                    "softened_acts",
                    "withheld_acts",
                    "deferred_topics",
                    "do_not_cross",
                    "residual_pressure",
                    "cues",
                ),
            ),
            "residual_reflection": _compact_mapping(
                inner_os_meta.get("residual_reflection"),
                (
                    "mode",
                    "focus",
                    "withheld_acts",
                    "softened_acts",
                    "deferred_topics",
                    "reason_tokens",
                    "strength",
                    "cues",
                ),
            ),
            "group_thread_registry_summary": _compact_mapping(
                packet.get("group_thread_registry_summary"),
                ("dominant_thread_id", "total_threads", "top_thread_ids"),
            ),
        }

        carry_snapshot = {
            "overnight_bias_roles": dict(packet.get("overnight_bias_roles") or {}),
            "reaction_vs_overnight_bias": dict(packet.get("reaction_vs_overnight_bias") or {}),
            "initiative_followup_bias": _compact_mapping(
                packet.get("initiative_followup_bias"),
                ("state", "score", "winner_margin", "scores", "dominant_inputs"),
            ),
            "agenda_carry": {
                "focus": _text(packet.get("agenda_focus")),
                "carry_bias": round(_float(packet.get("agenda_bias")), 4),
                "reason": _text(packet.get("agenda_reason")),
            },
            "agenda_window_carry": {
                "focus": _text(packet.get("agenda_window_focus")),
                "carry_bias": round(_float(packet.get("agenda_window_bias")), 4),
                "reason": _text(packet.get("agenda_window_reason")),
            },
            "expressive_style_history": {
                "focus": _text(packet.get("expressive_style_history_focus")),
                "carry_bias": round(_float(packet.get("expressive_style_history_bias")), 4),
                "banter_style_focus": _text(packet.get("banter_style_focus")),
                "lexical_variation_carry_bias": round(_float(packet.get("lexical_variation_carry_bias")), 4),
            },
            "identity_arc": _compact_mapping(
                inner_os_meta.get("identity_arc"),
                (
                    "arc_kind",
                    "phase",
                    "summary",
                    "dominant_driver",
                    "open_tension",
                    "stability",
                    "memory_anchor",
                    "related_person_id",
                    "group_thread_focus",
                ),
            ),
            "group_relation_arc": _compact_mapping(
                inner_os_meta.get("group_relation_arc"),
                (
                    "arc_kind",
                    "phase",
                    "summary",
                    "boundary_mode",
                    "reentry_window_focus",
                    "group_thread_id",
                    "topology_focus",
                    "dominant_person_id",
                    "stability",
                ),
            ),
            "group_thread_carry": {
                "focus": _text(packet.get("group_thread_focus")),
                "carry_bias": round(_float(packet.get("group_thread_carry_bias")), 4),
            },
        }

        output_fingerprint = {
            "response_text_sha256": _hash_text(response_text),
            "response_text_length": len(_text(response_text)),
            "surface_policy_level": _text(inner_os_meta.get("surface_policy_level")),
            "surface_policy_intent": _text(inner_os_meta.get("surface_policy_intent")),
            "boundary_transform_mode": _text((inner_os_meta.get("boundary_transform") or {}).get("transformation_mode")),
            "residual_reflection_mode": _text((inner_os_meta.get("residual_reflection") or {}).get("mode")),
            "residual_reflection_focus": _text((inner_os_meta.get("residual_reflection") or {}).get("focus")),
            "residual_reflection_strength": round(_float((inner_os_meta.get("residual_reflection") or {}).get("strength")), 4),
            "route": _text(inner_os_meta.get("route")),
            "talk_mode": _text(inner_os_meta.get("talk_mode")),
        }

        text_payload: dict[str, str] = {}
        if include_text:
            text_payload = {
                "user_text": _text(user_text),
                "context_text": _text(context_text),
                "response_text": _text(response_text),
            }

        return InnerOSDistillationRecord(
            turn_id=_text(turn_id),
            session_id=_text(session_id),
            timestamp_ms=int(timestamp_ms),
            model=model,
            input_fingerprint=input_fingerprint,
            decision_snapshot=decision_snapshot,
            carry_snapshot=carry_snapshot,
            output_fingerprint=output_fingerprint,
            text_payload=text_payload,
        )
