from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .daily_carry_summary import DailyCarrySummaryBuilder
from .discussion_thread_registry import summarize_discussion_thread_registry_snapshot
from .schemas import INNER_OS_TRANSFER_PACKAGE_SCHEMA
from .temperament_estimate import derive_temperament_estimate


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


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
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _merge_mapping(
    base: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(overlay or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (int, float)) and float(value) == 0.0:
            continue
        merged[key] = value
    return merged


def _prefixed_gate_value(
    last_gate_context: Mapping[str, Any] | None,
    suffix: str,
    default: Any = None,
) -> Any:
    gate = dict(last_gate_context or {})
    return gate.get(f"inner_os_{suffix}", default)


def _discussion_registry_summary(value: Any) -> dict[str, Any]:
    raw_snapshot = _mapping(value)
    if not raw_snapshot:
        return {}
    summary = _mapping(summarize_discussion_thread_registry_snapshot(raw_snapshot))
    if summary.get("dominant_thread_id"):
        return summary
    return raw_snapshot


@dataclass(frozen=True)
class InnerOSTransferPackage:
    schema: str = INNER_OS_TRANSFER_PACKAGE_SCHEMA
    package_version: str = "v1"
    session_id: str = ""
    turn_id: str = ""
    timestamp_ms: int = 0
    source_model: dict[str, Any] = field(default_factory=dict)
    portable_state: dict[str, Any] = field(default_factory=dict)
    runtime_seed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "package_version": self.package_version,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp_ms": int(self.timestamp_ms),
            "source_model": dict(self.source_model),
            "portable_state": dict(self.portable_state),
            "runtime_seed": dict(self.runtime_seed),
        }


class InnerOSTransferPackageBuilder:
    def __init__(self) -> None:
        self._daily_summary_builder = DailyCarrySummaryBuilder()

    def build(
        self,
        *,
        session_id: str,
        turn_id: str,
        timestamp_ms: int,
        current_state: Mapping[str, Any] | None = None,
        last_gate_context: Mapping[str, Any] | None = None,
        persona_meta_inner_os: Mapping[str, Any] | None = None,
        response_meta: Mapping[str, Any] | None = None,
        nightly_summary: Mapping[str, Any] | None = None,
    ) -> InnerOSTransferPackage:
        current_payload = dict(current_state or {})
        gate_payload = dict(last_gate_context or {})
        inner_os_meta = dict(persona_meta_inner_os or {})
        response_payload = dict(response_meta or {})
        merged_state = self._merged_state(current_payload, gate_payload, inner_os_meta)
        nightly_payload = dict(nightly_summary or {})

        source_model = {
            "name": _text(response_payload.get("model")) or _text(inner_os_meta.get("llm_model")),
            "source": _text(response_payload.get("model_source")) or _text(inner_os_meta.get("llm_model_source")),
            "trace_id": _text(response_payload.get("trace_id")),
        }

        portable_state = {
            "same_turn": {
                "workspace_decision": _compact_mapping(
                    inner_os_meta.get("workspace_decision"),
                    ("workspace_mode", "winner_margin", "slot_scores", "dominant_inputs"),
                ),
                "protection_mode": _compact_mapping(
                    inner_os_meta.get("protection_mode_decision"),
                    ("mode", "strength", "winner_margin", "scores", "dominant_inputs"),
                ),
                "memory_write_class": _compact_mapping(
                    inner_os_meta.get("memory_write_class_bias"),
                    (
                        "selected_class",
                        "winner_margin",
                        "combined_scores",
                        "dominant_inputs",
                        "protective_lock",
                    ),
                ),
                "body_recovery_guard": _compact_mapping(
                    inner_os_meta.get("body_recovery_guard"),
                    ("state", "score", "winner_margin", "scores", "dominant_inputs"),
                ),
                "body_homeostasis_state": _compact_mapping(
                    inner_os_meta.get("body_homeostasis_state"),
                    ("state", "score", "winner_margin", "scores", "dominant_inputs"),
                ),
                "homeostasis_budget_state": _compact_mapping(
                    inner_os_meta.get("homeostasis_budget_state"),
                    ("state", "score", "winner_margin", "scores", "reserve_level", "debt_level", "restoration_bias", "dominant_inputs"),
                ),
                "initiative_readiness": _compact_mapping(
                    inner_os_meta.get("initiative_readiness"),
                    ("state", "score", "winner_margin", "scores", "dominant_inputs"),
                ),
                "agenda_state": _compact_mapping(
                    inner_os_meta.get("agenda_state"),
                    ("state", "reason", "score", "winner_margin", "dominant_inputs"),
                ),
                "agenda_window_state": _compact_mapping(
                    inner_os_meta.get("agenda_window_state"),
                    ("state", "reason", "score", "winner_margin", "deferral_budget", "carry_target", "opportunistic_ok", "dominant_inputs"),
                ),
                "learning_mode_state": _compact_mapping(
                    inner_os_meta.get("learning_mode_state"),
                    ("state", "score", "winner_margin", "probe_room", "update_bias", "dominant_inputs"),
                ),
                "social_experiment_loop_state": _compact_mapping(
                    inner_os_meta.get("social_experiment_loop_state"),
                    (
                        "state",
                        "score",
                        "winner_margin",
                        "hypothesis",
                        "expected_signal",
                        "stop_rule",
                        "probe_intensity",
                        "dominant_inputs",
                    ),
                ),
                "commitment_state": _compact_mapping(
                    inner_os_meta.get("commitment_state"),
                    ("state", "target", "accepted_cost", "winner_margin", "dominant_inputs"),
                ),
                "expressive_style_state": _compact_mapping(
                    inner_os_meta.get("expressive_style_state"),
                    ("state", "score", "winner_margin", "lightness_room", "continuity_weight", "dominant_inputs"),
                ),
                "relational_style_memory_state": _compact_mapping(
                    inner_os_meta.get("relational_style_memory_state"),
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
                "relational_continuity_state": _compact_mapping(
                    inner_os_meta.get("relational_continuity_state"),
                    ("state", "score", "winner_margin", "scores", "dominant_inputs"),
                ),
                "social_topology_state": _compact_mapping(
                    inner_os_meta.get("social_topology_state"),
                    ("state", "score", "winner_margin", "scores", "visibility_pressure", "threading_pressure", "hierarchy_pressure", "total_people", "dominant_inputs"),
                ),
            },
            "carry": {
                "daily_carry_summary": self._daily_carry_summary_dict(nightly_payload),
                "overnight_bias_roles": dict(inner_os_meta.get("overnight_bias_roles") or {}),
                "reaction_vs_overnight_bias": dict(inner_os_meta.get("reaction_vs_overnight_bias") or {}),
                "initiative_followup_bias": _compact_mapping(
                    inner_os_meta.get("initiative_followup_bias"),
                    ("state", "score", "winner_margin", "scores", "dominant_inputs"),
                ),
                "agenda_carry": {
                    "focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_focus")),
                    "carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_bias")), 4),
                    "reason": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_reason")),
                },
                "agenda_window_carry": {
                    "focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_window_focus")),
                    "carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_window_bias")), 4),
                    "reason": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "agenda_window_reason")),
                },
                "learning_mode_carry": {
                    "focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "learning_mode_focus")),
                    "carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "learning_mode_carry_bias")), 4),
                },
                "social_experiment_carry": {
                    "focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "social_experiment_focus")),
                    "carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "social_experiment_carry_bias")), 4),
                },
                "temporal_membrane": {
                    "focus": _text(
                        self._state_value(
                            current_payload,
                            gate_payload,
                            inner_os_meta,
                            "temporal_membrane_focus",
                            default=self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_membrane_mode",
                            ),
                        )
                    ),
                    "timeline_bias": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_timeline_bias",
                                default=self._state_value(
                                    current_payload,
                                    gate_payload,
                                    inner_os_meta,
                                    "temporal_timeline_coherence",
                                ),
                            )
                        ),
                        4,
                    ),
                    "reentry_bias": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_reentry_bias",
                                default=self._state_value(
                                    current_payload,
                                    gate_payload,
                                    inner_os_meta,
                                    "temporal_reentry_pull",
                                ),
                            )
                        ),
                        4,
                    ),
                    "supersession_bias": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_supersession_bias",
                                default=self._state_value(
                                    current_payload,
                                    gate_payload,
                                    inner_os_meta,
                                    "temporal_supersession_pressure",
                                ),
                            )
                        ),
                        4,
                    ),
                    "continuity_bias": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_continuity_bias",
                                default=self._state_value(
                                    current_payload,
                                    gate_payload,
                                    inner_os_meta,
                                    "temporal_continuity_pressure",
                                ),
                            )
                        ),
                        4,
                    ),
                    "relation_reentry_bias": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "temporal_relation_reentry_bias",
                                default=self._state_value(
                                    current_payload,
                                    gate_payload,
                                    inner_os_meta,
                                    "temporal_relation_reentry_pull",
                                ),
                            )
                        ),
                        4,
                    ),
                },
                "style_history": {
                    "expressive_style_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "expressive_style_focus")),
                    "expressive_style_carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "expressive_style_carry_bias")), 4),
                    "expressive_style_history_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "expressive_style_history_focus")),
                    "expressive_style_history_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "expressive_style_history_bias")), 4),
                    "banter_style_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "banter_style_focus")),
                    "lexical_variation_carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "lexical_variation_carry_bias")), 4),
                },
                "identity_arc": _compact_mapping(
                    inner_os_meta.get("identity_arc"),
                    (
                        "arc_kind",
                        "phase",
                        "summary",
                        "dominant_driver",
                        "supporting_drivers",
                        "open_tension",
                        "stability",
                        "memory_anchor",
                        "related_person_id",
                        "group_thread_focus",
                        "long_term_theme_kind",
                        "long_term_theme_focus",
                        "learning_mode_focus",
                        "social_experiment_focus",
                    ),
                ),
                "identity_arc_registry_summary": _mapping(
                    self._state_value(
                        current_payload,
                        gate_payload,
                        inner_os_meta,
                        "identity_arc_registry_summary",
                        default={},
                    )
                ),
                "relation_arc": _compact_mapping(
                    {
                        "arc_kind": _text(inner_os_meta.get("relation_arc_kind")),
                        "phase": _text(inner_os_meta.get("relation_arc_phase")),
                        "summary": _text(inner_os_meta.get("relation_arc_summary")),
                        "open_tension": _text(inner_os_meta.get("relation_arc_open_tension")),
                        "stability": round(_float(inner_os_meta.get("relation_arc_stability")), 4),
                        "related_person_id": _text(inner_os_meta.get("related_person_id")),
                        "group_thread_id": _text(inner_os_meta.get("group_thread_focus")),
                        "learning_mode_focus": _text(inner_os_meta.get("learning_mode_focus")),
                        "social_experiment_focus": _text(inner_os_meta.get("social_experiment_focus")),
                    },
                    (
                        "arc_kind",
                        "phase",
                        "summary",
                        "open_tension",
                        "stability",
                        "related_person_id",
                        "group_thread_id",
                        "learning_mode_focus",
                        "social_experiment_focus",
                    ),
                ),
                "relation_arc_registry_summary": _mapping(
                    self._state_value(
                        current_payload,
                        gate_payload,
                        inner_os_meta,
                        "relation_arc_registry_summary",
                        default={},
                    )
                ),
                "group_relation_arc": _compact_mapping(
                    {
                        "arc_kind": _text(inner_os_meta.get("group_relation_arc_kind")),
                        "phase": _text(inner_os_meta.get("group_relation_arc_phase")),
                        "summary": _text(inner_os_meta.get("group_relation_arc_summary")),
                        "boundary_mode": _text(inner_os_meta.get("group_relation_arc_boundary_mode")),
                        "reentry_window_focus": _text(inner_os_meta.get("group_relation_arc_reentry_window_focus")),
                        "group_thread_id": _text(inner_os_meta.get("group_relation_arc_group_thread_id")),
                        "topology_focus": _text(inner_os_meta.get("group_relation_arc_topology_focus")),
                        "dominant_person_id": _text(inner_os_meta.get("group_relation_arc_dominant_person_id")),
                        "stability": round(_float(inner_os_meta.get("group_relation_arc_stability")), 4),
                    },
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
                "commitment_carry": {
                    "target_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_target_focus")),
                    "state_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_state_focus", default="waver")) or "waver",
                    "carry_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_carry_bias")), 4),
                    "followup_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_followup_focus")),
                    "mode_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_mode_focus")),
                    "carry_reason": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "commitment_carry_reason")),
                },
                "temperament_estimate": derive_temperament_estimate(merged_state).to_dict(),
                "relationship_summary": {
                    "related_person_id": _text(inner_os_meta.get("related_person_id")),
                    "attachment": round(_float(inner_os_meta.get("attachment")), 4),
                    "familiarity": round(_float(inner_os_meta.get("familiarity")), 4),
                    "trust_memory": round(_float(inner_os_meta.get("trust_memory")), 4),
                    "relation_seed_summary": _text(inner_os_meta.get("relation_seed_summary")),
                    "partner_address_hint": _text(inner_os_meta.get("partner_address_hint")),
                    "partner_timing_hint": _text(inner_os_meta.get("partner_timing_hint")),
                    "partner_stance_hint": _text(inner_os_meta.get("partner_stance_hint")),
                    "partner_social_interpretation": _text(inner_os_meta.get("partner_social_interpretation")),
                },
                "relationship_registry_summary": _mapping(
                    self._state_value(
                        current_payload,
                        gate_payload,
                        inner_os_meta,
                        "person_registry_snapshot",
                        default={},
                    )
                ),
                "group_thread_registry_summary": _mapping(
                    self._state_value(
                        current_payload,
                        gate_payload,
                        inner_os_meta,
                        "group_thread_registry_snapshot",
                        default={},
                    )
                ),
                "discussion_thread_registry_summary": _discussion_registry_summary(
                    self._state_value(
                        current_payload,
                        gate_payload,
                        inner_os_meta,
                        "discussion_thread_registry_snapshot",
                        default={},
                    )
                ),
                "autobiographical_thread": {
                    "mode": _text(
                        self._state_value(
                            current_payload,
                            gate_payload,
                            inner_os_meta,
                            "autobiographical_thread_mode",
                        )
                    ),
                    "anchor": _text(
                        self._state_value(
                            current_payload,
                            gate_payload,
                            inner_os_meta,
                            "autobiographical_thread_anchor",
                        )
                    ),
                    "focus": _text(
                        self._state_value(
                            current_payload,
                            gate_payload,
                            inner_os_meta,
                            "autobiographical_thread_focus",
                        )
                    ),
                    "strength": round(
                        _float(
                            self._state_value(
                                current_payload,
                                gate_payload,
                                inner_os_meta,
                                "autobiographical_thread_strength",
                            )
                        ),
                        4,
                    ),
                },
                "monument_carry": {
                    "memory_anchor": _text(inner_os_meta.get("memory_anchor")),
                    "semantic_seed_focus": _text(inner_os_meta.get("semantic_seed_focus")),
                    "semantic_seed_anchor": _text(inner_os_meta.get("semantic_seed_anchor")),
                    "semantic_seed_strength": round(_float(inner_os_meta.get("semantic_seed_strength")), 4),
                    "semantic_seed_recurrence": round(_float(inner_os_meta.get("semantic_seed_recurrence")), 4),
                    "long_term_theme_focus": _text(inner_os_meta.get("long_term_theme_focus")),
                    "long_term_theme_anchor": _text(inner_os_meta.get("long_term_theme_anchor")),
                    "long_term_theme_kind": _text(inner_os_meta.get("long_term_theme_kind")),
                    "long_term_theme_summary": _text(inner_os_meta.get("long_term_theme_summary")),
                    "long_term_theme_strength": round(_float(inner_os_meta.get("long_term_theme_strength")), 4),
                    "relation_seed_summary": _text(inner_os_meta.get("relation_seed_summary")),
                    "relation_seed_strength": round(_float(inner_os_meta.get("relation_seed_strength")), 4),
                    "monument_salience": round(_float(inner_os_meta.get("monument_salience")), 4),
                    "monument_kind": _text(inner_os_meta.get("monument_kind")),
                },
            },
            "continuity": {
                "terrain_bias": {
                    "terrain_reweighting_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "terrain_reweighting_bias")), 4),
                    "association_reweighting_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "association_reweighting_bias")), 4),
                    "association_reweighting_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "association_reweighting_focus")),
                    "association_reweighting_reason": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "association_reweighting_reason")),
                },
                "insight_bias": {
                    "insight_reframing_bias": round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "insight_reframing_bias")), 4),
                    "insight_class_focus": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "insight_class_focus")),
                    "insight_terrain_shape_target": _text(self._state_value(current_payload, gate_payload, inner_os_meta, "insight_terrain_shape_target")),
                    "insight_link_counts": dict(self._state_value(current_payload, gate_payload, inner_os_meta, "insight_link_counts", default={}) or {}),
                    "insight_class_counts": dict(self._state_value(current_payload, gate_payload, inner_os_meta, "insight_class_counts", default={}) or {}),
                },
                "temperament_bias": {
                    "forward_bias": round(_float(merged_state.get("temperament_forward_bias")), 4),
                    "guard_bias": round(_float(merged_state.get("temperament_guard_bias")), 4),
                    "bond_bias": round(_float(merged_state.get("temperament_bond_bias")), 4),
                    "recovery_bias": round(_float(merged_state.get("temperament_recovery_bias")), 4),
                },
            },
        }
        if not portable_state["same_turn"]["homeostasis_budget_state"]:
            budget_focus = _text(self._state_value(current_payload, gate_payload, inner_os_meta, "homeostasis_budget_focus"))
            budget_bias = round(_float(self._state_value(current_payload, gate_payload, inner_os_meta, "homeostasis_budget_bias")), 4)
            if budget_focus:
                portable_state["same_turn"]["homeostasis_budget_state"] = {
                    "state": budget_focus,
                    "score": budget_bias,
                    "winner_margin": 0.0,
                }

        runtime_seed = self._build_runtime_seed(current_payload, gate_payload, inner_os_meta)

        return InnerOSTransferPackage(
            session_id=_text(session_id),
            turn_id=_text(turn_id),
            timestamp_ms=int(timestamp_ms),
            source_model=source_model,
            portable_state=portable_state,
            runtime_seed=runtime_seed,
        )

    def to_runtime_seed(self, payload: Mapping[str, Any] | InnerOSTransferPackage) -> dict[str, Any]:
        package = payload.to_dict() if isinstance(payload, InnerOSTransferPackage) else dict(payload or {})
        seed = dict(package.get("runtime_seed") or package.get("state_seed") or {})
        legacy_identity_arc = dict(package.get("identity_arc_summary") or {})
        legacy_identity_arc_registry = dict(package.get("identity_arc_registry_summary") or {})
        legacy_relation_arc = dict(package.get("relation_arc_summary") or {})
        legacy_relation_arc_registry = dict(package.get("relation_arc_registry_summary") or {})
        legacy_group_relation_arc = dict(package.get("group_relation_arc_summary") or {})

        def _seed_value(key: str, *legacy_keys: str, default: Any = None) -> Any:
            if key in seed and seed.get(key) is not None:
                return seed.get(key)
            for legacy_key in legacy_keys:
                if legacy_key in package and package.get(legacy_key) is not None:
                    return package.get(legacy_key)
            return default

        return {
            "prev_qualia": list(_seed_value("prev_qualia", "prev_qualia", "inner_os_prev_qualia", default=[])),
            "prev_qualia_habituation": list(_seed_value("prev_qualia_habituation", "prev_qualia_habituation", "inner_os_prev_qualia_habituation", default=[])),
            "prev_protection_grad_x": list(_seed_value("prev_protection_grad_x", "prev_protection_grad_x", "inner_os_prev_protection_grad_x", default=[])),
            "prev_affective_position": dict(_seed_value("prev_affective_position", "prev_affective_position", "inner_os_prev_affective_position", default={})),
            "affective_terrain_state": dict(_seed_value("affective_terrain_state", "affective_terrain_state", "inner_os_affective_terrain_state", default={})),
            "association_graph_state": dict(_seed_value("association_graph_state", "association_graph_state", "inner_os_association_graph_state", default={})),
            "terrain_reweighting_bias": _float(_seed_value("terrain_reweighting_bias", "terrain_reweighting_bias")),
            "association_reweighting_bias": _float(_seed_value("association_reweighting_bias", "association_reweighting_bias")),
            "association_reweighting_focus": _text(_seed_value("association_reweighting_focus", "association_reweighting_focus")),
            "association_reweighting_reason": _text(_seed_value("association_reweighting_reason", "association_reweighting_reason")),
            "insight_reframing_bias": _float(_seed_value("insight_reframing_bias", "insight_reframing_bias")),
            "insight_class_focus": _text(_seed_value("insight_class_focus", "insight_class_focus")),
            "insight_terrain_shape_target": _text(_seed_value("insight_terrain_shape_target", "insight_terrain_shape_target")),
            "insight_link_counts": dict(_seed_value("insight_link_counts", "insight_link_counts", default={}) or {}),
            "insight_class_counts": dict(_seed_value("insight_class_counts", "insight_class_counts", default={}) or {}),
            "initiative_followup_bias": _float(_seed_value("initiative_followup_bias", "initiative_followup_bias")),
            "initiative_followup_state": _text(_seed_value("initiative_followup_state", "initiative_followup_state", default="hold")) or "hold",
            "agenda_focus": _text(_seed_value("agenda_focus", "agenda_focus")),
            "agenda_bias": _float(_seed_value("agenda_bias", "agenda_bias")),
            "agenda_reason": _text(_seed_value("agenda_reason", "agenda_reason")),
            "agenda_window_focus": _text(_seed_value("agenda_window_focus", "agenda_window_focus")),
            "agenda_window_bias": _float(_seed_value("agenda_window_bias", "agenda_window_bias")),
            "agenda_window_reason": _text(_seed_value("agenda_window_reason", "agenda_window_reason")),
            "learning_mode_focus": _text(_seed_value("learning_mode_focus", "learning_mode_focus")),
            "learning_mode_carry_bias": _float(_seed_value("learning_mode_carry_bias", "learning_mode_carry_bias")),
            "social_experiment_focus": _text(_seed_value("social_experiment_focus", "social_experiment_focus")),
            "social_experiment_carry_bias": _float(_seed_value("social_experiment_carry_bias", "social_experiment_carry_bias")),
            "temporal_membrane_focus": _text(_seed_value("temporal_membrane_focus", "temporal_membrane_focus", "temporal_membrane_mode")),
            "temporal_timeline_bias": _float(_seed_value("temporal_timeline_bias", "temporal_timeline_bias", "temporal_timeline_coherence")),
            "temporal_reentry_bias": _float(_seed_value("temporal_reentry_bias", "temporal_reentry_bias", "temporal_reentry_pull")),
            "temporal_supersession_bias": _float(_seed_value("temporal_supersession_bias", "temporal_supersession_bias", "temporal_supersession_pressure")),
            "temporal_continuity_bias": _float(_seed_value("temporal_continuity_bias", "temporal_continuity_bias", "temporal_continuity_pressure")),
            "temporal_relation_reentry_bias": _float(_seed_value("temporal_relation_reentry_bias", "temporal_relation_reentry_bias", "temporal_relation_reentry_pull")),
            "identity_arc_kind": _text(_seed_value("identity_arc_kind", "identity_arc_kind", default=legacy_identity_arc.get("arc_kind"))),
            "identity_arc_phase": _text(_seed_value("identity_arc_phase", "identity_arc_phase", default=legacy_identity_arc.get("phase"))),
            "identity_arc_summary": _text(_seed_value("identity_arc_summary", "identity_arc_summary", default=legacy_identity_arc.get("summary"))),
            "identity_arc_open_tension": _text(_seed_value("identity_arc_open_tension", "identity_arc_open_tension", default=legacy_identity_arc.get("open_tension"))),
            "identity_arc_stability": _float(_seed_value("identity_arc_stability", "identity_arc_stability", default=legacy_identity_arc.get("stability"))),
            "identity_arc_registry_summary": dict(_seed_value("identity_arc_registry_summary", "identity_arc_registry_summary", default=legacy_identity_arc_registry) or {}),
            "relation_arc_kind": _text(_seed_value("relation_arc_kind", "relation_arc_kind", default=legacy_relation_arc.get("arc_kind"))),
            "relation_arc_phase": _text(_seed_value("relation_arc_phase", "relation_arc_phase", default=legacy_relation_arc.get("phase"))),
            "relation_arc_summary": _text(_seed_value("relation_arc_summary", "relation_arc_summary", default=legacy_relation_arc.get("summary"))),
            "relation_arc_open_tension": _text(_seed_value("relation_arc_open_tension", "relation_arc_open_tension", default=legacy_relation_arc.get("open_tension"))),
            "relation_arc_stability": _float(_seed_value("relation_arc_stability", "relation_arc_stability", default=legacy_relation_arc.get("stability"))),
            "relation_arc_registry_summary": dict(_seed_value("relation_arc_registry_summary", "relation_arc_registry_summary", default=legacy_relation_arc_registry) or {}),
            "group_relation_arc_kind": _text(_seed_value("group_relation_arc_kind", "group_relation_arc_kind", default=legacy_group_relation_arc.get("arc_kind"))),
            "group_relation_arc_phase": _text(_seed_value("group_relation_arc_phase", "group_relation_arc_phase", default=legacy_group_relation_arc.get("phase"))),
            "group_relation_arc_summary": _text(_seed_value("group_relation_arc_summary", "group_relation_arc_summary", default=legacy_group_relation_arc.get("summary"))),
            "group_relation_arc_boundary_mode": _text(_seed_value("group_relation_arc_boundary_mode", "group_relation_arc_boundary_mode", default=legacy_group_relation_arc.get("boundary_mode"))),
            "group_relation_arc_reentry_window_focus": _text(_seed_value("group_relation_arc_reentry_window_focus", "group_relation_arc_reentry_window_focus", default=legacy_group_relation_arc.get("reentry_window_focus"))),
            "group_relation_arc_group_thread_id": _text(_seed_value("group_relation_arc_group_thread_id", "group_relation_arc_group_thread_id", default=legacy_group_relation_arc.get("group_thread_id"))),
            "group_relation_arc_topology_focus": _text(_seed_value("group_relation_arc_topology_focus", "group_relation_arc_topology_focus", default=legacy_group_relation_arc.get("topology_focus"))),
            "group_relation_arc_dominant_person_id": _text(_seed_value("group_relation_arc_dominant_person_id", "group_relation_arc_dominant_person_id", default=legacy_group_relation_arc.get("dominant_person_id"))),
            "group_relation_arc_stability": _float(_seed_value("group_relation_arc_stability", "group_relation_arc_stability", default=legacy_group_relation_arc.get("stability"))),
            "commitment_target_focus": _text(_seed_value("commitment_target_focus", "commitment_target_focus")),
            "commitment_state_focus": _text(_seed_value("commitment_state_focus", "commitment_state_focus", default="waver")) or "waver",
            "commitment_carry_bias": _float(_seed_value("commitment_carry_bias", "commitment_carry_bias")),
            "commitment_followup_focus": _text(_seed_value("commitment_followup_focus", "commitment_followup_focus")),
            "commitment_mode_focus": _text(_seed_value("commitment_mode_focus", "commitment_mode_focus")),
            "commitment_carry_reason": _text(_seed_value("commitment_carry_reason", "commitment_carry_reason")),
            "body_homeostasis_focus": _text(_seed_value("body_homeostasis_focus", "body_homeostasis_focus")),
            "body_homeostasis_carry_bias": _float(_seed_value("body_homeostasis_carry_bias", "body_homeostasis_carry_bias")),
            "homeostasis_budget_state": _mapping(_seed_value("homeostasis_budget_state", "homeostasis_budget_state", default={})),
            "social_topology_state": _mapping(_seed_value("social_topology_state", "social_topology_state", default={})),
            "homeostasis_budget_focus": _text(_seed_value("homeostasis_budget_focus", "homeostasis_budget_focus")),
            "homeostasis_budget_bias": _float(_seed_value("homeostasis_budget_bias", "homeostasis_budget_bias")),
            "relational_continuity_focus": _text(_seed_value("relational_continuity_focus", "relational_continuity_focus")),
            "relational_continuity_carry_bias": _float(_seed_value("relational_continuity_carry_bias", "relational_continuity_carry_bias")),
            "group_thread_registry_snapshot": dict(_seed_value("group_thread_registry_snapshot", "group_thread_registry_snapshot", default={}) or {}),
            "discussion_thread_registry_snapshot": dict(
                _seed_value(
                    "discussion_thread_registry_snapshot",
                    "discussion_thread_registry_snapshot",
                    default=_seed_value(
                        "discussion_thread_registry_summary",
                        "discussion_thread_registry_summary",
                        default={},
                    ),
                )
                or {}
            ),
            "autobiographical_thread_mode": _text(_seed_value("autobiographical_thread_mode", "autobiographical_thread_mode")),
            "autobiographical_thread_anchor": _text(_seed_value("autobiographical_thread_anchor", "autobiographical_thread_anchor")),
            "autobiographical_thread_focus": _text(_seed_value("autobiographical_thread_focus", "autobiographical_thread_focus")),
            "autobiographical_thread_strength": _float(_seed_value("autobiographical_thread_strength", "autobiographical_thread_strength")),
            "group_thread_focus": _text(_seed_value("group_thread_focus", "group_thread_focus")),
            "group_thread_carry_bias": _float(_seed_value("group_thread_carry_bias", "group_thread_carry_bias")),
            "expressive_style_focus": _text(_seed_value("expressive_style_focus", "expressive_style_focus")),
            "expressive_style_carry_bias": _float(_seed_value("expressive_style_carry_bias", "expressive_style_carry_bias")),
            "expressive_style_history_focus": _text(_seed_value("expressive_style_history_focus", "expressive_style_history_focus")),
            "expressive_style_history_bias": _float(_seed_value("expressive_style_history_bias", "expressive_style_history_bias")),
            "banter_style_focus": _text(_seed_value("banter_style_focus", "banter_style_focus")),
            "lexical_variation_carry_bias": _float(_seed_value("lexical_variation_carry_bias", "lexical_variation_carry_bias")),
            "person_registry_snapshot": dict(_seed_value("person_registry_snapshot", "person_registry_snapshot", default={}) or {}),
            "temperament_forward_trace": _float(_seed_value("temperament_forward_trace", "temperament_forward_trace")),
            "temperament_guard_trace": _float(_seed_value("temperament_guard_trace", "temperament_guard_trace")),
            "temperament_bond_trace": _float(_seed_value("temperament_bond_trace", "temperament_bond_trace")),
            "temperament_recovery_trace": _float(_seed_value("temperament_recovery_trace", "temperament_recovery_trace")),
            "temperament_forward_bias": _float(_seed_value("temperament_forward_bias", "temperament_forward_bias")),
            "temperament_guard_bias": _float(_seed_value("temperament_guard_bias", "temperament_guard_bias")),
            "temperament_bond_bias": _float(_seed_value("temperament_bond_bias", "temperament_bond_bias")),
            "temperament_recovery_bias": _float(_seed_value("temperament_recovery_bias", "temperament_recovery_bias")),
        }

    def to_working_memory_seed(self, payload: Mapping[str, Any] | InnerOSTransferPackage) -> dict[str, Any]:
        package = payload.to_dict() if isinstance(payload, InnerOSTransferPackage) else dict(payload or {})
        portable_state = dict(package.get("portable_state") or {})
        carry = dict(portable_state.get("carry") or {})
        runtime_seed = dict(package.get("runtime_seed") or package.get("state_seed") or {})
        legacy_seed = dict(package.get("working_memory_seed") or {})
        monument = _merge_mapping(legacy_seed, carry.get("monument_carry"))
        relationship = _merge_mapping(legacy_seed, carry.get("relationship_summary"))
        identity_arc = _mapping(dict(carry.get("identity_arc") or {}) or dict(package.get("identity_arc_summary") or {}))
        relation_arc = _mapping(dict(carry.get("relation_arc") or {}) or dict(package.get("relation_arc_summary") or {}))
        group_relation_arc = _mapping(dict(carry.get("group_relation_arc") or {}) or dict(package.get("group_relation_arc_summary") or {}))
        identity_arc_registry_summary = _mapping(
            dict(carry.get("identity_arc_registry_summary") or {})
            or dict(runtime_seed.get("identity_arc_registry_summary") or {})
            or dict(package.get("identity_arc_registry_summary") or {})
        )
        relation_arc_registry_summary = _mapping(
            dict(carry.get("relation_arc_registry_summary") or {})
            or dict(runtime_seed.get("relation_arc_registry_summary") or {})
            or dict(package.get("relation_arc_registry_summary") or {})
        )
        relationship_registry = dict(carry.get("relationship_registry_summary") or {})
        group_thread_registry = dict(carry.get("group_thread_registry_summary") or runtime_seed.get("group_thread_registry_snapshot") or {})
        discussion_thread_registry = dict(
            carry.get("discussion_thread_registry_summary")
            or runtime_seed.get("discussion_thread_registry_snapshot")
            or {}
        )
        autobiographical_thread = _mapping(
            dict(carry.get("autobiographical_thread") or {})
            or {
                "mode": runtime_seed.get("autobiographical_thread_mode"),
                "anchor": runtime_seed.get("autobiographical_thread_anchor"),
                "focus": runtime_seed.get("autobiographical_thread_focus"),
                "strength": runtime_seed.get("autobiographical_thread_strength"),
            }
        )
        related_person_ids = [
            str(item).strip()
            for item in list(relationship_registry.get("top_person_ids") or [])
            if str(item).strip()
        ]
        return {
            "semantic_seed_focus": _text(monument.get("semantic_seed_focus")),
            "semantic_seed_anchor": _text(monument.get("semantic_seed_anchor") or monument.get("memory_anchor")),
            "semantic_seed_strength": _float(monument.get("semantic_seed_strength")),
            "semantic_seed_recurrence": _float(monument.get("semantic_seed_recurrence")),
            "long_term_theme_focus": _text(monument.get("long_term_theme_focus")),
            "long_term_theme_anchor": _text(monument.get("long_term_theme_anchor") or monument.get("memory_anchor")),
            "long_term_theme_strength": _float(monument.get("long_term_theme_strength")),
            "long_term_theme_kind": _text(monument.get("long_term_theme_kind")),
            "long_term_theme_summary": _text(monument.get("long_term_theme_summary")),
            "relation_seed_summary": _text(monument.get("relation_seed_summary") or relationship.get("relation_seed_summary")),
            "relation_seed_strength": _float(monument.get("relation_seed_strength")),
            "related_person_id": _text(relationship.get("related_person_id")),
            "related_person_ids": related_person_ids,
            "attachment": _float(relationship.get("attachment")),
            "familiarity": _float(relationship.get("familiarity")),
            "trust_memory": _float(relationship.get("trust_memory")),
            "partner_address_hint": _text(relationship.get("partner_address_hint")),
            "partner_timing_hint": _text(relationship.get("partner_timing_hint")),
            "partner_stance_hint": _text(relationship.get("partner_stance_hint")),
            "partner_social_interpretation": _text(relationship.get("partner_social_interpretation")),
            "group_thread_id": _text(
                group_thread_registry.get("dominant_thread_id")
                or runtime_seed.get("group_thread_id")
            ),
            "discussion_thread_id": _text(discussion_thread_registry.get("dominant_thread_id")),
            "discussion_anchor": _text(discussion_thread_registry.get("dominant_anchor")),
            "discussion_issue_state": _text(discussion_thread_registry.get("dominant_issue_state")),
            "autobiographical_thread_mode": _text(autobiographical_thread.get("mode")),
            "autobiographical_thread_anchor": _text(autobiographical_thread.get("anchor")),
            "autobiographical_thread_focus": _text(autobiographical_thread.get("focus")),
            "autobiographical_thread_strength": _float(autobiographical_thread.get("strength")),
            "identity_arc_kind": _text(identity_arc.get("arc_kind")),
            "identity_arc_phase": _text(identity_arc.get("phase")),
            "identity_arc_summary": _text(identity_arc.get("summary")),
            "identity_arc_open_tension": _text(identity_arc.get("open_tension")),
            "identity_arc_stability": _float(identity_arc.get("stability")),
            "identity_arc_registry_summary": identity_arc_registry_summary,
            "relation_arc_kind": _text(relation_arc.get("arc_kind")),
            "relation_arc_phase": _text(relation_arc.get("phase")),
            "relation_arc_summary": _text(relation_arc.get("summary")),
            "relation_arc_open_tension": _text(relation_arc.get("open_tension")),
            "relation_arc_stability": _float(relation_arc.get("stability")),
            "relation_arc_registry_summary": relation_arc_registry_summary,
            "group_relation_arc_kind": _text(group_relation_arc.get("arc_kind")),
            "group_relation_arc_phase": _text(group_relation_arc.get("phase")),
            "group_relation_arc_summary": _text(group_relation_arc.get("summary")),
            "group_relation_arc_boundary_mode": _text(group_relation_arc.get("boundary_mode")),
            "group_relation_arc_reentry_window_focus": _text(group_relation_arc.get("reentry_window_focus")),
            "group_relation_arc_group_thread_id": _text(group_relation_arc.get("group_thread_id")),
            "group_relation_arc_topology_focus": _text(group_relation_arc.get("topology_focus")),
            "group_relation_arc_dominant_person_id": _text(group_relation_arc.get("dominant_person_id")),
            "group_relation_arc_stability": _float(group_relation_arc.get("stability")),
        }

    def normalize(self, payload: Mapping[str, Any] | InnerOSTransferPackage) -> dict[str, Any]:
        package = payload.to_dict() if isinstance(payload, InnerOSTransferPackage) else dict(payload or {})
        portable_state = dict(package.get("portable_state") or {})
        same_turn = dict(portable_state.get("same_turn") or {})
        carry = dict(portable_state.get("carry") or {})
        continuity = dict(portable_state.get("continuity") or {})
        source_model_raw = package.get("source_model")
        source_model: dict[str, Any]
        if isinstance(source_model_raw, Mapping):
            source_model = {
                "name": _text(source_model_raw.get("name")),
                "source": _text(source_model_raw.get("source")),
                "trace_id": _text(source_model_raw.get("trace_id")),
            }
        else:
            source_model = {
                "name": _text(source_model_raw),
                "source": _text(package.get("model_source")),
                "trace_id": _text(package.get("trace_id")),
            }

        daily_carry_summary = (
            dict(carry.get("daily_carry_summary") or {})
            or dict(package.get("daily_carry_summary") or {})
            or dict(package.get("inner_os_daily_carry_summary") or {})
        )
        if not daily_carry_summary:
            daily_carry_summary = self._daily_carry_summary_dict({})

        runtime_seed = self.to_runtime_seed(package)
        monument_carry = self.to_working_memory_seed(package)
        relationship_registry_summary = _mapping(
            dict(carry.get("relationship_registry_summary") or {})
            or dict(package.get("person_registry_snapshot") or {})
            or dict(package.get("person_registry") or {})
        )
        group_thread_registry_summary = _mapping(
            dict(carry.get("group_thread_registry_summary") or {})
            or dict(package.get("group_thread_registry_snapshot") or {})
            or dict(package.get("group_thread_registry") or {})
        )
        discussion_thread_registry_summary = _mapping(
            dict(carry.get("discussion_thread_registry_summary") or {})
            or dict(package.get("discussion_thread_registry_snapshot") or {})
            or dict(package.get("discussion_thread_registry_summary") or {})
        )
        if discussion_thread_registry_summary and not discussion_thread_registry_summary.get("dominant_thread_id"):
            discussion_thread_registry_summary = _mapping(
                summarize_discussion_thread_registry_snapshot(discussion_thread_registry_summary)
            )
        style_history = dict(carry.get("style_history") or {})
        temporal_membrane = _mapping(
            dict(carry.get("temporal_membrane") or {})
            or {
                "focus": _text(
                    package.get("temporal_membrane_focus")
                    or package.get("temporal_membrane_mode")
                    or runtime_seed.get("temporal_membrane_focus")
                ),
                "timeline_bias": _float(
                    package.get("temporal_timeline_bias")
                    or package.get("temporal_timeline_coherence")
                    or runtime_seed.get("temporal_timeline_bias")
                ),
                "reentry_bias": _float(
                    package.get("temporal_reentry_bias")
                    or package.get("temporal_reentry_pull")
                    or runtime_seed.get("temporal_reentry_bias")
                ),
                "supersession_bias": _float(
                    package.get("temporal_supersession_bias")
                    or package.get("temporal_supersession_pressure")
                    or runtime_seed.get("temporal_supersession_bias")
                ),
                "continuity_bias": _float(
                    package.get("temporal_continuity_bias")
                    or package.get("temporal_continuity_pressure")
                    or runtime_seed.get("temporal_continuity_bias")
                ),
                "relation_reentry_bias": _float(
                    package.get("temporal_relation_reentry_bias")
                    or package.get("temporal_relation_reentry_pull")
                    or runtime_seed.get("temporal_relation_reentry_bias")
                ),
            }
        )
        autobiographical_thread = _mapping(
            dict(carry.get("autobiographical_thread") or {})
            or {
                "mode": _text(
                    package.get("autobiographical_thread_mode")
                    or runtime_seed.get("autobiographical_thread_mode")
                ),
                "anchor": _text(
                    package.get("autobiographical_thread_anchor")
                    or runtime_seed.get("autobiographical_thread_anchor")
                ),
                "focus": _text(
                    package.get("autobiographical_thread_focus")
                    or runtime_seed.get("autobiographical_thread_focus")
                ),
                "strength": _float(
                    package.get("autobiographical_thread_strength")
                    or runtime_seed.get("autobiographical_thread_strength")
                ),
            }
        )
        identity_arc = dict(carry.get("identity_arc") or package.get("identity_arc_summary") or {})
        identity_arc_registry_summary = dict(
            carry.get("identity_arc_registry_summary")
            or package.get("identity_arc_registry_summary")
            or {}
        )
        relation_arc = dict(carry.get("relation_arc") or package.get("relation_arc_summary") or {})
        group_relation_arc = dict(carry.get("group_relation_arc") or package.get("group_relation_arc_summary") or {})
        relation_arc_registry_summary = dict(
            carry.get("relation_arc_registry_summary")
            or package.get("relation_arc_registry_summary")
            or {}
        )
        relationship_summary = {
            "related_person_id": _text(monument_carry.get("related_person_id")),
            "attachment": round(_float(monument_carry.get("attachment")), 4),
            "familiarity": round(_float(monument_carry.get("familiarity")), 4),
            "trust_memory": round(_float(monument_carry.get("trust_memory")), 4),
            "relation_seed_summary": _text(monument_carry.get("relation_seed_summary")),
            "partner_address_hint": _text(monument_carry.get("partner_address_hint")),
            "partner_timing_hint": _text(monument_carry.get("partner_timing_hint")),
            "partner_stance_hint": _text(monument_carry.get("partner_stance_hint")),
            "partner_social_interpretation": _text(monument_carry.get("partner_social_interpretation")),
        }
        monument_summary = {
            "memory_anchor": _text(package.get("memory_anchor") or monument_carry.get("semantic_seed_anchor")),
            "semantic_seed_focus": _text(monument_carry.get("semantic_seed_focus")),
            "semantic_seed_anchor": _text(monument_carry.get("semantic_seed_anchor")),
            "semantic_seed_strength": round(_float(monument_carry.get("semantic_seed_strength")), 4),
            "semantic_seed_recurrence": round(_float(monument_carry.get("semantic_seed_recurrence")), 4),
            "long_term_theme_focus": _text(monument_carry.get("long_term_theme_focus")),
            "long_term_theme_anchor": _text(monument_carry.get("long_term_theme_anchor")),
            "long_term_theme_kind": _text(monument_carry.get("long_term_theme_kind")),
            "long_term_theme_summary": _text(monument_carry.get("long_term_theme_summary")),
            "long_term_theme_strength": round(_float(monument_carry.get("long_term_theme_strength")), 4),
            "relation_seed_summary": _text(monument_carry.get("relation_seed_summary")),
            "relation_seed_strength": round(_float(monument_carry.get("relation_seed_strength")), 4),
            "monument_salience": round(_float(package.get("monument_salience") or carry.get("monument_salience")), 4),
            "monument_kind": _text(package.get("monument_kind") or carry.get("monument_kind")),
        }

        existing_migration = dict(package.get("migration") or {})
        migration_applied = bool(existing_migration.get("applied")) or (
            _text(package.get("schema")) != INNER_OS_TRANSFER_PACKAGE_SCHEMA
            or _text(package.get("package_version")) not in {"", "v1"}
            or not isinstance(source_model_raw, Mapping)
            or "portable_state" not in package
            or "runtime_seed" not in package
            or "working_memory_seed" in package
            or "daily_carry_summary" in package
            or "inner_os_daily_carry_summary" in package
        )

        return {
            "schema": INNER_OS_TRANSFER_PACKAGE_SCHEMA,
            "package_version": "v1",
            "session_id": _text(package.get("session_id")),
            "turn_id": _text(package.get("turn_id")),
            "timestamp_ms": int(_float(package.get("timestamp_ms"), 0.0)),
            "source_model": source_model,
            "portable_state": {
                "same_turn": same_turn,
                "carry": {
                    "daily_carry_summary": daily_carry_summary,
                    "overnight_bias_roles": dict(carry.get("overnight_bias_roles") or {}),
                    "reaction_vs_overnight_bias": dict(carry.get("reaction_vs_overnight_bias") or {}),
                    "initiative_followup_bias": dict(carry.get("initiative_followup_bias") or {}),
                    "agenda_carry": dict(carry.get("agenda_carry") or {}),
                    "temporal_membrane": temporal_membrane,
                    "style_history": style_history,
                    "identity_arc": identity_arc,
                    "identity_arc_registry_summary": identity_arc_registry_summary,
                    "relation_arc": relation_arc,
                    "relation_arc_registry_summary": relation_arc_registry_summary,
                    "group_relation_arc": group_relation_arc,
                    "commitment_carry": dict(carry.get("commitment_carry") or {}),
                    "temperament_estimate": dict(carry.get("temperament_estimate") or {}),
                    "relationship_summary": relationship_summary,
                    "relationship_registry_summary": relationship_registry_summary,
                    "group_thread_registry_summary": group_thread_registry_summary,
                    "discussion_thread_registry_summary": discussion_thread_registry_summary,
                    "autobiographical_thread": autobiographical_thread,
                    "monument_carry": monument_summary,
                },
                "continuity": {
                    "terrain_bias": dict(continuity.get("terrain_bias") or {}),
                    "insight_bias": dict(continuity.get("insight_bias") or {}),
                    "temperament_bias": dict(continuity.get("temperament_bias") or {}),
                },
            },
            "runtime_seed": runtime_seed,
            "migration": {
                "applied": migration_applied,
                "source_schema": _text(existing_migration.get("source_schema")) or _text(package.get("schema")) or "legacy",
                "source_version": _text(existing_migration.get("source_version")) or _text(package.get("package_version")) or "legacy",
            },
        }

    def _daily_carry_summary_dict(self, nightly_payload: Mapping[str, Any]) -> dict[str, Any]:
        existing = nightly_payload.get("inner_os_daily_carry_summary")
        if isinstance(existing, Mapping):
            return dict(existing)
        if nightly_payload:
            return self._daily_summary_builder.build(nightly_payload).to_dict()
        return {}

    def _merged_state(
        self,
        current_state: Mapping[str, Any],
        last_gate_context: Mapping[str, Any],
        persona_meta_inner_os: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(current_state)
        trace = _mapping(persona_meta_inner_os.get("temperament_trace"))
        for key, value in trace.items():
            merged.setdefault(key, value)
        for key in (
            "terrain_reweighting_bias",
            "association_reweighting_bias",
            "association_reweighting_focus",
            "association_reweighting_reason",
            "insight_reframing_bias",
            "insight_class_focus",
            "insight_terrain_shape_target",
            "insight_link_counts",
            "insight_class_counts",
            "initiative_followup_bias",
            "initiative_followup_state",
            "agenda_focus",
            "agenda_bias",
            "agenda_reason",
            "agenda_window_focus",
            "agenda_window_bias",
            "agenda_window_reason",
            "identity_arc_kind",
            "identity_arc_phase",
            "identity_arc_summary",
            "identity_arc_open_tension",
            "identity_arc_stability",
            "identity_arc_registry_summary",
            "relation_arc_kind",
            "relation_arc_phase",
            "relation_arc_summary",
            "relation_arc_open_tension",
            "relation_arc_stability",
            "relation_arc_registry_summary",
            "group_relation_arc_kind",
            "group_relation_arc_phase",
            "group_relation_arc_summary",
            "group_relation_arc_boundary_mode",
            "group_relation_arc_reentry_window_focus",
            "group_relation_arc_group_thread_id",
            "group_relation_arc_topology_focus",
            "group_relation_arc_dominant_person_id",
            "group_relation_arc_stability",
            "commitment_target_focus",
            "commitment_state_focus",
            "commitment_carry_bias",
            "commitment_followup_focus",
            "commitment_mode_focus",
            "commitment_carry_reason",
            "body_homeostasis_focus",
            "body_homeostasis_carry_bias",
            "homeostasis_budget_state",
            "social_topology_state",
            "homeostasis_budget_focus",
            "homeostasis_budget_bias",
            "relational_continuity_focus",
            "relational_continuity_carry_bias",
            "group_thread_registry_snapshot",
            "discussion_thread_registry_snapshot",
            "autobiographical_thread_mode",
            "autobiographical_thread_anchor",
            "autobiographical_thread_focus",
            "autobiographical_thread_strength",
            "group_thread_focus",
            "group_thread_carry_bias",
            "expressive_style_focus",
            "expressive_style_carry_bias",
            "expressive_style_history_focus",
            "expressive_style_history_bias",
            "banter_style_focus",
            "lexical_variation_carry_bias",
            "person_registry_snapshot",
            "temperament_forward_bias",
            "temperament_guard_bias",
            "temperament_bond_bias",
            "temperament_recovery_bias",
        ):
            if key not in merged or merged.get(key) is None:
                merged[key] = self._state_value(current_state, last_gate_context, persona_meta_inner_os, key)
        return merged

    def _state_value(
        self,
        current_state: Mapping[str, Any],
        last_gate_context: Mapping[str, Any],
        persona_meta_inner_os: Mapping[str, Any],
        key: str,
        *,
        default: Any = None,
    ) -> Any:
        if key in current_state and current_state.get(key) is not None:
            return current_state.get(key)
        gate_value = _prefixed_gate_value(last_gate_context, key, None)
        if gate_value is not None:
            return gate_value
        if key in persona_meta_inner_os and persona_meta_inner_os.get(key) is not None:
            return persona_meta_inner_os.get(key)
        return default

    def _build_runtime_seed(
        self,
        current_state: Mapping[str, Any],
        last_gate_context: Mapping[str, Any],
        persona_meta_inner_os: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "prev_qualia": _sequence(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "prev_qualia", default=[])),
            "prev_qualia_habituation": _sequence(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "prev_qualia_habituation", default=[])),
            "prev_protection_grad_x": _sequence(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "prev_protection_grad_x", default=[])),
            "prev_affective_position": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "prev_affective_position", default={})),
            "affective_terrain_state": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "affective_terrain_state", default={})),
            "association_graph_state": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "association_graph_state", default={})),
            "terrain_reweighting_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "terrain_reweighting_bias")), 4),
            "association_reweighting_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "association_reweighting_bias")), 4),
            "association_reweighting_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "association_reweighting_focus")),
            "association_reweighting_reason": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "association_reweighting_reason")),
            "insight_reframing_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "insight_reframing_bias")), 4),
            "insight_class_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "insight_class_focus")),
            "insight_terrain_shape_target": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "insight_terrain_shape_target")),
            "insight_link_counts": dict(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "insight_link_counts", default={}) or {}),
            "insight_class_counts": dict(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "insight_class_counts", default={}) or {}),
            "initiative_followup_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "initiative_followup_bias")), 4),
            "initiative_followup_state": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "initiative_followup_state", default="hold")) or "hold",
            "agenda_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_focus")),
            "agenda_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_bias")), 4),
            "agenda_reason": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_reason")),
            "agenda_window_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_window_focus")),
            "agenda_window_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_window_bias")), 4),
            "agenda_window_reason": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "agenda_window_reason")),
            "learning_mode_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "learning_mode_focus")),
            "learning_mode_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "learning_mode_carry_bias")), 4),
            "social_experiment_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "social_experiment_focus")),
            "social_experiment_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "social_experiment_carry_bias")), 4),
            "temporal_membrane_focus": _text(
                self._state_value(
                    current_state,
                    last_gate_context,
                    persona_meta_inner_os,
                    "temporal_membrane_focus",
                    default=self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_membrane_mode",
                    ),
                )
            ),
            "temporal_timeline_bias": round(
                _float(
                    self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_timeline_bias",
                        default=self._state_value(
                            current_state,
                            last_gate_context,
                            persona_meta_inner_os,
                            "temporal_timeline_coherence",
                        ),
                    )
                ),
                4,
            ),
            "temporal_reentry_bias": round(
                _float(
                    self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_reentry_bias",
                        default=self._state_value(
                            current_state,
                            last_gate_context,
                            persona_meta_inner_os,
                            "temporal_reentry_pull",
                        ),
                    )
                ),
                4,
            ),
            "temporal_supersession_bias": round(
                _float(
                    self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_supersession_bias",
                        default=self._state_value(
                            current_state,
                            last_gate_context,
                            persona_meta_inner_os,
                            "temporal_supersession_pressure",
                        ),
                    )
                ),
                4,
            ),
            "temporal_continuity_bias": round(
                _float(
                    self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_continuity_bias",
                        default=self._state_value(
                            current_state,
                            last_gate_context,
                            persona_meta_inner_os,
                            "temporal_continuity_pressure",
                        ),
                    )
                ),
                4,
            ),
            "temporal_relation_reentry_bias": round(
                _float(
                    self._state_value(
                        current_state,
                        last_gate_context,
                        persona_meta_inner_os,
                        "temporal_relation_reentry_bias",
                        default=self._state_value(
                            current_state,
                            last_gate_context,
                            persona_meta_inner_os,
                            "temporal_relation_reentry_pull",
                        ),
                    )
                ),
                4,
            ),
            "identity_arc_kind": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_kind")),
            "identity_arc_phase": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_phase")),
            "identity_arc_summary": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_summary")),
            "identity_arc_open_tension": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_open_tension")),
            "identity_arc_stability": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_stability")), 4),
            "identity_arc_registry_summary": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "identity_arc_registry_summary", default={})),
            "relation_arc_kind": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_kind")),
            "relation_arc_phase": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_phase")),
            "relation_arc_summary": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_summary")),
            "relation_arc_open_tension": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_open_tension")),
            "relation_arc_stability": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_stability")), 4),
            "relation_arc_registry_summary": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relation_arc_registry_summary", default={})),
            "group_relation_arc_kind": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_kind")),
            "group_relation_arc_phase": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_phase")),
            "group_relation_arc_summary": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_summary")),
            "group_relation_arc_boundary_mode": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_boundary_mode")),
            "group_relation_arc_reentry_window_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_reentry_window_focus")),
            "group_relation_arc_group_thread_id": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_group_thread_id")),
            "group_relation_arc_topology_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_topology_focus")),
            "group_relation_arc_dominant_person_id": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_dominant_person_id")),
            "group_relation_arc_stability": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_relation_arc_stability")), 4),
            "commitment_target_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_target_focus")),
            "commitment_state_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_state_focus", default="waver")) or "waver",
            "commitment_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_carry_bias")), 4),
            "commitment_followup_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_followup_focus")),
            "commitment_mode_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_mode_focus")),
            "commitment_carry_reason": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "commitment_carry_reason")),
            "body_homeostasis_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "body_homeostasis_focus")),
            "body_homeostasis_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "body_homeostasis_carry_bias")), 4),
            "homeostasis_budget_state": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "homeostasis_budget_state", default={})),
            "social_topology_state": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "social_topology_state", default={})),
            "homeostasis_budget_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "homeostasis_budget_focus")),
            "homeostasis_budget_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "homeostasis_budget_bias")), 4),
            "relational_continuity_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relational_continuity_focus")),
            "relational_continuity_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "relational_continuity_carry_bias")), 4),
            "group_thread_registry_snapshot": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_thread_registry_snapshot", default={})),
            "discussion_thread_registry_snapshot": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "discussion_thread_registry_snapshot", default={})),
            "autobiographical_thread_mode": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "autobiographical_thread_mode")),
            "autobiographical_thread_anchor": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "autobiographical_thread_anchor")),
            "autobiographical_thread_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "autobiographical_thread_focus")),
            "autobiographical_thread_strength": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "autobiographical_thread_strength")), 4),
            "group_thread_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_thread_focus")),
            "group_thread_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "group_thread_carry_bias")), 4),
            "expressive_style_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "expressive_style_focus")),
            "expressive_style_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "expressive_style_carry_bias")), 4),
            "expressive_style_history_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "expressive_style_history_focus")),
            "expressive_style_history_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "expressive_style_history_bias")), 4),
            "banter_style_focus": _text(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "banter_style_focus")),
            "lexical_variation_carry_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "lexical_variation_carry_bias")), 4),
            "person_registry_snapshot": _mapping(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "person_registry_snapshot", default={})),
            "temperament_forward_trace": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_forward_trace")), 4),
            "temperament_guard_trace": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_guard_trace")), 4),
            "temperament_bond_trace": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_bond_trace")), 4),
            "temperament_recovery_trace": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_recovery_trace")), 4),
            "temperament_forward_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_forward_bias")), 4),
            "temperament_guard_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_guard_bias")), 4),
            "temperament_bond_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_bond_bias")), 4),
            "temperament_recovery_bias": round(_float(self._state_value(current_state, last_gate_context, persona_meta_inner_os, "temperament_recovery_bias")), 4),
        }
