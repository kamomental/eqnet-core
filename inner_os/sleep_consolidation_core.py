from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


SLEEP_CONSOLIDATION_WEIGHTS = {
    "sleep_recovery": 0.36,
    "sleep_forgetting": 0.18,
    "sleep_recent_strain": 0.16,
    "sleep_roughness_dwell": 0.14,
    "sleep_defensive_dwell": 0.12,
    "sleep_social_grounding_gap": 0.08,
    "defrag_forgetting": 0.28,
    "defrag_interference": 0.34,
    "defrag_roughness": 0.12,
    "defrag_defensive": 0.12,
    "defrag_recent_strain": 0.08,
    "replay_consolidation": 0.26,
    "replay_reuse": 0.18,
    "replay_monument": 0.16,
    "replay_prospective": 0.12,
    "replay_recent_strain": 0.12,
    "replay_identity_gap": 0.1,
    "replay_partner_relation": 0.12,
    "reconsolidation_norm": 0.18,
    "reconsolidation_role": 0.14,
    "reconsolidation_culture": 0.14,
    "reconsolidation_community": 0.14,
    "reconsolidation_ritual_memory": 0.12,
    "reconsolidation_institutional_memory": 0.12,
    "reconsolidation_reflective": 0.08,
    "reconsolidation_partner_relation": 0.18,
    "autobio_continuity_gap": 0.24,
    "autobio_belonging": 0.14,
    "autobio_trust": 0.12,
    "autobio_culture": 0.12,
    "autobio_community": 0.14,
    "autobio_recent_strain": 0.14,
    "autobio_prospective": 0.1,
    "autobio_partner_relation": 0.16,
    "abstraction_consolidation": 0.34,
    "abstraction_low_interference": 0.2,
    "abstraction_recentness": 0.16,
    "abstraction_role": 0.12,
    "abstraction_trust": 0.1,
}

SLEEP_CONSOLIDATION_THRESHOLDS = {
    "restabilize": 0.62,
    "defragment": 0.56,
    "reconsolidate": 0.54,
    "replay": 0.48,
    "abstract": 0.5,
}


@dataclass(frozen=True)
class SleepConsolidationSnapshot:
    mode: str = "settle"
    sleep_pressure: float = 0.0
    defrag_pressure: float = 0.0
    replay_priority: float = 0.0
    reconsolidation_priority: float = 0.0
    autobiographical_pull: float = 0.0
    abstraction_readiness: float = 0.0
    identity_preservation_bias: float = 0.0
    memory_class_focus: str = "episodic"
    agenda_focus: str = ""
    agenda_bias: float = 0.0
    agenda_reason: str = ""
    commitment_target_focus: str = ""
    commitment_state_focus: str = "waver"
    commitment_carry_bias: float = 0.0
    commitment_followup_focus: str = ""
    commitment_mode_focus: str = ""
    commitment_carry_reason: str = ""
    terrain_reweighting_bias: float = 0.0
    insight_class_focus: str = "none"
    insight_reframing_bias: float = 0.0
    association_reweighting_bias: float = 0.0
    association_reweighting_focus: str = ""
    association_reweighting_reason: str = ""
    insight_terrain_shape_bias: float = 0.0
    insight_terrain_shape_reason: str = ""
    insight_terrain_shape_target: str = ""
    insight_anchor_center: tuple[float, ...] = ()
    insight_anchor_dispersion: float = 0.0
    temperament_focus: str = ""
    temperament_forward_bias: float = 0.0
    temperament_guard_bias: float = 0.0
    temperament_bond_bias: float = 0.0
    temperament_recovery_bias: float = 0.0
    homeostasis_budget_focus: str = ""
    homeostasis_budget_bias: float = 0.0
    body_homeostasis_focus: str = ""
    body_homeostasis_carry_bias: float = 0.0
    relational_continuity_focus: str = ""
    relational_continuity_carry_bias: float = 0.0
    group_thread_focus: str = ""
    group_thread_carry_bias: float = 0.0
    expressive_style_focus: str = ""
    expressive_style_carry_bias: float = 0.0
    expressive_style_history_focus: str = ""
    expressive_style_history_bias: float = 0.0
    banter_style_focus: str = ""
    lexical_variation_carry_bias: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SleepConsolidationCore:
    """Small nightly-facing planner for reusable inner-life consolidation.

    This is intentionally not a full sleep engine.
    It provides a stable inner_os boundary describing what the overnight path
    should emphasize: restabilization, defragmentation, replay, reconsolidation,
    or abstraction.
    """

    def snapshot(
        self,
        *,
        current_state: Optional[Mapping[str, Any]] = None,
        forgetting_snapshot: Optional[Mapping[str, Any]] = None,
        memory_orchestration: Optional[Mapping[str, Any]] = None,
        development_state: Optional[Mapping[str, Any]] = None,
        persistence_state: Optional[Mapping[str, Any]] = None,
        personality_state: Optional[Mapping[str, Any]] = None,
    ) -> SleepConsolidationSnapshot:
        recovery_need = _float_from(current_state, "recovery_need", 0.0)
        recent_strain = _float_from(persistence_state, "recent_strain", _float_from(current_state, "recent_strain", 0.0))
        social_grounding = _float_from(persistence_state, "social_grounding", _float_from(current_state, "social_grounding", 0.0))
        continuity_score = _float_from(persistence_state, "continuity_score", _float_from(current_state, "continuity_score", 0.0))
        culture_resonance = _float_from(persistence_state, "culture_resonance", _float_from(current_state, "culture_resonance", 0.0))
        community_resonance = _float_from(persistence_state, "community_resonance", _float_from(current_state, "community_resonance", 0.0))
        roughness_dwell = _float_from(current_state, "roughness_dwell", 0.0)
        defensive_dwell = _float_from(current_state, "defensive_dwell", 0.0)
        forgetting_pressure = _float_from(forgetting_snapshot, "forgetting_pressure", 0.0)
        replay_horizon = max(1.0, _float_from(forgetting_snapshot, "replay_horizon", 2.0))

        reuse_trajectory = _float_from(memory_orchestration, "reuse_trajectory", 0.0)
        interference_pressure = _float_from(memory_orchestration, "interference_pressure", 0.0)
        consolidation_priority = _float_from(memory_orchestration, "consolidation_priority", 0.0)
        monument_salience = _float_from(memory_orchestration, "monument_salience", 0.0)
        prospective_memory_pull = _float_from(memory_orchestration, "prospective_memory_pull", 0.0)
        conscious_mosaic_recentness = _float_from(memory_orchestration, "conscious_mosaic_recentness", 0.0)

        belonging = _float_from(development_state, "belonging", _float_from(current_state, "belonging", 0.0))
        trust_bias = _float_from(development_state, "trust_bias", _float_from(current_state, "trust_bias", 0.0))
        norm_pressure = _float_from(development_state, "norm_pressure", _float_from(current_state, "norm_pressure", 0.0))
        role_commitment = _float_from(development_state, "role_commitment", _float_from(current_state, "role_commitment", 0.0))
        reflective_bias = _float_from(personality_state, "reflective_bias", _float_from(current_state, "reflective_bias", 0.0))
        caution_bias = _float_from(personality_state, "caution_bias", _float_from(current_state, "caution_bias", 0.0))

        ritual_memory = _float_from(current_state, "ritual_memory", 0.0)
        institutional_memory = _float_from(current_state, "institutional_memory", 0.0)
        memory_class_bias = _derive_memory_class_bias(current_state)
        commitment_bias = _derive_commitment_bias(current_state)
        insight_bias = _derive_insight_bias(current_state)
        related_person_id = _text_from(current_state, "related_person_id")
        attachment = _float_from(current_state, "attachment", 0.0)
        familiarity = _float_from(current_state, "familiarity", 0.0)
        trust_memory = _float_from(current_state, "trust_memory", 0.0)
        relation_seed_strength = _float_from(current_state, "relation_seed_strength", 0.0)
        partner_relation_bias = _clamp01(
            attachment * 0.3
            + familiarity * 0.24
            + trust_memory * 0.24
            + relation_seed_strength * 0.22
        ) if related_person_id else 0.0

        sleep_pressure = _clamp01(
            recovery_need * SLEEP_CONSOLIDATION_WEIGHTS["sleep_recovery"]
            + forgetting_pressure * SLEEP_CONSOLIDATION_WEIGHTS["sleep_forgetting"]
            + recent_strain * SLEEP_CONSOLIDATION_WEIGHTS["sleep_recent_strain"]
            + roughness_dwell * SLEEP_CONSOLIDATION_WEIGHTS["sleep_roughness_dwell"]
            + defensive_dwell * SLEEP_CONSOLIDATION_WEIGHTS["sleep_defensive_dwell"]
            + (1.0 - social_grounding) * SLEEP_CONSOLIDATION_WEIGHTS["sleep_social_grounding_gap"]
            + memory_class_bias["body_risk"] * 0.16
            + memory_class_bias["unresolved_tension"] * 0.08
        )
        defrag_pressure = _clamp01(
            forgetting_pressure * SLEEP_CONSOLIDATION_WEIGHTS["defrag_forgetting"]
            + interference_pressure * SLEEP_CONSOLIDATION_WEIGHTS["defrag_interference"]
            + roughness_dwell * SLEEP_CONSOLIDATION_WEIGHTS["defrag_roughness"]
            + defensive_dwell * SLEEP_CONSOLIDATION_WEIGHTS["defrag_defensive"]
            + recent_strain * SLEEP_CONSOLIDATION_WEIGHTS["defrag_recent_strain"]
            + memory_class_bias["unresolved_tension"] * 0.16
        )
        replay_priority = _clamp01(
            consolidation_priority * SLEEP_CONSOLIDATION_WEIGHTS["replay_consolidation"]
            + reuse_trajectory * SLEEP_CONSOLIDATION_WEIGHTS["replay_reuse"]
            + monument_salience * SLEEP_CONSOLIDATION_WEIGHTS["replay_monument"]
            + prospective_memory_pull * SLEEP_CONSOLIDATION_WEIGHTS["replay_prospective"]
            + recent_strain * SLEEP_CONSOLIDATION_WEIGHTS["replay_recent_strain"]
            + (1.0 - continuity_score) * SLEEP_CONSOLIDATION_WEIGHTS["replay_identity_gap"]
            + partner_relation_bias * SLEEP_CONSOLIDATION_WEIGHTS["replay_partner_relation"]
            + max(0.0, (2.0 - replay_horizon) / 2.0) * 0.08
            + memory_class_bias["bond_protection"] * 0.18
            + memory_class_bias["repair_trace"] * 0.12
            + memory_class_bias["unresolved_tension"] * 0.12
            + memory_class_bias["safe_repeat"] * 0.08
            + memory_class_bias["insight_trace"] * 0.06
        )
        reconsolidation_priority = _clamp01(
            norm_pressure * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_norm"]
            + role_commitment * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_role"]
            + culture_resonance * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_culture"]
            + community_resonance * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_community"]
            + ritual_memory * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_ritual_memory"]
            + institutional_memory * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_institutional_memory"]
            + reflective_bias * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_reflective"]
            + partner_relation_bias * SLEEP_CONSOLIDATION_WEIGHTS["reconsolidation_partner_relation"]
            + memory_class_bias["bond_protection"] * 0.18
            + memory_class_bias["repair_trace"] * 0.14
        )
        autobiographical_pull = _clamp01(
            (1.0 - continuity_score) * SLEEP_CONSOLIDATION_WEIGHTS["autobio_continuity_gap"]
            + belonging * SLEEP_CONSOLIDATION_WEIGHTS["autobio_belonging"]
            + trust_bias * SLEEP_CONSOLIDATION_WEIGHTS["autobio_trust"]
            + culture_resonance * SLEEP_CONSOLIDATION_WEIGHTS["autobio_culture"]
            + community_resonance * SLEEP_CONSOLIDATION_WEIGHTS["autobio_community"]
            + recent_strain * SLEEP_CONSOLIDATION_WEIGHTS["autobio_recent_strain"]
            + prospective_memory_pull * SLEEP_CONSOLIDATION_WEIGHTS["autobio_prospective"]
            + partner_relation_bias * SLEEP_CONSOLIDATION_WEIGHTS["autobio_partner_relation"]
            + memory_class_bias["bond_protection"] * 0.16
            + memory_class_bias["unresolved_tension"] * 0.08
        )
        abstraction_readiness = _clamp01(
            consolidation_priority * SLEEP_CONSOLIDATION_WEIGHTS["abstraction_consolidation"]
            + max(0.0, 1.0 - interference_pressure) * SLEEP_CONSOLIDATION_WEIGHTS["abstraction_low_interference"]
            + conscious_mosaic_recentness * SLEEP_CONSOLIDATION_WEIGHTS["abstraction_recentness"]
            + role_commitment * SLEEP_CONSOLIDATION_WEIGHTS["abstraction_role"]
            + trust_bias * SLEEP_CONSOLIDATION_WEIGHTS["abstraction_trust"]
            + memory_class_bias["safe_repeat"] * 0.12
            + memory_class_bias["insight_trace"] * 0.1
        )
        identity_preservation_bias = _clamp01(
            continuity_score * 0.32
            + social_grounding * 0.22
            + community_resonance * 0.18
            + culture_resonance * 0.18
            + partner_relation_bias * 0.12
            + caution_bias * 0.1
            + memory_class_bias["bond_protection"] * 0.14
            + memory_class_bias["body_risk"] * 0.12
        )
        terrain_reweighting_bias = _clamp01(
            memory_class_bias["body_risk"] * 0.28
            + memory_class_bias["bond_protection"] * 0.24
            + memory_class_bias["unresolved_tension"] * 0.18
            + memory_class_bias["repair_trace"] * 0.14
            + memory_class_bias["safe_repeat"] * 0.08
            + memory_class_bias["insight_trace"] * 0.06
        )

        if sleep_pressure >= SLEEP_CONSOLIDATION_THRESHOLDS["restabilize"]:
            mode = "restabilize"
        elif defrag_pressure >= SLEEP_CONSOLIDATION_THRESHOLDS["defragment"]:
            mode = "defragment"
        elif replay_priority >= SLEEP_CONSOLIDATION_THRESHOLDS["reconsolidate"] and reconsolidation_priority >= SLEEP_CONSOLIDATION_THRESHOLDS["reconsolidate"]:
            mode = "reconsolidate"
        elif replay_priority >= SLEEP_CONSOLIDATION_THRESHOLDS["replay"]:
            mode = "replay"
        elif abstraction_readiness >= SLEEP_CONSOLIDATION_THRESHOLDS["abstract"]:
            mode = "abstract"
        else:
            mode = "settle"

        agenda_bias = _derive_agenda_bias(current_state, mode=mode)
        temperament_bias = _derive_temperament_bias(current_state, mode=mode)
        homeostasis_budget_bias = _derive_state_carry_bias(
            current_state,
            mode=mode,
            state_field="homeostasis_budget_state",
            explicit_focus_field="homeostasis_budget_focus",
            explicit_bias_field="homeostasis_budget_bias",
            mode_scale={
                "restabilize": 0.28,
                "defragment": 0.22,
                "reconsolidate": 0.14,
                "replay": 0.1,
                "abstract": 0.08,
                "settle": 0.12,
            },
            state_boosts={
                "steady": 0.9,
                "strained": 1.02,
                "recovering": 1.08,
                "depleted": 1.14,
            },
            max_bias=0.16,
        )
        body_homeostasis_bias = _derive_state_carry_bias(
            current_state,
            mode=mode,
            state_field="body_homeostasis_state",
            explicit_focus_field="body_homeostasis_focus",
            explicit_bias_field="body_homeostasis_carry_bias",
            mode_scale={
                "restabilize": 0.42,
                "defragment": 0.36,
                "reconsolidate": 0.22,
                "replay": 0.18,
                "abstract": 0.14,
                "settle": 0.16,
            },
            state_boosts={
                "steady": 0.92,
                "strained": 1.04,
                "recovering": 1.12,
                "depleted": 1.18,
            },
            max_bias=0.18,
        )
        relational_continuity_bias = _derive_state_carry_bias(
            current_state,
            mode=mode,
            state_field="relational_continuity_state",
            explicit_focus_field="relational_continuity_focus",
            explicit_bias_field="relational_continuity_carry_bias",
            mode_scale={
                "restabilize": 0.14,
                "defragment": 0.12,
                "reconsolidate": 0.34,
                "replay": 0.28,
                "abstract": 0.14,
                "settle": 0.16,
            },
            state_boosts={
                "distant": 0.9,
                "holding_thread": 1.02,
                "reopening": 1.08,
                "co_regulating": 1.12,
            },
            max_bias=0.16,
        )
        group_thread_bias = _derive_group_thread_carry(
            current_state,
            mode=mode,
        )
        expressive_style_bias = _derive_state_carry_bias(
            current_state,
            mode=mode,
            state_field="expressive_style_state",
            explicit_focus_field="expressive_style_focus",
            explicit_bias_field="expressive_style_carry_bias",
            mode_scale={
                "restabilize": 0.1,
                "defragment": 0.12,
                "reconsolidate": 0.22,
                "replay": 0.2,
                "abstract": 0.16,
                "settle": 0.14,
            },
            state_boosts={
                "grounded_gentle": 1.04,
                "warm_companion": 1.08,
                "light_playful": 0.96,
                "quiet_repair": 1.1,
                "reverent_measured": 1.06,
            },
            max_bias=0.14,
        )
        expressive_style_history_bias = _derive_state_carry_bias(
            current_state,
            mode=mode,
            state_field="expressive_style_state",
            explicit_focus_field="expressive_style_history_focus",
            explicit_bias_field="expressive_style_history_bias",
            mode_scale={
                "restabilize": 0.08,
                "defragment": 0.1,
                "reconsolidate": 0.24,
                "replay": 0.22,
                "abstract": 0.18,
                "settle": 0.12,
            },
            state_boosts={
                "grounded_gentle": 1.02,
                "warm_companion": 1.1,
                "light_playful": 0.94,
                "quiet_repair": 1.12,
                "reverent_measured": 1.04,
            },
            max_bias=0.12,
        )
        relational_style_bias = _derive_relational_style_carry(
            current_state,
            mode=mode,
        )

        return SleepConsolidationSnapshot(
            mode=mode,
            sleep_pressure=sleep_pressure,
            defrag_pressure=defrag_pressure,
            replay_priority=replay_priority,
            reconsolidation_priority=reconsolidation_priority,
            autobiographical_pull=autobiographical_pull,
            abstraction_readiness=abstraction_readiness,
            identity_preservation_bias=identity_preservation_bias,
            memory_class_focus=memory_class_bias["focus"],
            agenda_focus=agenda_bias["focus"],
            agenda_bias=agenda_bias["carry_bias"],
            agenda_reason=agenda_bias["reason"],
            commitment_target_focus=commitment_bias["target_focus"],
            commitment_state_focus=commitment_bias["state_focus"],
            commitment_carry_bias=commitment_bias["carry_bias"],
            commitment_followup_focus=commitment_bias["followup_focus"],
            commitment_mode_focus=commitment_bias["mode_focus"],
            commitment_carry_reason=commitment_bias["carry_reason"],
            terrain_reweighting_bias=terrain_reweighting_bias,
            insight_class_focus=insight_bias["focus"],
            insight_reframing_bias=insight_bias["reframing_bias"],
            association_reweighting_bias=insight_bias["association_bias"],
            association_reweighting_focus=insight_bias["association_focus"],
            association_reweighting_reason=insight_bias["association_reason"],
            insight_terrain_shape_bias=insight_bias["terrain_shape_bias"],
            insight_terrain_shape_reason=insight_bias["terrain_shape_reason"],
            insight_terrain_shape_target=insight_bias["terrain_shape_target"],
            insight_anchor_center=insight_bias["anchor_center"],
            insight_anchor_dispersion=insight_bias["anchor_dispersion"],
            temperament_focus=temperament_bias["focus"],
            temperament_forward_bias=temperament_bias["forward_bias"],
            temperament_guard_bias=temperament_bias["guard_bias"],
            temperament_bond_bias=temperament_bias["bond_bias"],
            temperament_recovery_bias=temperament_bias["recovery_bias"],
            homeostasis_budget_focus=homeostasis_budget_bias["focus"],
            homeostasis_budget_bias=homeostasis_budget_bias["carry_bias"],
            body_homeostasis_focus=body_homeostasis_bias["focus"],
            body_homeostasis_carry_bias=body_homeostasis_bias["carry_bias"],
            relational_continuity_focus=relational_continuity_bias["focus"],
            relational_continuity_carry_bias=relational_continuity_bias["carry_bias"],
            group_thread_focus=group_thread_bias["focus"],
            group_thread_carry_bias=group_thread_bias["carry_bias"],
            expressive_style_focus=expressive_style_bias["focus"],
            expressive_style_carry_bias=expressive_style_bias["carry_bias"],
            expressive_style_history_focus=expressive_style_history_bias["focus"],
            expressive_style_history_bias=expressive_style_history_bias["carry_bias"],
            banter_style_focus=relational_style_bias["banter_style_focus"],
            lexical_variation_carry_bias=relational_style_bias["lexical_variation_carry_bias"],
            summary=_summary(mode),
        )


def _summary(mode: str) -> str:
    if mode == "restabilize":
        return "overnight priority shifts toward recovery and lowering field strain"
    if mode == "defragment":
        return "overnight priority shifts toward reducing interference and defragmenting memory load"
    if mode == "reconsolidate":
        return "overnight priority shifts toward replay plus social and autobiographical reconsolidation"
    if mode == "replay":
        return "overnight priority shifts toward selective replay and carry-over shaping"
    if mode == "abstract":
        return "overnight priority shifts toward abstraction and pattern compression"
    return "overnight state remains in a light settling phase"


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _text_from(mapping: Optional[Mapping[str, Any]], key: str) -> str:
    if not isinstance(mapping, Mapping):
        return ""
    value = mapping.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _derive_memory_class_bias(current_state: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = dict(current_state or {})
    counts = payload.get("memory_write_class_counts")
    if not isinstance(counts, Mapping):
        counts = {}
    total = 0.0
    normalized_counts: dict[str, float] = {}
    for key, value in counts.items():
        try:
            numeric = max(0.0, float(value))
        except (TypeError, ValueError):
            numeric = 0.0
        normalized_counts[str(key)] = numeric
        total += numeric
    if total > 0.0:
        normalized_counts = {key: _clamp01(value / total) for key, value in normalized_counts.items()}

    focus = _text_from(payload, "memory_write_class") or "episodic"
    bias = {
        "episodic": _clamp01(normalized_counts.get("episodic", 0.0)),
        "body_risk": _clamp01(normalized_counts.get("body_risk", 0.0)),
        "bond_protection": _clamp01(normalized_counts.get("bond_protection", 0.0)),
        "repair_trace": _clamp01(normalized_counts.get("repair_trace", 0.0)),
        "unresolved_tension": _clamp01(normalized_counts.get("unresolved_tension", 0.0)),
        "safe_repeat": _clamp01(normalized_counts.get("safe_repeat", 0.0)),
        "insight_trace": _clamp01(normalized_counts.get("insight_trace", 0.0)),
        "focus": focus,
    }
    if focus in bias and focus != "focus":
        bias[focus] = _clamp01(max(float(bias[focus]), 0.35))
    return bias


def _derive_insight_bias(current_state: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = dict(current_state or {})
    counts = payload.get("insight_class_counts")
    if not isinstance(counts, Mapping):
        counts = {}
    link_counts = payload.get("insight_link_counts")
    if not isinstance(link_counts, Mapping):
        link_counts = {}

    normalized_counts: dict[str, float] = {}
    total = 0.0
    for key, value in counts.items():
        try:
            numeric = max(0.0, float(value))
        except (TypeError, ValueError):
            numeric = 0.0
        normalized_counts[str(key)] = numeric
        total += numeric
    if total > 0.0:
        normalized_counts = {key: _clamp01(value / total) for key, value in normalized_counts.items()}

    repeated_links = 0.0
    for value in link_counts.values():
        try:
            repeated_links += max(0.0, float(value) - 1.0)
        except (TypeError, ValueError):
            continue
    repeated_link_bias = _clamp01(repeated_links / 6.0)
    focus = _text_from(payload, "insight_class_focus") or "none"
    if focus in normalized_counts:
        normalized_counts[focus] = _clamp01(max(float(normalized_counts[focus]), 0.35))

    reframing_bias = _clamp01(
        normalized_counts.get("reframed_relation", 0.0) * 0.56
        + normalized_counts.get("new_link_hypothesis", 0.0) * 0.22
        + repeated_link_bias * 0.22
    )
    association_bias = _clamp01(
        normalized_counts.get("insight_trace", 0.0) * 0.18
        + normalized_counts.get("new_link_hypothesis", 0.0) * 0.28
        + normalized_counts.get("reframed_relation", 0.0) * 0.18
        + repeated_link_bias * 0.36
    )
    explicit_shape_bias = _clamp01(_float_from(payload, "insight_terrain_shape_bias", 0.0))
    terrain_shape_reason = _text_from(payload, "insight_terrain_shape_reason")
    if explicit_shape_bias <= 0.0:
        if focus == "reframed_relation":
            explicit_shape_bias = _clamp01(reframing_bias * 0.62)
            terrain_shape_reason = terrain_shape_reason or "reframed_relation"
        elif focus == "insight_trace" and repeated_link_bias >= 0.18:
            explicit_shape_bias = _clamp01(repeated_link_bias * 0.34)
            terrain_shape_reason = terrain_shape_reason or "repeated_insight_trace"
        elif focus == "new_link_hypothesis":
            explicit_shape_bias = _clamp01(reframing_bias * 0.08)
            terrain_shape_reason = terrain_shape_reason or "new_link_hypothesis"
    if focus == "new_link_hypothesis":
        explicit_shape_bias = min(explicit_shape_bias, 0.08)
    association_focus = _text_from(payload, "association_reweighting_focus")
    association_reason = _text_from(payload, "association_reweighting_reason")
    if not association_focus:
        association_focus, association_reason = _derive_association_focus(
            focus=focus,
            normalized_counts=normalized_counts,
            repeated_link_bias=repeated_link_bias,
            association_bias=association_bias,
        )
    terrain_shape_target = _text_from(payload, "insight_terrain_shape_target")
    if not terrain_shape_target:
        terrain_shape_target = _terrain_shape_target_for_reason(terrain_shape_reason)
    anchor_center = _vector_from(payload.get("insight_anchor_center"))
    anchor_dispersion = max(0.0, _float_from(payload, "insight_anchor_dispersion", 0.0))
    return {
        "focus": focus,
        "reframing_bias": reframing_bias,
        "association_bias": association_bias,
        "association_focus": association_focus,
        "association_reason": association_reason,
        "terrain_shape_bias": explicit_shape_bias,
        "terrain_shape_reason": terrain_shape_reason,
        "terrain_shape_target": terrain_shape_target,
        "anchor_center": anchor_center,
        "anchor_dispersion": anchor_dispersion,
    }


def _derive_commitment_bias(current_state: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = dict(current_state or {})
    counts = payload.get("commitment_target_counts")
    if not isinstance(counts, Mapping):
        counts = {}

    normalized_counts: dict[str, float] = {}
    total = 0.0
    for key, value in counts.items():
        try:
            numeric = max(0.0, float(value))
        except (TypeError, ValueError):
            numeric = 0.0
        normalized_counts[str(key)] = numeric
        total += numeric
    if total > 0.0:
        normalized_counts = {key: _clamp01(value / total) for key, value in normalized_counts.items()}

    target_focus = _text_from(payload, "commitment_target_focus") or "hold"
    state_focus = _text_from(payload, "commitment_state_focus") or "waver"
    carry_reason = _text_from(payload, "commitment_carry_reason")
    followup_focus = _text_from(payload, "commitment_followup_focus")
    mode_focus = _text_from(payload, "commitment_mode_focus")
    carry_bias = _clamp01(_float_from(payload, "commitment_carry_bias", 0.0))

    if target_focus in normalized_counts:
        normalized_counts[target_focus] = _clamp01(max(float(normalized_counts[target_focus]), 0.35))
    if carry_bias <= 0.0 and target_focus:
        carry_bias = _clamp01(
            normalized_counts.get(target_focus, 0.0) * 0.42
            + (0.14 if state_focus == "commit" else 0.06 if state_focus == "settle" else 0.0)
        )
    if not followup_focus:
        followup_focus = _commitment_followup_focus_for_target(target_focus)
    if not mode_focus:
        mode_focus = _commitment_mode_focus_for_target(target_focus)
    if not carry_reason:
        carry_reason = f"{state_focus}:{target_focus}" if target_focus else ""

    return {
        "target_focus": target_focus,
        "state_focus": state_focus,
        "carry_bias": carry_bias,
        "followup_focus": followup_focus,
        "mode_focus": mode_focus,
        "carry_reason": carry_reason,
    }


def _derive_agenda_bias(
    current_state: Optional[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    payload = dict(current_state or {})
    agenda_payload = payload.get("agenda_state")
    if not isinstance(agenda_payload, Mapping):
        agenda_payload = {}
    focus = _text_from(payload, "agenda_focus") or _text_from(agenda_payload, "state")
    carry_bias = _clamp01(_float_from(payload, "agenda_bias", 0.0))
    reason = _text_from(payload, "agenda_reason") or _text_from(agenda_payload, "reason")
    agenda_score = _clamp01(_float_from(agenda_payload, "score", 0.0))
    agenda_margin = _clamp01(_float_from(agenda_payload, "winner_margin", 0.0))
    if not focus:
        return {"focus": "", "carry_bias": 0.0, "reason": ""}
    mode_scale = {
        "restabilize": 0.08,
        "defragment": 0.12,
        "reconsolidate": 0.18,
        "replay": 0.16,
        "abstract": 0.12,
        "settle": 0.1,
    }.get(str(mode or "").strip(), 0.1)
    focus_boost = {
        "hold": 1.02,
        "revisit": 1.08,
        "repair": 1.12,
        "step_forward": 0.96,
    }.get(focus, 1.0)
    derived_bias = _clamp01(
        (agenda_score * 0.28 + agenda_margin * 0.16) * mode_scale * focus_boost
    )
    carry_bias = min(0.14, max(carry_bias, derived_bias))
    if not reason:
        reason = focus
    return {
        "focus": focus,
        "carry_bias": round(_clamp01(carry_bias), 4),
        "reason": reason,
    }


def _derive_temperament_bias(
    current_state: Optional[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    payload = dict(current_state or {})
    forward_trace = _clamp01(_float_from(payload, "temperament_forward_trace", 0.0))
    guard_trace = _clamp01(_float_from(payload, "temperament_guard_trace", 0.0))
    bond_trace = _clamp01(_float_from(payload, "temperament_bond_trace", 0.0))
    recovery_trace = _clamp01(_float_from(payload, "temperament_recovery_trace", 0.0))

    mode_scale = {
        "restabilize": 0.10,
        "defragment": 0.11,
        "reconsolidate": 0.16,
        "replay": 0.15,
        "abstract": 0.13,
        "settle": 0.09,
    }.get(str(mode or "").strip(), 0.09)

    forward_bias = _clamp01(
        forward_trace
        * mode_scale
        * (0.85 if mode == "restabilize" else 1.0)
    )
    guard_bias = _clamp01(
        guard_trace
        * mode_scale
        * (1.18 if mode in {"restabilize", "defragment"} else 1.0)
    )
    bond_bias = _clamp01(
        bond_trace
        * mode_scale
        * (1.14 if mode in {"replay", "reconsolidate"} else 0.96)
    )
    recovery_bias = _clamp01(
        recovery_trace
        * mode_scale
        * (1.22 if mode == "restabilize" else 1.04 if mode == "reconsolidate" else 1.0)
    )

    focus_scores = {
        "forward": forward_bias,
        "guard": guard_bias,
        "bond": bond_bias,
        "recovery": recovery_bias,
    }
    focus = ""
    if any(score > 0.0 for score in focus_scores.values()):
        focus = max(
            focus_scores.keys(),
            key=lambda key: (focus_scores[key], key),
        )

    return {
        "focus": focus,
        "forward_bias": round(float(forward_bias), 4),
        "guard_bias": round(float(guard_bias), 4),
        "bond_bias": round(float(bond_bias), 4),
        "recovery_bias": round(float(recovery_bias), 4),
    }


def _derive_state_carry_bias(
    current_state: Optional[Mapping[str, Any]],
    *,
    mode: str,
    state_field: str,
    explicit_focus_field: str,
    explicit_bias_field: str,
    mode_scale: Mapping[str, float],
    state_boosts: Mapping[str, float],
    max_bias: float,
) -> dict[str, Any]:
    payload = dict(current_state or {})
    focus = _text_from(payload, explicit_focus_field)
    explicit_bias = _clamp01(_float_from(payload, explicit_bias_field, 0.0))
    state_payload = payload.get(state_field)
    state_focus = ""
    state_score = 0.0
    state_margin = 0.0
    if isinstance(state_payload, Mapping):
        state_focus = _text_from(state_payload, "state")
        state_score = _clamp01(_float_from(state_payload, "score", 0.0))
        state_margin = _clamp01(_float_from(state_payload, "winner_margin", 0.0))
    focus = focus or state_focus
    if not focus:
        return {"focus": "", "carry_bias": 0.0}

    derived_bias = _clamp01(
        (state_score * 0.32 + state_margin * 0.14)
        * float(mode_scale.get(str(mode or "").strip(), 0.14) or 0.14)
        * float(state_boosts.get(focus, 1.0) or 1.0)
    )
    carry_bias = min(float(max_bias), max(explicit_bias, derived_bias))
    return {
        "focus": focus,
        "carry_bias": round(_clamp01(carry_bias), 4),
    }


def _derive_relational_style_carry(
    current_state: Optional[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    payload = dict(current_state or {})
    relation_style = payload.get("relational_style_memory_state")
    if not isinstance(relation_style, Mapping):
        relation_style = {}
    banter_style_focus = _text_from(payload, "banter_style_focus") or _text_from(relation_style, "banter_style")
    lexical_variation_carry_bias = _clamp01(
        _float_from(payload, "lexical_variation_carry_bias", 0.0)
    )
    relation_score = _clamp01(_float_from(relation_style, "score", 0.0))
    relation_margin = _clamp01(_float_from(relation_style, "winner_margin", 0.0))
    lexical_variation_bias = _clamp01(_float_from(relation_style, "lexical_variation_bias", 0.0))
    mode_scale = {
        "restabilize": 0.08,
        "defragment": 0.1,
        "reconsolidate": 0.18,
        "replay": 0.16,
        "abstract": 0.12,
        "settle": 0.1,
    }.get(str(mode or "").strip(), 0.1)
    style_boost = 1.0
    if banter_style_focus in {"gentle_tease", "compact_wit"}:
        style_boost = 1.08
    elif banter_style_focus in {"respectful_light", "soft_formal"}:
        style_boost = 0.96
    derived_bias = _clamp01(
        (lexical_variation_bias * 0.28 + relation_score * 0.12 + relation_margin * 0.08)
        * mode_scale
        * style_boost
    )
    lexical_variation_carry_bias = min(
        0.12,
        max(lexical_variation_carry_bias, derived_bias),
    )
    return {
        "banter_style_focus": banter_style_focus,
        "lexical_variation_carry_bias": round(_clamp01(lexical_variation_carry_bias), 4),
    }


def _derive_group_thread_carry(
    current_state: Optional[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    payload = dict(current_state or {})
    focus = _text_from(payload, "group_thread_focus")
    carry_bias = _clamp01(_float_from(payload, "group_thread_carry_bias", 0.0))
    registry = payload.get("group_thread_registry_snapshot")
    dominant_thread = ""
    total_threads = 0.0
    if isinstance(registry, Mapping):
        dominant_thread = _text_from(registry, "dominant_thread_id")
        total_threads = float(registry.get("total_threads") or 0.0)
    topology_state = payload.get("social_topology_state")
    topology_name = _text_from(topology_state if isinstance(topology_state, Mapping) else {}, "state")
    focus = focus or topology_name
    if not focus:
        return {"focus": "", "carry_bias": 0.0}
    mode_scale = {
        "restabilize": 0.08,
        "defragment": 0.12,
        "reconsolidate": 0.22,
        "replay": 0.18,
        "abstract": 0.12,
        "settle": 0.14,
    }.get(str(mode or "").strip(), 0.12)
    focus_boost = {
        "threaded_group": 1.12,
        "public_visible": 1.04,
        "hierarchical": 1.02,
        "one_to_one": 0.94,
        "ambient": 0.88,
    }.get(focus, 1.0)
    derived_bias = _clamp01(
        (min(total_threads, 3.0) / 3.0 * 0.18 + (0.12 if dominant_thread else 0.0))
        * mode_scale
        * focus_boost
    )
    return {
        "focus": focus,
        "carry_bias": round(min(0.16, max(carry_bias, derived_bias)), 4),
    }


def _vector_from(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _derive_association_focus(
    *,
    focus: str,
    normalized_counts: Mapping[str, float],
    repeated_link_bias: float,
    association_bias: float,
) -> tuple[str, str]:
    if association_bias <= 0.0:
        return "", ""
    if repeated_link_bias >= 0.12 or float(normalized_counts.get("insight_trace", 0.0)) >= 0.28:
        return "repeated_links", "repeated_insight_trace"
    if float(normalized_counts.get("new_link_hypothesis", 0.0)) > 0.0:
        return "hypothesis_links", "new_link_hypothesis"
    if float(normalized_counts.get("reframed_relation", 0.0)) > 0.0:
        return "reframed_links", "reframed_relation"
    if focus == "insight_trace":
        return "repeated_links", "repeated_insight_trace"
    if focus == "new_link_hypothesis":
        return "hypothesis_links", "new_link_hypothesis"
    if focus == "reframed_relation":
        return "reframed_links", "reframed_relation"
    return "", ""


def _terrain_shape_target_for_reason(reason: str) -> str:
    normalized_reason = str(reason or "").strip()
    if normalized_reason == "reframed_relation":
        return "soft_relation"
    if normalized_reason == "repeated_insight_trace":
        return "trace_basin"
    if normalized_reason == "new_link_hypothesis":
        return "hypothesis_hold"
    return ""


def _commitment_followup_focus_for_target(target: str) -> str:
    normalized_target = str(target or "").strip()
    if normalized_target == "step_forward":
        return "offer_next_step"
    if normalized_target in {"repair", "bond_protect"}:
        return "reopen_softly"
    if normalized_target in {"hold", "stabilize"}:
        return "hold"
    return ""


def _commitment_mode_focus_for_target(target: str) -> str:
    normalized_target = str(target or "").strip()
    if normalized_target == "step_forward":
        return "monitor"
    if normalized_target in {"repair", "bond_protect"}:
        return "repair"
    if normalized_target == "stabilize":
        return "stabilize"
    if normalized_target == "hold":
        return "contain"
    return ""
