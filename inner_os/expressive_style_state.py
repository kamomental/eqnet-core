from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExpressiveStyleState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    lightness_room: float
    continuity_weight: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "lightness_room": round(self.lightness_room, 4),
            "continuity_weight": round(self.continuity_weight, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_expressive_style_state(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    relational_continuity_state: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    attention_regulation_state: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    contact_readiness: float,
    coherence_score: float,
    human_presence_signal: float,
) -> ExpressiveStyleState:
    state = dict(self_state or {})
    temperament = dict(temperament_estimate or {})
    recovery = dict(body_recovery_guard or {})
    homeostasis = dict(body_homeostasis_state or {})
    homeostasis_budget = dict(homeostasis_budget_state or {})
    readiness = dict(initiative_readiness or {})
    commitment = dict(commitment_state or {})
    relation = dict(relational_continuity_state or {})
    relation_style = dict(relational_style_memory_state or {})
    culture = dict(cultural_conversation_state or {})
    topology = dict(social_topology_state or {})
    attention = dict(attention_regulation_state or {})
    grice = dict(grice_guard_state or {})
    protection = dict(protection_mode or {})

    stress = _float01(state.get("stress"))
    recovery_need = _float01(state.get("recovery_need"))
    recent_strain = _float01(state.get("recent_strain"))
    continuity_score = _float01(state.get("continuity_score"))
    social_grounding = _float01(state.get("social_grounding"))
    relation_seed_strength = _float01(state.get("relation_seed_strength"))
    long_term_theme_strength = _float01(state.get("long_term_theme_strength"))
    conscious_residue_strength = _float01(state.get("conscious_residue_strength"))

    bond_drive = _float01(temperament.get("bond_drive"))
    curiosity_drive = _float01(temperament.get("curiosity_drive"))
    recovery_discipline = _float01(temperament.get("recovery_discipline"))
    protect_floor = _float01(temperament.get("protect_floor"))
    initiative_persistence = _float01(temperament.get("initiative_persistence"))
    leader_tendency = _float01(temperament.get("leader_tendency"))
    hero_tendency = _float01(temperament.get("hero_tendency"))
    forward_trace = _float01(temperament.get("forward_trace"))
    bond_trace = _float01(temperament.get("bond_trace"))
    carry_focus = str(state.get("expressive_style_focus") or "").strip()
    carry_bias = _float01(state.get("expressive_style_carry_bias"))
    history_focus = str(state.get("expressive_style_history_focus") or "").strip()
    history_bias = _float01(state.get("expressive_style_history_bias"))
    banter_style_focus = str(state.get("banter_style_focus") or "").strip()
    lexical_variation_carry_bias = _float01(state.get("lexical_variation_carry_bias"))
    relation_style_name = str(relation_style.get("state") or "grounded_gentle").strip()
    relation_style_score = _float01(relation_style.get("score"))
    relational_warmth_bias = _float01(relation_style.get("warmth_bias"))
    playful_ceiling = _float01(relation_style.get("playful_ceiling"))
    advice_tolerance = _float01(relation_style.get("advice_tolerance"))
    lexical_familiarity = _float01(relation_style.get("lexical_familiarity"))
    lexical_variation_bias = _float01(relation_style.get("lexical_variation_bias"))
    relational_banter_room = _float01(relation_style.get("banter_room"))
    banter_style = str(relation_style.get("banter_style") or "grounded_companion").strip() or "grounded_companion"
    cultural_state_name = str(culture.get("state") or "careful_polite").strip()
    cultural_score = _float01(culture.get("score"))
    cultural_directness_ceiling = _float01(culture.get("directness_ceiling"))
    cultural_joke_ratio_ceiling = _float01(culture.get("joke_ratio_ceiling"))
    cultural_politeness_pressure = _float01(culture.get("politeness_pressure"))
    cultural_group_attunement = _float01(culture.get("group_attunement"))

    recovery_state = str(recovery.get("state") or "open").strip()
    recovery_score = _float01(recovery.get("score"))
    homeostasis_state = str(homeostasis.get("state") or "steady").strip()
    homeostasis_score = _float01(homeostasis.get("score"))
    budget_state = str(homeostasis_budget.get("state") or "steady").strip()
    budget_score = _float01(homeostasis_budget.get("score"))
    readiness_state = str(readiness.get("state") or "hold").strip()
    readiness_score = _float01(readiness.get("score"))
    commitment_target = str(commitment.get("target") or "").strip()
    commitment_score = _float01(commitment.get("score"))
    relation_state = str(relation.get("state") or "distant").strip()
    relation_score = _float01(relation.get("score"))
    topology_state = str(topology.get("state") or "ambient").strip()
    topology_score = _float01(topology.get("score"))
    attention_state = str(attention.get("state") or "selective_hold").strip()
    grice_state = str(grice.get("state") or "advise_openly").strip()
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_strength = _float01(protection.get("strength"))

    formal_pressure = 0.0
    if topology_state == "public_visible":
        formal_pressure = max(formal_pressure, max(topology_score, _float01(topology.get("visibility_pressure"))))
    elif topology_state == "hierarchical":
        formal_pressure = max(formal_pressure, max(topology_score, _float01(topology.get("hierarchy_pressure"))))
    elif topology_state == "threaded_group":
        formal_pressure = max(formal_pressure, _float01(topology.get("threading_pressure")) * 0.58)
    formal_pressure = _clamp01(formal_pressure + cultural_politeness_pressure * 0.28)
    banter_style_bonus = 0.0
    if banter_style in {"gentle_tease", "compact_wit"}:
        banter_style_bonus = 0.08
    elif banter_style == "warm_refrain":
        banter_style_bonus = 0.05
    elif banter_style in {"respectful_light", "soft_formal"}:
        banter_style_bonus = -0.04

    guard_pressure = _clamp01(
        (0.58 if recovery_state == "recovery_first" else 0.0)
        + (0.34 if recovery_state == "guarded" else 0.0)
        + (0.22 if homeostasis_state == "depleted" else 0.0)
        + (0.16 if homeostasis_state == "recovering" else 0.0)
        + (0.18 if budget_state == "depleted" else 0.0)
        + 0.16 * recovery_score
        + 0.12 * max(homeostasis_score, budget_score)
    )
    protective_pressure = _clamp01(
        protection_strength
        * (
            1.0
            if protection_mode_name in {"contain", "stabilize", "shield"}
            else 0.58 if protection_mode_name == "repair" else 0.32
        )
    )
    grice_suppression = 1.0 if grice_state == "hold_obvious_advice" else 0.56 if grice_state == "attune_without_repeating" else 0.24 if grice_state == "acknowledge_then_extend" else 0.0
    attention_guard = 1.0 if attention_state == "reflex_guard" else 0.54 if attention_state == "split_guarded" else 0.0
    relation_open = 1.0 * relation_score if relation_state == "co_regulating" else 0.82 * relation_score if relation_state == "reopening" else 0.5 * relation_score if relation_state == "holding_thread" else 0.0

    continuity_weight = _clamp01(
        0.24 * continuity_score
        + 0.18 * social_grounding
        + 0.12 * relation_seed_strength
        + 0.1 * long_term_theme_strength
        + 0.08 * conscious_residue_strength
        + 0.1 * bond_trace
        + 0.08 * forward_trace
        + 0.1 * relation_score
    )
    lightness_room = _clamp01(
        0.22 * (1.0 - stress)
        + 0.18 * (1.0 - recovery_need)
        + 0.14 * _clamp01(contact_readiness)
        + 0.1 * _clamp01(coherence_score)
        + 0.08 * _clamp01(human_presence_signal)
        + 0.14 * curiosity_drive
        + 0.08 * initiative_persistence
        + 0.08 * relation_open
        + 0.1 * playful_ceiling
        + 0.08 * lexical_familiarity
        + 0.08 * lexical_variation_bias
        + 0.06 * relational_banter_room
        + banter_style_bonus
        + 0.1 * cultural_joke_ratio_ceiling
        + 0.08 * cultural_directness_ceiling
        + 0.06 * cultural_group_attunement
        - 0.18 * guard_pressure
        - 0.16 * protective_pressure
        - 0.16 * formal_pressure
        - 0.1 * recent_strain
        - 0.1 * grice_suppression
        - 0.08 * attention_guard
    )

    repair_target = 1.0 if commitment_target in {"repair", "bond_protect"} else 0.72 if commitment_target == "stabilize" else 0.0
    forward_target = 1.0 if commitment_target == "step_forward" else 0.48 if readiness_state == "ready" else 0.24 if readiness_state == "tentative" else 0.0

    scores = {
        "grounded_gentle": _clamp01(
            0.18
            + 0.24 * continuity_weight
            + 0.12 * (1.0 - formal_pressure)
            + 0.12 * _clamp01(human_presence_signal)
            + 0.08 * recovery_discipline
            + 0.06 * bond_drive
            + 0.08 * advice_tolerance
            + 0.06 * lexical_familiarity
            + 0.06 * (1.0 - max(grice_suppression, attention_guard) * 0.5)
        ),
        "warm_companion": _clamp01(
            0.12
            + 0.24 * relation_open
            + 0.18 * bond_drive
            + 0.16 * continuity_weight
            + 0.1 * leader_tendency
            + 0.14 * relational_warmth_bias
            + 0.08 * lexical_familiarity
            + 0.06 * lexical_variation_bias
            + 0.08 * cultural_group_attunement
            + 0.08 * _clamp01(human_presence_signal)
            - 0.1 * formal_pressure
            - 0.08 * guard_pressure
        ),
        "light_playful": _clamp01(
            0.08
            + 0.3 * lightness_room
            + 0.18 * curiosity_drive
            + 0.12 * initiative_persistence
            + 0.08 * hero_tendency
            + 0.08 * continuity_weight
            + 0.06 * forward_target
            + 0.16 * playful_ceiling
            + 0.12 * relational_banter_room
            + 0.06 * lexical_familiarity
            + 0.08 * lexical_variation_bias
            + 0.14 * cultural_joke_ratio_ceiling
            - 0.16 * guard_pressure
            - 0.14 * formal_pressure
            - 0.12 * grice_suppression
        ),
        "quiet_repair": _clamp01(
            0.1
            + 0.2 * recent_strain
            + 0.18 * repair_target
            + 0.14 * continuity_weight
            + 0.12 * leader_tendency
            + 0.1 * max(relation_open, 0.2 * relation_score)
            + 0.08 * recovery_discipline
            + 0.08 * relational_warmth_bias
            + 0.08 * cultural_group_attunement
            + 0.08 * guard_pressure
            - 0.06 * formal_pressure
        ),
        "reverent_measured": _clamp01(
            0.08
            + 0.26 * formal_pressure
            + 0.18 * guard_pressure
            + 0.14 * grice_suppression
            + 0.1 * attention_guard
            + 0.08 * recovery_discipline
            + 0.08 * protect_floor
            + 0.14 * cultural_politeness_pressure
            + 0.08 * (1.0 - playful_ceiling)
        ),
    }
    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.12
        if carry_focus == "warm_companion":
            carry_scale = 0.15
        elif carry_focus == "quiet_repair":
            carry_scale = 0.16
        elif carry_focus == "grounded_gentle":
            carry_scale = 0.14
        elif carry_focus == "reverent_measured":
            carry_scale = 0.14
        if carry_focus == "light_playful" and lightness_room < 0.28:
            carry_scale *= 0.6
        if carry_focus == "warm_companion" and formal_pressure >= 0.34:
            carry_scale *= 0.75
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)
    if history_focus in scores and history_bias > 0.0:
        history_scale = 0.1 if history_focus != "light_playful" else 0.08
        if history_focus == "light_playful" and (lightness_room < 0.3 or formal_pressure >= 0.3):
            history_scale *= 0.55
        scores[history_focus] = _clamp01(scores[history_focus] + history_bias * history_scale)
    if lexical_variation_carry_bias > 0.0:
        if banter_style_focus in {"gentle_tease", "compact_wit"}:
            scores["light_playful"] = _clamp01(
                scores["light_playful"] + lexical_variation_carry_bias * 0.1 * (1.0 - formal_pressure * 0.45)
            )
        elif banter_style_focus == "warm_refrain":
            scores["warm_companion"] = _clamp01(scores["warm_companion"] + lexical_variation_carry_bias * 0.1)
        elif banter_style_focus in {"respectful_light", "soft_formal"}:
            scores["reverent_measured"] = _clamp01(
                scores["reverent_measured"] + lexical_variation_carry_bias * 0.08
            )

    style, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "continuity_thread" if continuity_weight >= 0.48 else "",
            "safe_lightness_room" if lightness_room >= 0.5 else "",
            "bond_trace" if bond_trace >= 0.38 or bond_drive >= 0.52 else "",
            "forward_trace" if forward_trace >= 0.34 or initiative_persistence >= 0.52 else "",
            "relational_style_warmth" if relational_warmth_bias >= 0.36 and style in {"warm_companion", "quiet_repair"} else "",
            "relational_playful_ceiling" if playful_ceiling >= 0.34 and style == "light_playful" else "",
            "relational_lexical_familiarity" if lexical_familiarity >= 0.32 else "",
            "relational_lexical_variation" if lexical_variation_bias >= 0.34 else "",
            "relational_style_memory" if relation_style_name and relation_style_score >= 0.24 else "",
            f"banter_style_{banter_style}" if banter_style and banter_style != "grounded_companion" else "",
            "cultural_joke_ceiling" if cultural_joke_ratio_ceiling >= 0.34 and style == "light_playful" else "",
            "cultural_group_attunement" if cultural_group_attunement >= 0.28 and style in {"warm_companion", "quiet_repair"} else "",
            "cultural_politeness" if cultural_politeness_pressure >= 0.28 and style == "reverent_measured" else "",
            "cultural_conversation_state" if cultural_state_name and cultural_score >= 0.24 else "",
            "repair_commitment" if repair_target >= 0.72 and commitment_score >= 0.28 else "",
            "reverent_topology" if formal_pressure >= 0.34 else "",
            "recovery_guard" if guard_pressure >= 0.34 else "",
            "grice_suppression" if grice_suppression >= 0.48 else "",
            "attention_guard" if attention_guard >= 0.48 else "",
            "leader_tendency" if leader_tendency >= 0.46 else "",
            "hero_tendency" if hero_tendency >= 0.46 else "",
            "overnight_expressive_style" if carry_focus in scores and carry_bias >= 0.08 else "",
            "overnight_expressive_style_history" if history_focus in scores and history_bias >= 0.08 else "",
            f"overnight_banter_style_{banter_style_focus}" if banter_style_focus and lexical_variation_carry_bias >= 0.08 else "",
        ]
    )

    return ExpressiveStyleState(
        state=style,
        score=_clamp01(scores.get(style, 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        lightness_room=lightness_room,
        continuity_weight=continuity_weight,
        dominant_inputs=dominant_inputs,
    )


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "grounded_gentle", 0.0
    winner_name, winner_score = items[0]
    runner_up = items[1][1] if len(items) > 1 else 0.0
    return winner_name, round(_clamp01(winner_score - runner_up), 4)


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
