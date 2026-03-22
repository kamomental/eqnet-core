from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RelationalStyleMemoryState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    warmth_bias: float
    playful_ceiling: float
    advice_tolerance: float
    lexical_familiarity: float
    lexical_variation_bias: float
    banter_room: float
    banter_style: str
    dominant_person_id: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "warmth_bias": round(self.warmth_bias, 4),
            "playful_ceiling": round(self.playful_ceiling, 4),
            "advice_tolerance": round(self.advice_tolerance, 4),
            "lexical_familiarity": round(self.lexical_familiarity, 4),
            "lexical_variation_bias": round(self.lexical_variation_bias, 4),
            "banter_room": round(self.banter_room, 4),
            "banter_style": self.banter_style,
            "dominant_person_id": self.dominant_person_id,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_relational_style_memory_state(
    *,
    self_state: Mapping[str, Any],
    relation_bias_strength: float,
    related_person_ids: Sequence[str],
    relation_competition_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
) -> RelationalStyleMemoryState:
    state = dict(self_state or {})
    competition = dict(relation_competition_state or {})
    topology = dict(social_topology_state or {})
    recovery = dict(body_recovery_guard or {})
    homeostasis = dict(body_homeostasis_state or {})
    registry_snapshot = dict(state.get("person_registry_snapshot") or {})

    dominant_person_id = (
        str(competition.get("dominant_person_id") or "").strip()
        or str(state.get("related_person_id") or "").strip()
    )
    if not dominant_person_id:
        for item in related_person_ids or ():
            text = str(item).strip()
            if text:
                dominant_person_id = text
                break

    adaptive_traits = _person_style_traits(registry_snapshot, dominant_person_id)

    attachment = _float01(state.get("attachment"))
    familiarity = _float01(state.get("familiarity"))
    trust_memory = _float01(state.get("trust_memory"))
    continuity_score = _float01(state.get("continuity_score"))
    recent_strain = _float01(state.get("recent_strain"))
    exploration_bias = _float01(state.get("exploration_bias"))
    caution_bias = _float01(state.get("caution_bias"))
    social_grounding = _float01(state.get("social_grounding"))

    partner_address_hint = str(state.get("partner_address_hint") or "").strip()
    partner_timing_hint = str(state.get("partner_timing_hint") or "").strip()
    partner_stance_hint = str(state.get("partner_stance_hint") or "").strip()
    partner_social_interpretation = str(state.get("partner_social_interpretation") or "").strip()

    competition_level = _float01(competition.get("competition_level"))
    total_people = int(competition.get("total_people") or 0)
    topology_state = str(topology.get("state") or "ambient").strip()
    topology_visibility = _float01(topology.get("visibility_pressure"))
    topology_threading = _float01(topology.get("threading_pressure"))
    topology_hierarchy = _float01(topology.get("hierarchy_pressure"))
    guard_state = str(recovery.get("state") or "open").strip()
    guard_score = _float01(recovery.get("score"))
    body_state = str(homeostasis.get("state") or "steady").strip()
    body_score = _float01(homeostasis.get("score"))

    familiar_open = 0.0
    careful_distance = 0.0
    if partner_stance_hint == "familiar":
        familiar_open += 0.18
    elif partner_stance_hint == "respectful":
        careful_distance += 0.16
    if partner_timing_hint == "open":
        familiar_open += 0.14
    elif partner_timing_hint == "delayed":
        careful_distance += 0.18
    if partner_address_hint == "companion":
        familiar_open += 0.12
    elif partner_address_hint in {"senpai", "respectful"}:
        careful_distance += 0.12
    if "familiar" in partner_social_interpretation or "future_open" in partner_social_interpretation:
        familiar_open += 0.1
    if "delayed" in partner_social_interpretation or "respectful" in partner_social_interpretation:
        careful_distance += 0.1

    warmth_bias = _clamp01(
        max(_float01(adaptive_traits.get("style_warmth_memory")), 0.22 + attachment * 0.18 + trust_memory * 0.14)
        + relation_bias_strength * 0.12
        + familiar_open * 0.18
        - careful_distance * 0.08
    )
    playful_ceiling = _clamp01(
        max(_float01(adaptive_traits.get("playful_ceiling")), 0.16 + familiarity * 0.16 + exploration_bias * 0.16)
        + familiar_open * 0.14
        - careful_distance * 0.12
        - competition_level * 0.08
        - topology_visibility * 0.08
        - topology_hierarchy * 0.1
    )
    advice_tolerance = _clamp01(
        max(_float01(adaptive_traits.get("advice_tolerance")), 0.18 + trust_memory * 0.16 + continuity_score * 0.14)
        + familiar_open * 0.08
        - careful_distance * 0.1
        - recent_strain * 0.08
    )
    lexical_familiarity = _clamp01(
        max(_float01(adaptive_traits.get("lexical_familiarity")), 0.16 + familiarity * 0.18 + continuity_score * 0.14)
        + social_grounding * 0.08
        + familiar_open * 0.08
        - careful_distance * 0.06
    )

    guard_pressure = _clamp01(
        (0.22 if guard_state == "recovery_first" else 0.0)
        + (0.12 if guard_state == "guarded" else 0.0)
        + (0.16 if body_state == "depleted" else 0.0)
        + (0.1 if body_state == "recovering" else 0.0)
        + guard_score * 0.2
        + body_score * 0.14
    )
    social_pressure = _clamp01(
        competition_level * 0.18
        + (0.08 if total_people > 2 else 0.0)
        + topology_visibility * 0.22
        + topology_hierarchy * 0.24
        + topology_threading * 0.1
    )
    banter_room = _clamp01(
        playful_ceiling * 0.52
        + lexical_familiarity * 0.18
        + warmth_bias * 0.1
        + relation_bias_strength * 0.08
        - guard_pressure * 0.28
        - social_pressure * 0.22
        - caution_bias * 0.14
    )
    lexical_variation_bias = _clamp01(
        max(_float01(adaptive_traits.get("lexical_variation_bias")), lexical_familiarity * 0.46 + familiarity * 0.08)
        + banter_room * 0.16
        + playful_ceiling * 0.1
        + familiar_open * 0.08
        - guard_pressure * 0.14
        - social_pressure * 0.18
        - careful_distance * 0.12
    )
    banter_style = _derive_banter_style(
        warmth_bias=warmth_bias,
        playful_ceiling=playful_ceiling,
        advice_tolerance=advice_tolerance,
        lexical_familiarity=lexical_familiarity,
        lexical_variation_bias=lexical_variation_bias,
        banter_room=banter_room,
        careful_distance=careful_distance,
        guard_pressure=guard_pressure,
        social_pressure=social_pressure,
    )

    scores = {
        "grounded_gentle": _clamp01(
            0.18
            + advice_tolerance * 0.16
            + lexical_familiarity * 0.1
            + warmth_bias * 0.1
            - social_pressure * 0.06
        ),
        "warm_companion": _clamp01(
            0.12
            + warmth_bias * 0.34
            + lexical_familiarity * 0.16
            + relation_bias_strength * 0.18
            + familiar_open * 0.14
            - social_pressure * 0.08
        ),
        "light_playful": _clamp01(
            0.08
            + playful_ceiling * 0.36
            + banter_room * 0.28
            + lexical_familiarity * 0.08
            + familiar_open * 0.08
            - careful_distance * 0.14
            - guard_pressure * 0.16
        ),
        "quiet_repair": _clamp01(
            0.1
            + warmth_bias * 0.16
            + advice_tolerance * 0.1
            + max(recent_strain, guard_pressure) * 0.18
            + (0.08 if "repair" in partner_social_interpretation else 0.0)
            + careful_distance * 0.1
        ),
        "reverent_measured": _clamp01(
            0.08
            + careful_distance * 0.28
            + social_pressure * 0.18
            + (1.0 - playful_ceiling) * 0.12
            + (1.0 - advice_tolerance) * 0.08
        ),
    }
    style, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "person_registry_style_memory" if dominant_person_id and adaptive_traits else "",
            "partner_familiar_open" if familiar_open >= 0.18 else "",
            "partner_careful_distance" if careful_distance >= 0.18 else "",
            "relation_bias" if relation_bias_strength >= 0.28 else "",
            "lexical_variation_room" if lexical_variation_bias >= 0.34 else "",
            f"banter_style_{banter_style}" if banter_style and banter_style != "grounded_companion" else "",
            "competition_threads" if competition_level >= 0.22 else "",
            "public_or_hierarchy" if topology_state in {"public_visible", "hierarchical"} else "",
            "body_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "body_homeostasis" if body_state in {"recovering", "depleted"} else "",
        ]
    )

    return RelationalStyleMemoryState(
        state=style,
        score=_clamp01(scores.get(style, 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        warmth_bias=warmth_bias,
        playful_ceiling=playful_ceiling,
        advice_tolerance=advice_tolerance,
        lexical_familiarity=lexical_familiarity,
        lexical_variation_bias=lexical_variation_bias,
        banter_room=banter_room,
        banter_style=banter_style,
        dominant_person_id=dominant_person_id,
        dominant_inputs=dominant_inputs,
    )


def _person_style_traits(
    registry_snapshot: Mapping[str, Any],
    person_id: str,
) -> dict[str, Any]:
    if not person_id:
        return {}
    persons = registry_snapshot.get("persons") if isinstance(registry_snapshot, Mapping) else None
    if not isinstance(persons, Mapping):
        return {}
    payload = persons.get(person_id)
    if not isinstance(payload, Mapping):
        return {}
    adaptive_traits = payload.get("adaptive_traits")
    if not isinstance(adaptive_traits, Mapping):
        return {}
    return dict(adaptive_traits)


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


def _derive_banter_style(
    *,
    warmth_bias: float,
    playful_ceiling: float,
    advice_tolerance: float,
    lexical_familiarity: float,
    lexical_variation_bias: float,
    banter_room: float,
    careful_distance: float,
    guard_pressure: float,
    social_pressure: float,
) -> str:
    if guard_pressure >= 0.38 or social_pressure >= 0.46:
        return "respectful_light"
    if banter_room >= 0.48 and playful_ceiling >= 0.46 and lexical_variation_bias >= 0.42:
        if warmth_bias >= 0.48:
            return "gentle_tease"
        return "compact_wit"
    if warmth_bias >= 0.52 and advice_tolerance >= 0.46:
        return "warm_refrain"
    if careful_distance >= 0.24 or social_pressure >= 0.28:
        return "soft_formal"
    if lexical_familiarity >= 0.34 or lexical_variation_bias >= 0.3:
        return "compact_wit"
    return "grounded_companion"


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
