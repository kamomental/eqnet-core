from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LightnessBudgetState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    banter_room: float
    banter_style: str
    playful_ceiling: float
    lexical_variation_bias: float
    suppression: float
    advice_tolerance: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "banter_room": round(self.banter_room, 4),
            "banter_style": self.banter_style,
            "playful_ceiling": round(self.playful_ceiling, 4),
            "lexical_variation_bias": round(self.lexical_variation_bias, 4),
            "suppression": round(self.suppression, 4),
            "advice_tolerance": round(self.advice_tolerance, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_lightness_budget_state(
    *,
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    attention_regulation_state: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    relation_competition_state: Mapping[str, Any],
    expressive_style_state: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
) -> LightnessBudgetState:
    recovery = dict(body_recovery_guard or {})
    body = dict(body_homeostasis_state or {})
    budget = dict(homeostasis_budget_state or {})
    protection = dict(protection_mode or {})
    attention = dict(attention_regulation_state or {})
    grice = dict(grice_guard_state or {})
    topology = dict(social_topology_state or {})
    competition = dict(relation_competition_state or {})
    expressive = dict(expressive_style_state or {})
    relation_style = dict(relational_style_memory_state or {})
    culture = dict(cultural_conversation_state or {})

    recovery_state = str(recovery.get("state") or "open").strip()
    recovery_score = _float01(recovery.get("score"))
    body_state = str(body.get("state") or "steady").strip()
    body_score = _float01(body.get("score"))
    budget_state = str(budget.get("state") or "steady").strip()
    budget_score = _float01(budget.get("score"))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_strength = _float01(protection.get("strength"))
    attention_state = str(attention.get("state") or "selective_hold").strip()
    grice_state = str(grice.get("state") or "advise_openly").strip()
    topology_state = str(topology.get("state") or "ambient").strip()
    topology_visibility = _float01(topology.get("visibility_pressure"))
    topology_hierarchy = _float01(topology.get("hierarchy_pressure"))
    topology_threading = _float01(topology.get("threading_pressure"))
    competition_level = _float01(competition.get("competition_level"))
    expressive_state = str(expressive.get("state") or "grounded_gentle").strip()
    expressive_lightness_room = _float01(expressive.get("lightness_room"))
    playful_ceiling = _float01(relation_style.get("playful_ceiling"))
    banter_room = _float01(relation_style.get("banter_room"))
    banter_style = str(relation_style.get("banter_style") or "grounded_companion").strip() or "grounded_companion"
    lexical_variation_bias = _float01(relation_style.get("lexical_variation_bias"))
    advice_tolerance = _float01(relation_style.get("advice_tolerance"))
    cultural_state = str(culture.get("state") or "careful_polite").strip()
    cultural_joke_ratio_ceiling = _float01(culture.get("joke_ratio_ceiling"))
    cultural_politeness_pressure = _float01(culture.get("politeness_pressure"))
    cultural_group_attunement = _float01(culture.get("group_attunement"))

    suppression = _clamp01(
        (0.28 if recovery_state == "recovery_first" else 0.0)
        + (0.14 if recovery_state == "guarded" else 0.0)
        + (0.18 if body_state == "depleted" else 0.0)
        + (0.1 if body_state == "recovering" else 0.0)
        + (0.12 if budget_state == "depleted" else 0.0)
        + protection_strength * (0.18 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.08)
        + (0.16 if grice_state == "hold_obvious_advice" else 0.1 if grice_state == "attune_without_repeating" else 0.0)
        + (0.1 if attention_state == "reflex_guard" else 0.06 if attention_state == "split_guarded" else 0.0)
        + topology_visibility * 0.16
        + topology_hierarchy * 0.18
        + topology_threading * 0.08
        + competition_level * 0.12
        + cultural_politeness_pressure * 0.18
    )
    open_room = _clamp01(
        expressive_lightness_room * 0.36
        + banter_room * 0.28
        + playful_ceiling * 0.18
        + lexical_variation_bias * 0.08
        + cultural_joke_ratio_ceiling * 0.14
        + advice_tolerance * 0.08
        - suppression * 0.42
        - recovery_score * 0.08
        - max(body_score, budget_score) * 0.06
    )

    scores = {
        "open_play": _clamp01(
            0.12
            + open_room * 0.52
            + (0.08 if expressive_state == "light_playful" else 0.0)
            + playful_ceiling * 0.16
            - suppression * 0.24
        ),
        "warm_only": _clamp01(
            0.14
            + open_room * 0.22
            + advice_tolerance * 0.18
            + cultural_group_attunement * 0.14
            + (0.08 if expressive_state == "warm_companion" else 0.0)
            + (1.0 - suppression) * 0.12
            - playful_ceiling * 0.04
        ),
        "grounded_only": _clamp01(
            0.12
            + suppression * 0.18
            + (0.1 if expressive_state in {"grounded_gentle", "quiet_repair"} else 0.0)
            + (1.0 - open_room) * 0.18
            + max(body_score, budget_score) * 0.08
        ),
        "suppress_play": _clamp01(
            suppression * 0.42
            + (0.18 if protection_mode_name in {"shield", "stabilize"} else 0.0)
            + (0.12 if topology_state in {"public_visible", "hierarchical"} else 0.0)
            + (0.08 if cultural_state in {"public_courteous", "hierarchy_respectful"} else 0.0)
            + (0.08 if competition_level >= 0.28 else 0.0)
        ),
    }
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "relational_style_memory" if banter_room > 0.0 or playful_ceiling > 0.0 else "",
            f"banter_style_{banter_style}" if banter_style and banter_style != "grounded_companion" else "",
            "recovery_guard" if recovery_state in {"guarded", "recovery_first"} else "",
            "body_homeostasis" if body_state in {"recovering", "depleted"} else "",
            "homeostasis_budget" if budget_state in {"recovering", "depleted"} else "",
            "protective_mode" if protection_mode_name in {"contain", "stabilize", "shield"} else "",
            "grice_guard" if grice_state in {"attune_without_repeating", "hold_obvious_advice"} else "",
            "attention_guard" if attention_state in {"reflex_guard", "split_guarded"} else "",
            "social_topology" if topology_state in {"threaded_group", "public_visible", "hierarchical"} else "",
            "relation_competition" if competition_level >= 0.22 else "",
            "cultural_conversation" if cultural_state else "",
        ]
    )

    return LightnessBudgetState(
        state=state,
        score=_clamp01(scores.get(state, 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        banter_room=open_room,
        banter_style=banter_style,
        playful_ceiling=playful_ceiling,
        lexical_variation_bias=lexical_variation_bias,
        suppression=suppression,
        advice_tolerance=advice_tolerance,
        dominant_inputs=dominant_inputs,
    )


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "grounded_only", 0.0
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
