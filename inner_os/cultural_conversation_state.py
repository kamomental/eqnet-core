from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from eqnet.culture_model import CultureContext, compute_culture_state, culture_to_behavior


@dataclass(frozen=True)
class CulturalConversationState:
    state: str
    score: float
    winner_margin: float
    scores: dict[str, float]
    tone: str
    directness_ceiling: float
    joke_ratio_ceiling: float
    politeness_pressure: float
    group_attunement: float
    compaction_bias: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "winner_margin": round(self.winner_margin, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "tone": self.tone,
            "directness_ceiling": round(self.directness_ceiling, 4),
            "joke_ratio_ceiling": round(self.joke_ratio_ceiling, 4),
            "politeness_pressure": round(self.politeness_pressure, 4),
            "group_attunement": round(self.group_attunement, 4),
            "compaction_bias": round(self.compaction_bias, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_cultural_conversation_state(
    *,
    self_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    relation_competition_state: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
) -> CulturalConversationState:
    state = dict(self_state or {})
    topology = dict(social_topology_state or {})
    competition = dict(relation_competition_state or {})
    relation_style = dict(relational_style_memory_state or {})
    recovery = dict(body_recovery_guard or {})

    culture_id = str(state.get("culture_id") or "default").strip() or "default"
    community_id = str(state.get("community_id") or "").strip()
    dominant_person_id = str(competition.get("dominant_person_id") or state.get("related_person_id") or "").strip()
    culture_resonance = _float01(state.get("culture_resonance"))
    profile = _culture_profile(culture_id)

    try:
        behavior = culture_to_behavior(
            compute_culture_state(
                CultureContext(
                    culture_tag=culture_id,
                    place_id=community_id or None,
                    partner_id=dominant_person_id or None,
                )
            )
        )
    except Exception:
        behavior = None

    topology_state = str(topology.get("state") or "ambient").strip()
    visibility_pressure = _float01(topology.get("visibility_pressure"))
    threading_pressure = _float01(topology.get("threading_pressure"))
    hierarchy_pressure = _float01(topology.get("hierarchy_pressure"))
    total_people = int(competition.get("total_people") or topology.get("total_people") or 0)
    playful_ceiling = _float01(relation_style.get("playful_ceiling"))
    advice_tolerance = _float01(relation_style.get("advice_tolerance"))
    lexical_familiarity = _float01(relation_style.get("lexical_familiarity"))
    warmth_bias = _float01(relation_style.get("warmth_bias"))
    guard_state = str(recovery.get("state") or "open").strip()
    guard_score = _float01(recovery.get("score"))

    behavior_directness = _float01(getattr(behavior, "directness", 0.6) if behavior else 0.6)
    behavior_joke_ratio = _float01(getattr(behavior, "joke_ratio", 0.3) if behavior else 0.3)
    behavior_tone = str(getattr(behavior, "tone", "neutral") if behavior else "neutral").strip() or "neutral"
    base_politeness = _float01(getattr(compute_culture_state(CultureContext(culture_tag=culture_id, place_id=community_id or None, partner_id=dominant_person_id or None)), "politeness", 0.0))

    public_restraint = _float01(profile.get("public_restraint"))
    hierarchy_restraint = _float01(profile.get("hierarchy_restraint"))
    group_harmony = _float01(profile.get("group_harmony"))
    compaction_bias = _clamp11(_float(profile.get("compaction_bias")))

    politeness_pressure = _float01(
        base_politeness * 0.42
        + public_restraint * max(visibility_pressure, 0.22 if topology_state == "public_visible" else 0.0)
        + hierarchy_restraint * max(hierarchy_pressure, 0.24 if topology_state == "hierarchical" else 0.0)
        + (0.08 if topology_state == "public_visible" else 0.0)
        + (0.06 if topology_state == "hierarchical" else 0.0)
        + (0.12 if guard_state in {"guarded", "recovery_first"} else 0.0)
        + culture_resonance * 0.08
    )
    group_attunement = _float01(
        group_harmony * (0.34 + threading_pressure * 0.42 + (0.18 if total_people > 1 else 0.0))
        + lexical_familiarity * 0.12
        + warmth_bias * 0.12
        + advice_tolerance * 0.08
        + (0.06 if topology_state == "threaded_group" else 0.0)
        - guard_score * 0.1
    )
    directness_ceiling = _float01(
        behavior_directness
        + _float(profile.get("directness_bias")) * 0.7
        - politeness_pressure * 0.34
        - hierarchy_pressure * 0.14
        - visibility_pressure * 0.08
        + culture_resonance * 0.06
    )
    joke_ratio_ceiling = _float01(
        behavior_joke_ratio
        + _float(profile.get("joke_ratio_bias")) * 0.72
        + playful_ceiling * 0.24
        + lexical_familiarity * 0.08
        - politeness_pressure * 0.28
        - visibility_pressure * 0.14
        - hierarchy_pressure * 0.18
        - guard_score * 0.12
    )
    register_warmth = _float01(
        warmth_bias * 0.46
        + _float01(profile.get("warmth_bias")) * 0.24
        + group_attunement * 0.18
        + advice_tolerance * 0.08
    )

    scores = {
        "casual_shared": _float01(
            0.08
            + directness_ceiling * 0.22
            + joke_ratio_ceiling * 0.28
            + register_warmth * 0.16
            + (0.12 if topology_state in {"ambient", "one_to_one"} else 0.0)
            - politeness_pressure * 0.18
        ),
        "group_attuned": _float01(
            0.1
            + group_attunement * 0.32
            + register_warmth * 0.14
            + joke_ratio_ceiling * 0.1
            + threading_pressure * 0.18
            + (0.1 if topology_state == "threaded_group" else 0.0)
            - hierarchy_pressure * 0.1
        ),
        "public_courteous": _float01(
            0.08
            + politeness_pressure * 0.32
            + visibility_pressure * 0.26
            + (0.12 if topology_state == "public_visible" else 0.0)
            + (1.0 - directness_ceiling) * 0.08
            + (1.0 - joke_ratio_ceiling) * 0.08
        ),
        "hierarchy_respectful": _float01(
            0.08
            + politeness_pressure * 0.3
            + hierarchy_pressure * 0.34
            + (0.14 if topology_state == "hierarchical" else 0.0)
            + (1.0 - directness_ceiling) * 0.08
        ),
        "careful_polite": _float01(
            0.1
            + politeness_pressure * 0.18
            + guard_score * 0.16
            + (0.1 if guard_state in {"guarded", "recovery_first"} else 0.0)
            + (1.0 - joke_ratio_ceiling) * 0.08
            + (1.0 - directness_ceiling) * 0.08
        ),
    }
    register_name, winner_margin = _winner_and_margin(scores)
    tone = behavior_tone
    if register_name in {"public_courteous", "hierarchy_respectful", "careful_polite"}:
        tone = "polite"
    elif register_name == "casual_shared" and joke_ratio_ceiling >= 0.36:
        tone = "casual"

    dominant_inputs = _compact(
        [
            f"culture:{culture_id}" if culture_id else "",
            "culture_resonance" if culture_resonance >= 0.18 else "",
            "public_topology" if topology_state == "public_visible" else "",
            "hierarchical_topology" if topology_state == "hierarchical" else "",
            "threaded_group" if topology_state == "threaded_group" else "",
            "relational_playful_ceiling" if playful_ceiling >= 0.34 else "",
            "group_harmony" if group_attunement >= 0.28 else "",
            "guarded_body" if guard_state in {"guarded", "recovery_first"} else "",
        ]
    )
    return CulturalConversationState(
        state=register_name,
        score=_float01(scores.get(register_name, 0.0)),
        winner_margin=winner_margin,
        scores=scores,
        tone=tone,
        directness_ceiling=directness_ceiling,
        joke_ratio_ceiling=joke_ratio_ceiling,
        politeness_pressure=politeness_pressure,
        group_attunement=group_attunement,
        compaction_bias=compaction_bias,
        dominant_inputs=dominant_inputs,
    )


@lru_cache(maxsize=1)
def _load_profiles() -> dict[str, dict[str, float]]:
    config_path = Path(__file__).resolve().parents[1] / "emot_terrain_lab" / "config" / "cultural_conversation.yaml"
    if not config_path.exists():
        return {"default": {}}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    result: dict[str, dict[str, float]] = {}
    if isinstance(raw, Mapping):
        for key, payload in raw.items():
            if not isinstance(payload, Mapping):
                continue
            result[str(key)] = {str(inner): _float(value) for inner, value in payload.items()}
    return result or {"default": {}}


def _culture_profile(culture_id: str) -> dict[str, float]:
    profiles = _load_profiles()
    text = str(culture_id or "").strip()
    if text in profiles:
        return dict(profiles[text])
    prefix = text.split("-")[0] if "-" in text else text
    for key, payload in profiles.items():
        if key.split("-")[0] == prefix and prefix:
            return dict(payload)
    return dict(profiles.get("default") or {})


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _float01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "careful_polite", 0.0
    winner_name, winner_score = items[0]
    runner_up = items[1][1] if len(items) > 1 else 0.0
    return winner_name, round(_float01(winner_score - runner_up), 4)


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp11(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
