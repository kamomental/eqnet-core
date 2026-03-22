from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class PersistenceState:
    continuity_score: float = 0.48
    social_grounding: float = 0.44
    recent_strain: float = 0.32
    culture_resonance: float = 0.0
    community_resonance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersistenceCore:
    """Keep slow continuity traces so repeated turns remain one life."""

    def snapshot(
        self,
        *,
        current_state: Optional[Mapping[str, Any]] = None,
        development: Optional[Mapping[str, Any]] = None,
        relationship: Optional[Mapping[str, Any]] = None,
        personality: Optional[Mapping[str, Any]] = None,
        environment_pressure: Optional[Mapping[str, Any]] = None,
        transition_signal: Optional[Mapping[str, Any]] = None,
    ) -> PersistenceState:
        previous_continuity = _float_from(current_state, "continuity_score", 0.48)
        previous_grounding = _float_from(current_state, "social_grounding", 0.44)
        previous_strain = _float_from(current_state, "recent_strain", 0.32)
        previous_culture_resonance = _float_from(current_state, "culture_resonance", 0.0)
        previous_community_resonance = _float_from(current_state, "community_resonance", 0.0)
        transition_intensity = _float_from(transition_signal, "transition_intensity", 0.0)
        roughness_level = _float_from(current_state, "roughness_level", _float_from(current_state, "terrain_transition_roughness", 0.0))
        roughness_dwell = _float_from(current_state, "roughness_dwell", 0.0)
        defensive_level = _float_from(current_state, "defensive_level", _float_from(current_state, "defensive_salience", 0.0))
        defensive_dwell = _float_from(current_state, "defensive_dwell", 0.0)

        belonging = _float_from(development, "belonging", _float_from(current_state, "belonging", 0.45))
        trust_bias = _float_from(development, "trust_bias", _float_from(current_state, "trust_bias", 0.45))
        attachment = _float_from(relationship, "attachment", _float_from(current_state, "attachment", 0.42))
        familiarity = _float_from(relationship, "familiarity", _float_from(current_state, "familiarity", 0.35))
        caution_bias = _float_from(personality, "caution_bias", _float_from(current_state, "caution_bias", 0.4))
        affiliation_bias = _float_from(personality, "affiliation_bias", _float_from(current_state, "affiliation_bias", 0.45))
        hazard_pressure = _float_from(environment_pressure, "hazard_pressure", 0.0)
        scarcity = _float_from(environment_pressure, "resource_pressure", 0.0)
        institutional = _float_from(environment_pressure, "institutional_pressure", 0.0)
        norm_pressure = _float_from(development, "norm_pressure", _float_from(current_state, "norm_pressure", 0.35))
        stress = _float_from(current_state, "stress", 0.0)
        temporal_pressure = _float_from(current_state, "temporal_pressure", 0.0)

        social_grounding = _clamp(
            previous_grounding * 0.72
            + attachment * 0.14
            + familiarity * 0.1
            + belonging * 0.08
            + trust_bias * 0.06
            - roughness_level * 0.04
            - roughness_dwell * 0.05
            - defensive_level * 0.05
            - defensive_dwell * 0.04
        )
        recent_strain = _clamp(
            previous_strain * 0.68
            + stress * 0.16
            + temporal_pressure * 0.09
            + hazard_pressure * 0.1
            + scarcity * 0.08
            + institutional * 0.05
            + transition_intensity * 0.12
            + roughness_level * 0.08
            + roughness_dwell * 0.05
            + defensive_level * 0.08
            + defensive_dwell * 0.05
        )
        culture_resonance = _clamp(
            previous_culture_resonance * (0.78 - transition_intensity * 0.08)
            + belonging * 0.1
            + norm_pressure * 0.12
            + (1.0 - scarcity) * 0.03
            + institutional * 0.04
            - roughness_level * 0.02
            - defensive_level * 0.02
        )
        community_resonance = _clamp(
            previous_community_resonance * (0.78 - transition_intensity * 0.12)
            + social_grounding * 0.12
            + trust_bias * 0.1
            + attachment * 0.08
            + familiarity * 0.06
            - recent_strain * 0.04
            - roughness_dwell * 0.02
            - defensive_dwell * 0.03
        )
        continuity_score = _clamp(
            previous_continuity * 0.74
            + social_grounding * 0.16
            + affiliation_bias * 0.08
            + community_resonance * 0.05
            - recent_strain * 0.08
            - transition_intensity * 0.06
            - caution_bias * 0.03
            - norm_pressure * 0.02
            - roughness_dwell * 0.04
            - defensive_dwell * 0.05
        )
        return PersistenceState(
            continuity_score=continuity_score,
            social_grounding=social_grounding,
            recent_strain=recent_strain,
            culture_resonance=culture_resonance,
            community_resonance=community_resonance,
        )

    def post_turn(
        self,
        *,
        previous: Optional[Mapping[str, Any]] = None,
        development: Optional[Mapping[str, Any]] = None,
        relationship: Optional[Mapping[str, Any]] = None,
        personality: Optional[Mapping[str, Any]] = None,
        environment_pressure: Optional[Mapping[str, Any]] = None,
        transition_signal: Optional[Mapping[str, Any]] = None,
        reply_present: bool,
        reconstructed_memory_appended: bool,
        transferred_lessons_used: int = 0,
    ) -> PersistenceState:
        current = self.snapshot(
            current_state=previous,
            development=development,
            relationship=relationship,
            personality=personality,
            environment_pressure=environment_pressure,
            transition_signal=transition_signal,
        )
        continuity_score = current.continuity_score
        social_grounding = current.social_grounding
        recent_strain = current.recent_strain

        if reply_present:
            continuity_score = _clamp(continuity_score + 0.05)
            social_grounding = _clamp(social_grounding + 0.03)
            recent_strain = _clamp(recent_strain - 0.03)
        if reconstructed_memory_appended:
            continuity_score = _clamp(continuity_score + 0.05)
        if transferred_lessons_used > 0:
            continuity_score = _clamp(continuity_score + min(0.06, 0.02 * transferred_lessons_used))
        culture_resonance = current.culture_resonance
        community_resonance = current.community_resonance

        transition_intensity = _float_from(transition_signal, "transition_intensity", 0.0)
        if recent_strain > 0.62:
            continuity_score = _clamp(continuity_score - 0.05)
        if transition_intensity > 0.0:
            culture_resonance = _clamp(culture_resonance - transition_intensity * 0.04)
            community_resonance = _clamp(community_resonance - transition_intensity * 0.06)
        if reply_present:
            community_resonance = _clamp(community_resonance + 0.03)
        if reconstructed_memory_appended:
            culture_resonance = _clamp(culture_resonance + 0.03)
        if transferred_lessons_used > 0:
            culture_resonance = _clamp(culture_resonance + min(0.05, 0.015 * transferred_lessons_used))

        return PersistenceState(
            continuity_score=continuity_score,
            social_grounding=social_grounding,
            recent_strain=recent_strain,
            culture_resonance=culture_resonance,
            community_resonance=community_resonance,
        )


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
