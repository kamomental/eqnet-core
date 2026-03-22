from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


@dataclass
class PersonalityIndexState:
    caution_bias: float = 0.4
    affiliation_bias: float = 0.45
    exploration_bias: float = 0.4
    reflective_bias: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonalityIndexCore:
    """Derive slower personality-facing biases from inner state."""

    def snapshot(
        self,
        *,
        current_state: Optional[Mapping[str, Any]] = None,
        development: Optional[Mapping[str, Any]] = None,
        relationship: Optional[Mapping[str, Any]] = None,
        environment_pressure: Optional[Mapping[str, Any]] = None,
    ) -> PersonalityIndexState:
        stress = _float_from(current_state, 'stress', 0.0)
        temporal_pressure = _float_from(current_state, 'temporal_pressure', 0.0)
        trust_bias = _float_from(development, 'trust_bias', _float_from(current_state, 'trust_bias', 0.45))
        belonging = _float_from(development, 'belonging', _float_from(current_state, 'belonging', 0.45))
        norm_pressure = _float_from(development, 'norm_pressure', _float_from(current_state, 'norm_pressure', 0.35))
        role_commitment = _float_from(development, 'role_commitment', _float_from(current_state, 'role_commitment', 0.4))
        attachment = _float_from(relationship, 'attachment', _float_from(current_state, 'attachment', 0.42))
        familiarity = _float_from(relationship, 'familiarity', _float_from(current_state, 'familiarity', 0.35))
        rupture = _float_from(relationship, 'rupture_sensitivity', _float_from(current_state, 'rupture_sensitivity', 0.38))
        trust_memory = _float_from(relationship, 'trust_memory', _float_from(current_state, 'trust_memory', 0.45))
        role_alignment = _float_from(relationship, 'role_alignment', _float_from(current_state, 'role_alignment', 0.4))
        continuity_score = _float_from(current_state, 'continuity_score', 0.48)
        social_grounding = _float_from(current_state, 'social_grounding', 0.44)
        recent_strain = _float_from(current_state, 'recent_strain', 0.32)
        culture_resonance = _float_from(current_state, 'culture_resonance', 0.0)
        community_resonance = _float_from(current_state, 'community_resonance', 0.0)
        roughness_level = _float_from(current_state, 'roughness_level', _float_from(current_state, 'terrain_transition_roughness', 0.0))
        roughness_dwell = _float_from(current_state, 'roughness_dwell', 0.0)
        defensive_level = _float_from(current_state, 'defensive_level', _float_from(current_state, 'defensive_salience', 0.0))
        defensive_dwell = _float_from(current_state, 'defensive_dwell', 0.0)
        hazard = _float_from(environment_pressure, 'hazard_pressure', 0.0)
        scarcity = _float_from(environment_pressure, 'resource_pressure', 0.0)
        institutional = _float_from(environment_pressure, 'institutional_pressure', 0.0)
        ritual = _float_from(environment_pressure, 'ritual_pressure', 0.0)

        caution_bias = _clamp(0.18 + stress * 0.22 + temporal_pressure * 0.12 + hazard * 0.18 + scarcity * 0.1 + rupture * 0.15 + norm_pressure * 0.12 + recent_strain * 0.1 + roughness_level * 0.12 + roughness_dwell * 0.08 + defensive_level * 0.14 + defensive_dwell * 0.09 - continuity_score * 0.05 - community_resonance * 0.04)
        affiliation_bias = _clamp(0.18 + attachment * 0.22 + belonging * 0.2 + trust_bias * 0.18 + familiarity * 0.1 + trust_memory * 0.08 + social_grounding * 0.08 + culture_resonance * 0.08 + community_resonance * 0.12 - roughness_dwell * 0.05 - defensive_level * 0.06 - defensive_dwell * 0.04)
        exploration_bias = _clamp(0.22 + (1.0 - caution_bias) * 0.22 + trust_bias * 0.14 + familiarity * 0.06 + role_alignment * 0.06 + continuity_score * 0.06 + community_resonance * 0.04 + (1.0 - institutional) * 0.1 - hazard * 0.12 - recent_strain * 0.05 - roughness_level * 0.08 - defensive_dwell * 0.08)
        reflective_bias = _clamp(0.18 + temporal_pressure * 0.18 + norm_pressure * 0.16 + role_commitment * 0.14 + ritual * 0.1 + institutional * 0.08 + continuity_score * 0.05 + social_grounding * 0.04 + culture_resonance * 0.06 + roughness_dwell * 0.05 + defensive_dwell * 0.04)

        return PersonalityIndexState(
            caution_bias=caution_bias,
            affiliation_bias=affiliation_bias,
            exploration_bias=exploration_bias,
            reflective_bias=reflective_bias,
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
