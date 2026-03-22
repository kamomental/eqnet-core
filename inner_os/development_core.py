from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Optional


DEVELOPMENT_SNAPSHOT_WEIGHTS = {
    "belonging_previous": 0.72,
    "belonging_culture": 0.1,
    "belonging_community": 0.12,
    "belonging_person_count": 0.05,
    "belonging_attachment": 0.06,
    "belonging_familiarity": 0.04,
    "belonging_resource_penalty": 0.04,
    "belonging_ritual": 0.03,
    "belonging_roughness_penalty": 0.04,
    "belonging_defensive_penalty": 0.03,
    "trust_previous": 0.74,
    "trust_community": 0.1,
    "trust_body_regulation": 0.06,
    "trust_voice": 0.04,
    "trust_memory": 0.06,
    "trust_hazard_penalty": 0.05,
    "trust_ritual": 0.03,
    "trust_roughness_penalty": 0.04,
    "trust_defensive_penalty": 0.06,
    "norm_previous": 0.7,
    "norm_culture": 0.12,
    "norm_role": 0.1,
    "norm_privacy": 0.08,
    "norm_safety": 0.05,
    "norm_ritual": 0.08,
    "norm_institution": 0.09,
    "norm_density": 0.04,
    "norm_roughness": 0.04,
    "norm_defensive": 0.04,
    "role_previous": 0.72,
    "role_presence": 0.1,
    "role_community": 0.08,
    "role_culture": 0.04,
    "role_alignment": 0.05,
    "role_ritual": 0.06,
    "role_institution": 0.08,
    "role_roughness_penalty": 0.04,
}

DEVELOPMENT_POST_WEIGHTS = {
    "belonging_previous": 0.82,
    "belonging_community": 0.08,
    "belonging_success": 0.06,
    "belonging_attachment": 0.04,
    "belonging_familiarity": 0.03,
    "belonging_strain_penalty": 0.04,
    "belonging_resource_penalty": 0.03,
    "belonging_ritual": 0.02,
    "belonging_roughness_penalty": 0.04,
    "belonging_defensive_penalty": 0.03,
    "trust_previous": 0.8,
    "trust_success": 0.07,
    "trust_memory": 0.05,
    "trust_strain_penalty": 0.05,
    "trust_hazard_penalty": 0.04,
    "trust_ritual": 0.02,
    "trust_roughness_penalty": 0.04,
    "trust_defensive_penalty": 0.05,
    "norm_previous": 0.83,
    "norm_role": 0.05,
    "norm_community": 0.03,
    "norm_ritual": 0.05,
    "norm_institution": 0.06,
    "norm_defensive": 0.05,
    "norm_roughness": 0.04,
    "role_previous": 0.84,
    "role_presence": 0.06,
    "role_success": 0.03,
    "role_institution": 0.04,
    "role_alignment": 0.04,
    "role_roughness_penalty": 0.04,
    "social_roughness_penalty": 0.18,
    "social_defensive_penalty": 0.12,
    "social_tentative_penalty": 0.22,
    "identity_roughness_penalty": 0.28,
    "identity_defensive_penalty": 0.18,
    "identity_tentative_penalty": 0.34,
    "belonging_blend_floor": 0.48,
    "trust_blend_floor": 0.44,
    "norm_blend_floor": 0.32,
    "role_blend_floor": 0.28,
}

DEVELOPMENT_REOPENING_WEIGHTS = {
    "social_recovery_boost": 0.12,
    "identity_recovery_boost": 0.18,
    "belonging_recovery_blend": 0.08,
    "trust_recovery_blend": 0.1,
    "norm_recovery_blend": 0.14,
    "role_recovery_blend": 0.16,
}


@dataclass
class DevelopmentState:
    belonging: float = 0.45
    trust_bias: float = 0.45
    norm_pressure: float = 0.35
    role_commitment: float = 0.4
    social_update_strength: float = 1.0
    identity_update_strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DevelopmentCore:
    """Small developmental layer linking relation, culture, and role pressure.

    This is intentionally light: it does not claim a full developmental model,
    but it gives the reusable OS a place where culture/community/role can alter
    behavior over time instead of remaining passive labels.
    """

    def snapshot(
        self,
        *,
        relational_world: Mapping[str, Any],
        sensor_input: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]] = None,
        safety_bias: float = 0.0,
        environment_pressure: Optional[Mapping[str, Any]] = None,
    ) -> DevelopmentState:
        prev_belonging = _float_from(current_state, 'belonging', 0.45)
        prev_trust = _float_from(current_state, 'trust_bias', 0.45)
        prev_norm = _float_from(current_state, 'norm_pressure', 0.35)
        prev_role = _float_from(current_state, 'role_commitment', 0.4)

        culture_present = 1.0 if relational_world.get('culture_id') else 0.0
        community_present = 1.0 if relational_world.get('community_id') else 0.0
        role_present = 1.0 if relational_world.get('social_role') else 0.0
        person_count = max(int(sensor_input.get('person_count', 0) or 0), 0)
        privacy_tags = [str(tag).lower() for tag in (sensor_input.get('privacy_tags') or [])]
        private_mode = 1.0 if 'private' in privacy_tags else 0.0
        voice_level = _float_from(sensor_input, 'voice_level', 0.0)
        body_stress = _float_from(sensor_input, 'body_stress_index', 0.0)
        attachment = _float_from(current_state, 'attachment', 0.42)
        familiarity = _float_from(current_state, 'familiarity', 0.35)
        trust_memory = _float_from(current_state, 'trust_memory', 0.45)
        role_alignment = _float_from(current_state, 'role_alignment', 0.4)
        roughness_level = _float_from(current_state, 'roughness_level', _float_from(current_state, 'terrain_transition_roughness', 0.0))
        roughness_dwell = _float_from(current_state, 'roughness_dwell', 0.0)
        defensive_level = _float_from(current_state, 'defensive_level', _float_from(current_state, 'defensive_salience', 0.0))
        defensive_dwell = _float_from(current_state, 'defensive_dwell', 0.0)
        resource_pressure = _float_from(environment_pressure, 'resource_pressure', 0.0)
        hazard_pressure = _float_from(environment_pressure, 'hazard_pressure', 0.0)
        ritual_pressure = _float_from(environment_pressure, 'ritual_pressure', 0.0)
        institutional_pressure = _float_from(environment_pressure, 'institutional_pressure', 0.0)
        social_density = _float_from(environment_pressure, 'social_density', 0.0)

        belonging = _clamp(prev_belonging * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_previous'] + culture_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_culture'] + community_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_community'] + min(person_count, 2) * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_person_count'] + attachment * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_attachment'] + familiarity * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_familiarity'] - resource_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_resource_penalty'] + ritual_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_ritual'] - roughness_dwell * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_roughness_penalty'] - defensive_dwell * DEVELOPMENT_SNAPSHOT_WEIGHTS['belonging_defensive_penalty'])
        trust_bias = _clamp(prev_trust * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_previous'] + community_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_community'] + max(0.0, 1.0 - body_stress) * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_body_regulation'] + min(voice_level, 1.0) * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_voice'] + trust_memory * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_memory'] - hazard_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_hazard_penalty'] + ritual_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_ritual'] - roughness_level * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_roughness_penalty'] - defensive_level * DEVELOPMENT_SNAPSHOT_WEIGHTS['trust_defensive_penalty'])
        norm_pressure = _clamp(prev_norm * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_previous'] + culture_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_culture'] + role_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_role'] + private_mode * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_privacy'] + safety_bias * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_safety'] + ritual_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_ritual'] + institutional_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_institution'] + social_density * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_density'] + roughness_level * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_roughness'] + defensive_level * DEVELOPMENT_SNAPSHOT_WEIGHTS['norm_defensive'])
        role_commitment = _clamp(prev_role * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_previous'] + role_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_presence'] + community_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_community'] + culture_present * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_culture'] + role_alignment * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_alignment'] + ritual_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_ritual'] + institutional_pressure * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_institution'] - roughness_dwell * DEVELOPMENT_SNAPSHOT_WEIGHTS['role_roughness_penalty'])

        return DevelopmentState(
            belonging=belonging,
            trust_bias=trust_bias,
            norm_pressure=norm_pressure,
            role_commitment=role_commitment,
            social_update_strength=1.0,
            identity_update_strength=1.0,
        )

    def post_turn(
        self,
        *,
        previous: Mapping[str, Any],
        relational_world: Mapping[str, Any],
        reply_present: bool,
        stress: float,
        recovery_need: float,
        environment_pressure: Optional[Mapping[str, Any]] = None,
        terrain_transition_roughness: float = 0.0,
        recalled_tentative_bias: float = 0.0,
        recovery_reopening: float = 0.0,
    ) -> DevelopmentState:
        prev_belonging = _float_from(previous, 'belonging', 0.45)
        prev_trust = _float_from(previous, 'trust_bias', 0.45)
        prev_norm = _float_from(previous, 'norm_pressure', 0.35)
        prev_role = _float_from(previous, 'role_commitment', 0.4)

        community_present = 1.0 if relational_world.get('community_id') else 0.0
        role_present = 1.0 if relational_world.get('social_role') else 0.0
        success = 1.0 if reply_present else 0.0
        strain = max(stress, recovery_need)
        attachment = _float_from(previous, 'attachment', 0.42)
        familiarity = _float_from(previous, 'familiarity', 0.35)
        trust_memory = _float_from(previous, 'trust_memory', 0.45)
        role_alignment = _float_from(previous, 'role_alignment', 0.4)
        roughness_level = max(_float_from(previous, 'roughness_level', 0.0), _clamp(terrain_transition_roughness))
        roughness_dwell = _float_from(previous, 'roughness_dwell', 0.0)
        defensive_level = _float_from(previous, 'defensive_level', _float_from(previous, 'defensive_salience', 0.0))
        defensive_dwell = _float_from(previous, 'defensive_dwell', 0.0)
        hazard_pressure = _float_from(environment_pressure, 'hazard_pressure', 0.0)
        ritual_pressure = _float_from(environment_pressure, 'ritual_pressure', 0.0)
        institutional_pressure = _float_from(environment_pressure, 'institutional_pressure', 0.0)
        resource_pressure = _float_from(environment_pressure, 'resource_pressure', 0.0)

        raw_belonging = _clamp(prev_belonging * DEVELOPMENT_POST_WEIGHTS['belonging_previous'] + community_present * DEVELOPMENT_POST_WEIGHTS['belonging_community'] + success * DEVELOPMENT_POST_WEIGHTS['belonging_success'] + attachment * DEVELOPMENT_POST_WEIGHTS['belonging_attachment'] + familiarity * DEVELOPMENT_POST_WEIGHTS['belonging_familiarity'] - strain * DEVELOPMENT_POST_WEIGHTS['belonging_strain_penalty'] - resource_pressure * DEVELOPMENT_POST_WEIGHTS['belonging_resource_penalty'] + ritual_pressure * DEVELOPMENT_POST_WEIGHTS['belonging_ritual'] - roughness_dwell * DEVELOPMENT_POST_WEIGHTS['belonging_roughness_penalty'] - defensive_dwell * DEVELOPMENT_POST_WEIGHTS['belonging_defensive_penalty'])
        raw_trust = _clamp(prev_trust * DEVELOPMENT_POST_WEIGHTS['trust_previous'] + success * DEVELOPMENT_POST_WEIGHTS['trust_success'] + trust_memory * DEVELOPMENT_POST_WEIGHTS['trust_memory'] - strain * DEVELOPMENT_POST_WEIGHTS['trust_strain_penalty'] - hazard_pressure * DEVELOPMENT_POST_WEIGHTS['trust_hazard_penalty'] + ritual_pressure * DEVELOPMENT_POST_WEIGHTS['trust_ritual'] - roughness_level * DEVELOPMENT_POST_WEIGHTS['trust_roughness_penalty'] - defensive_level * DEVELOPMENT_POST_WEIGHTS['trust_defensive_penalty'])
        raw_norm = _clamp(prev_norm * DEVELOPMENT_POST_WEIGHTS['norm_previous'] + role_present * DEVELOPMENT_POST_WEIGHTS['norm_role'] + community_present * DEVELOPMENT_POST_WEIGHTS['norm_community'] + ritual_pressure * DEVELOPMENT_POST_WEIGHTS['norm_ritual'] + institutional_pressure * DEVELOPMENT_POST_WEIGHTS['norm_institution'] + defensive_level * DEVELOPMENT_POST_WEIGHTS['norm_defensive'] + roughness_level * DEVELOPMENT_POST_WEIGHTS['norm_roughness'])
        raw_role = _clamp(prev_role * DEVELOPMENT_POST_WEIGHTS['role_previous'] + role_present * DEVELOPMENT_POST_WEIGHTS['role_presence'] + success * DEVELOPMENT_POST_WEIGHTS['role_success'] + institutional_pressure * DEVELOPMENT_POST_WEIGHTS['role_institution'] + role_alignment * DEVELOPMENT_POST_WEIGHTS['role_alignment'] - roughness_dwell * DEVELOPMENT_POST_WEIGHTS['role_roughness_penalty'])

        transition_roughness = _clamp(terrain_transition_roughness)
        tentative_bias = _clamp(recalled_tentative_bias)
        reopening = _clamp(recovery_reopening)
        social_update_strength = _clamp(1.0 - transition_roughness * DEVELOPMENT_POST_WEIGHTS['social_roughness_penalty'] - defensive_dwell * DEVELOPMENT_POST_WEIGHTS['social_defensive_penalty'] - tentative_bias * DEVELOPMENT_POST_WEIGHTS['social_tentative_penalty'] + reopening * DEVELOPMENT_REOPENING_WEIGHTS['social_recovery_boost'])
        identity_update_strength = _clamp(1.0 - transition_roughness * DEVELOPMENT_POST_WEIGHTS['identity_roughness_penalty'] - defensive_dwell * DEVELOPMENT_POST_WEIGHTS['identity_defensive_penalty'] - tentative_bias * DEVELOPMENT_POST_WEIGHTS['identity_tentative_penalty'] + reopening * DEVELOPMENT_REOPENING_WEIGHTS['identity_recovery_boost'])

        belonging = _blend(prev_belonging, raw_belonging, max(DEVELOPMENT_POST_WEIGHTS['belonging_blend_floor'], social_update_strength + reopening * DEVELOPMENT_REOPENING_WEIGHTS['belonging_recovery_blend']))
        trust_bias = _blend(prev_trust, raw_trust, max(DEVELOPMENT_POST_WEIGHTS['trust_blend_floor'], social_update_strength + reopening * DEVELOPMENT_REOPENING_WEIGHTS['trust_recovery_blend']))
        norm_pressure = _blend(prev_norm, raw_norm, max(DEVELOPMENT_POST_WEIGHTS['norm_blend_floor'], identity_update_strength + reopening * DEVELOPMENT_REOPENING_WEIGHTS['norm_recovery_blend']))
        role_commitment = _blend(prev_role, raw_role, max(DEVELOPMENT_POST_WEIGHTS['role_blend_floor'], identity_update_strength + reopening * DEVELOPMENT_REOPENING_WEIGHTS['role_recovery_blend']))

        return DevelopmentState(
            belonging=belonging,
            trust_bias=trust_bias,
            norm_pressure=norm_pressure,
            role_commitment=role_commitment,
            social_update_strength=social_update_strength,
            identity_update_strength=identity_update_strength,
        )

    def memory_kind_biases(
        self,
        *,
        state: Mapping[str, Any],
    ) -> Dict[str, float]:
        belonging = _float_from(state, 'belonging', 0.45)
        trust_bias = _float_from(state, 'trust_bias', 0.45)
        norm_pressure = _float_from(state, 'norm_pressure', 0.35)
        role_commitment = _float_from(state, 'role_commitment', 0.4)
        return {
            'observed_real': 0.04 * trust_bias + 0.05 * norm_pressure,
            'verified': 0.03 * trust_bias + 0.06 * norm_pressure,
            'reconstructed': 0.05 * belonging + 0.04 * role_commitment,
            'relationship_trace': 0.07 * belonging + 0.05 * role_commitment,
            'identity_trace': 0.04 * role_commitment + 0.03 * norm_pressure,
            'experienced_sim': 0.03 * role_commitment,
            'transferred_learning': 0.04 * role_commitment + 0.02 * trust_bias,
        }

    def apply_transferred_learning(
        self,
        *,
        previous: Mapping[str, Any],
        lessons: Iterable[Mapping[str, Any]],
    ) -> DevelopmentState:
        state = DevelopmentState(
            belonging=_float_from(previous, 'belonging', 0.45),
            trust_bias=_float_from(previous, 'trust_bias', 0.45),
            norm_pressure=_float_from(previous, 'norm_pressure', 0.35),
            role_commitment=_float_from(previous, 'role_commitment', 0.4),
            social_update_strength=_float_from(previous, 'social_update_strength', 1.0),
            identity_update_strength=_float_from(previous, 'identity_update_strength', 1.0),
        )
        for lesson in lessons:
            confidence = _float_from(lesson, 'confidence', 0.4)
            scale = max(0.0, min(1.0, confidence)) * 0.12
            hint = str(lesson.get('policy_hint') or '').strip().lower()
            if hint == 'pause_and_observe_under_ambiguity':
                state.norm_pressure = _clamp(state.norm_pressure + scale)
                state.role_commitment = _clamp(state.role_commitment + scale * 0.65)
            elif hint == 'gentle_clarification_before_commitment':
                state.trust_bias = _clamp(state.trust_bias + scale * 0.7)
                state.norm_pressure = _clamp(state.norm_pressure + scale * 0.55)
            elif hint == 'preserve_boundary_before_escalation':
                state.norm_pressure = _clamp(state.norm_pressure + scale * 0.8)
                state.role_commitment = _clamp(state.role_commitment + scale * 0.45)
                state.trust_bias = _clamp(state.trust_bias - scale * 0.15)
            else:
                state.belonging = _clamp(state.belonging + scale * 0.35)
                state.trust_bias = _clamp(state.trust_bias + scale * 0.25)
        return state


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _blend(previous: float, updated: float, strength: float) -> float:
    return _clamp(previous + (updated - previous) * _clamp(strength))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
