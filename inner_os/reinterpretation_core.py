from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


REINTERPRETATION_WEIGHTS = {
    "social_self_norm": 0.55,
    "social_self_role": 0.3,
    "social_self_culture_presence": 0.08,
    "social_self_community_presence": 0.07,
    "social_self_density": 0.08,
    "social_self_institution": 0.08,
    "social_self_recent_strain": 0.06,
    "social_self_culture_resonance": 0.08,
    "social_self_community_resonance": 0.1,
    "social_self_transition": 0.1,
    "social_self_profile_ritual": 0.08,
    "social_self_profile_institution": 0.08,
    "reflective_temporal": 0.35,
    "reflective_social_self": 0.4,
    "reflective_low_trust": 0.18,
    "reflective_low_belonging": 0.07,
    "reflective_hazard": 0.12,
    "reflective_resource": 0.08,
    "reflective_recent_strain": 0.14,
    "reflective_discontinuity": 0.08,
    "reflective_transition": 0.12,
    "meaning_reflective": 0.5,
    "meaning_social_self": 0.28,
    "meaning_ritual": 0.12,
    "meaning_institution": 0.08,
    "meaning_recent_strain": 0.08,
    "meaning_culture_resonance": 0.05,
    "meaning_community_resonance": 0.06,
    "meaning_transition": 0.12,
    "meaning_profile_ritual": 0.06,
    "meaning_profile_institution": 0.05,
    "meaning_sim_bonus": 0.08,
    "meaning_roughness_penalty": 0.12,
    "meaning_profile_roughness_penalty": 0.08,
    "meaning_check_in_afterglow_penalty": 0.12,
    "meaning_afterglow_penalty": 0.08,
    "meaning_inertia_penalty": 0.16,
    "meaning_clarity_penalty": 0.06,
    "meaning_reopening": 0.1,
    "meaning_fragility_penalty": 0.1,
    "meaning_avoidance_penalty": 0.12,
    "meaning_attachment_hold": 0.06,
    "meaning_defensive_penalty": 0.1,
    "meaning_near_body_penalty": 0.06,
    "meaning_approach_bonus": 0.04,
    "narrative_belonging": 0.35,
    "narrative_role": 0.35,
    "narrative_trust": 0.2,
    "narrative_community_presence": 0.1,
    "narrative_social_grounding": 0.08,
    "narrative_culture_resonance": 0.08,
    "narrative_community_resonance": 0.1,
    "narrative_profile_ritual": 0.05,
    "narrative_profile_institution": 0.05,
    "narrative_recent_strain_penalty": 0.06,
    "narrative_transition_penalty": 0.08,
    "narrative_roughness_penalty": 0.15,
    "narrative_profile_roughness_penalty": 0.06,
    "narrative_afterglow_penalty": 0.04,
    "narrative_anticipation_penalty": 0.04,
    "narrative_replay_penalty": 0.02,
    "narrative_reopening": 0.06,
    "narrative_attachment": 0.08,
    "narrative_fragility_penalty": 0.04,
    "narrative_defensive_penalty": 0.04,
    "narrative_approach_bonus": 0.03,
    "reconstructed_meaning_floor": 0.16,
    "reconstructed_meaning_roughness": 0.05,
    "reconstructed_reflective_floor": 0.2,
    "reconstructed_reflective_roughness": 0.03,
    "tentative_roughness": 0.62,
    "tentative_reflective": 0.08,
    "tentative_low_narrative": 0.04,
    "confidence_base": 0.32,
    "confidence_meaning": 0.35,
    "confidence_narrative": 0.15,
    "confidence_tentative_penalty": 0.12,
    "confidence_afterglow_penalty": 0.04,
    "confidence_anticipation_penalty": 0.04,
    "confidence_inertia_penalty": 0.05,
    "confidence_reopening": 0.04,
}

REINTERPRETATION_THRESHOLDS = {
    "check_in_afterglow": 0.24,
    "stabilizing_anticipation": 0.44,
    "stabilizing_inertia": 0.4,
    "reopening_recovery": 0.24,
    "reopening_clarity": 0.38,
    "reopening_inertia_ceiling": 0.5,
    "community_transition": 0.35,
    "community_profile_reframing": 0.46,
    "grounding_deferral_roughness": 0.48,
    "grounding_deferral_defensive": 0.34,
    "grounding_deferral_narrative": 0.4,
    "grounding_deferral_grounding": 0.38,
    "social_reframing_pressure": 0.6,
    "reflective_reconsolidation": 0.45,
    "light_reinterpretation_meaning": 0.28,
    "light_reinterpretation_social": 0.42,
}

MEANING_ALLOCATION_WEIGHTS = {
    "base": 0.5,
    "reflective_push": 0.24,
    "social_push": 0.14,
    "ritual_push": 0.06,
    "institution_push": 0.04,
    "transition_push": 0.06,
    "sim_push": 0.04,
    "roughness_hold": 0.14,
    "check_in_hold": 0.12,
    "afterglow_hold": 0.08,
    "inertia_hold": 0.16,
    "clarity_hold": 0.08,
    "reopening_push": 0.1,
}


@dataclass
class ReinterpretationSnapshot:
    mode: str = 'steady_recall'
    reflective_tension: float = 0.0
    social_self_pressure: float = 0.0
    meaning_shift: float = 0.0
    meaning_push: float = 0.0
    meaning_hold: float = 0.0
    narrative_pull: float = 0.0
    terrain_transition_roughness: float = 0.0
    interaction_afterglow: float = 0.0
    interaction_afterglow_intent: Optional[str] = None
    recovery_reopening: float = 0.0
    community_profile_pressure: float = 0.0
    object_affordance_bias: float = 0.0
    fragility_guard: float = 0.0
    object_attachment: float = 0.0
    object_avoidance: float = 0.0
    reachability: float = 0.0
    near_body_risk: float = 0.0
    defensive_salience: float = 0.0
    approach_confidence: float = 0.0
    summary: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReinterpretationCore:
    """Apply developmental and social pressure to recalled memory.

    This keeps reinterpretation outside the LLM. Recall can become more careful,
    duty-shaped, or transfer-cautious depending on culture, community, and role.
    """

    def snapshot(
        self,
        *,
        recall_payload: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]] = None,
        relational_world: Optional[Mapping[str, Any]] = None,
        environment_pressure: Optional[Mapping[str, Any]] = None,
        transition_signal: Optional[Mapping[str, Any]] = None,
    ) -> ReinterpretationSnapshot:
        record_kind = str(recall_payload.get('record_kind') or recall_payload.get('kind') or 'observed_real').strip().lower()
        norm_pressure = _float_from(current_state, 'norm_pressure', 0.35)
        trust_bias = _float_from(current_state, 'trust_bias', 0.45)
        belonging = _float_from(current_state, 'belonging', 0.45)
        role_commitment = _float_from(current_state, 'role_commitment', 0.4)
        temporal_pressure = _float_from(current_state, 'temporal_pressure', 0.0)
        continuity_score = _float_from(current_state, 'continuity_score', 0.48)
        social_grounding = _float_from(current_state, 'social_grounding', 0.44)
        recent_strain = _float_from(current_state, 'recent_strain', 0.32)
        culture_resonance = _float_from(current_state, 'culture_resonance', 0.0)
        community_resonance = _float_from(current_state, 'community_resonance', 0.0)
        ritual_memory = _float_from(current_state, 'ritual_memory', 0.0)
        institutional_memory = _float_from(current_state, 'institutional_memory', 0.0)
        roughness_dwell = _float_from(current_state, 'roughness_dwell', 0.0)
        defensive_dwell = _float_from(current_state, 'defensive_dwell', 0.0)
        resource_pressure = _float_from(environment_pressure, 'resource_pressure', 0.0)
        hazard_pressure = _float_from(environment_pressure, 'hazard_pressure', 0.0)
        ritual_pressure = _float_from(environment_pressure, 'ritual_pressure', 0.0)
        institutional_pressure = _float_from(environment_pressure, 'institutional_pressure', 0.0)
        social_density = _float_from(environment_pressure, 'social_density', 0.0)
        culture_present = 1.0 if _text(relational_world, 'culture_id') else 0.0
        community_present = 1.0 if _text(relational_world, 'community_id') else 0.0
        role_present = 1.0 if _text(relational_world, 'social_role') else 0.0
        transition_intensity = _float_from(transition_signal, 'transition_intensity', 0.0)
        terrain_transition_roughness = _float_from(current_state, 'terrain_transition_roughness', 0.0)
        interaction_afterglow = _float_from(current_state, 'interaction_afterglow', 0.0)
        interaction_afterglow_intent = _text(current_state, 'interaction_afterglow_intent')
        replay_intensity = _float_from(current_state, 'replay_intensity', 0.0)
        anticipation_tension = _float_from(current_state, 'anticipation_tension', 0.0)
        stabilization_drive = _float_from(current_state, 'stabilization_drive', 0.0)
        relational_clarity = _float_from(current_state, 'relational_clarity', 0.0)
        meaning_inertia = _float_from(current_state, 'meaning_inertia', 0.0)
        recovery_reopening = _float_from(current_state, 'recovery_reopening', 0.0)
        object_affordance_bias = _float_from(current_state, 'object_affordance_bias', 0.0)
        fragility_guard = _float_from(current_state, 'fragility_guard', 0.0)
        object_attachment = _float_from(current_state, 'object_attachment', 0.0)
        object_avoidance = _float_from(current_state, 'object_avoidance', 0.0)
        reachability = _float_from(current_state, 'reachability', 0.0)
        near_body_risk = _float_from(current_state, 'near_body_risk', 0.0)
        defensive_salience = _float_from(current_state, 'defensive_salience', 0.0)
        approach_confidence = _float_from(current_state, 'approach_confidence', 0.0)

        community_profile_pressure = _clamp(
            ritual_memory * 0.46
            + institutional_memory * 0.44
            + culture_resonance * 0.08
            + community_resonance * 0.1
        )
        social_self_pressure = _clamp(norm_pressure * REINTERPRETATION_WEIGHTS['social_self_norm'] + role_commitment * REINTERPRETATION_WEIGHTS['social_self_role'] + culture_present * REINTERPRETATION_WEIGHTS['social_self_culture_presence'] + community_present * REINTERPRETATION_WEIGHTS['social_self_community_presence'] + social_density * REINTERPRETATION_WEIGHTS['social_self_density'] + institutional_pressure * REINTERPRETATION_WEIGHTS['social_self_institution'] + recent_strain * REINTERPRETATION_WEIGHTS['social_self_recent_strain'] + culture_resonance * REINTERPRETATION_WEIGHTS['social_self_culture_resonance'] + community_resonance * REINTERPRETATION_WEIGHTS['social_self_community_resonance'] + transition_intensity * REINTERPRETATION_WEIGHTS['social_self_transition'] + ritual_memory * REINTERPRETATION_WEIGHTS['social_self_profile_ritual'] + institutional_memory * REINTERPRETATION_WEIGHTS['social_self_profile_institution'])
        reflective_tension = _clamp(temporal_pressure * REINTERPRETATION_WEIGHTS['reflective_temporal'] + social_self_pressure * REINTERPRETATION_WEIGHTS['reflective_social_self'] + (1.0 - trust_bias) * REINTERPRETATION_WEIGHTS['reflective_low_trust'] + (1.0 - belonging) * REINTERPRETATION_WEIGHTS['reflective_low_belonging'] + hazard_pressure * REINTERPRETATION_WEIGHTS['reflective_hazard'] + resource_pressure * REINTERPRETATION_WEIGHTS['reflective_resource'] + recent_strain * REINTERPRETATION_WEIGHTS['reflective_recent_strain'] + (1.0 - continuity_score) * REINTERPRETATION_WEIGHTS['reflective_discontinuity'] + transition_intensity * REINTERPRETATION_WEIGHTS['reflective_transition'])
        meaning_push = _clamp(
            MEANING_ALLOCATION_WEIGHTS['base']
            + reflective_tension * MEANING_ALLOCATION_WEIGHTS['reflective_push']
            + social_self_pressure * MEANING_ALLOCATION_WEIGHTS['social_push']
            + ritual_pressure * MEANING_ALLOCATION_WEIGHTS['ritual_push']
            + institutional_pressure * MEANING_ALLOCATION_WEIGHTS['institution_push']
            + transition_intensity * MEANING_ALLOCATION_WEIGHTS['transition_push']
            + ritual_memory * REINTERPRETATION_WEIGHTS['meaning_profile_ritual']
            + institutional_memory * REINTERPRETATION_WEIGHTS['meaning_profile_institution']
            + (MEANING_ALLOCATION_WEIGHTS['sim_push'] if record_kind in {'experienced_sim', 'transferred_learning'} else 0.0)
            - terrain_transition_roughness * MEANING_ALLOCATION_WEIGHTS['roughness_hold']
            - roughness_dwell * REINTERPRETATION_WEIGHTS['meaning_profile_roughness_penalty']
            - defensive_dwell * REINTERPRETATION_WEIGHTS['meaning_profile_roughness_penalty']
            - interaction_afterglow * (MEANING_ALLOCATION_WEIGHTS['check_in_hold'] if interaction_afterglow_intent == 'check_in' else MEANING_ALLOCATION_WEIGHTS['afterglow_hold'])
            - meaning_inertia * MEANING_ALLOCATION_WEIGHTS['inertia_hold']
            - relational_clarity * MEANING_ALLOCATION_WEIGHTS['clarity_hold']
            - fragility_guard * REINTERPRETATION_WEIGHTS['meaning_fragility_penalty']
            - object_avoidance * REINTERPRETATION_WEIGHTS['meaning_avoidance_penalty']
            - object_attachment * REINTERPRETATION_WEIGHTS['meaning_attachment_hold']
            - defensive_salience * REINTERPRETATION_WEIGHTS['meaning_defensive_penalty']
            - near_body_risk * REINTERPRETATION_WEIGHTS['meaning_near_body_penalty']
            + recovery_reopening * MEANING_ALLOCATION_WEIGHTS['reopening_push']
            + object_affordance_bias * 0.04
            + approach_confidence * REINTERPRETATION_WEIGHTS['meaning_approach_bonus']
        )
        meaning_hold = 1.0 - meaning_push
        meaning_shift = meaning_push
        narrative_pull = _clamp(
            belonging * REINTERPRETATION_WEIGHTS['narrative_belonging']
            + role_commitment * REINTERPRETATION_WEIGHTS['narrative_role']
            + trust_bias * REINTERPRETATION_WEIGHTS['narrative_trust']
            + community_present * REINTERPRETATION_WEIGHTS['narrative_community_presence']
            + social_grounding * REINTERPRETATION_WEIGHTS['narrative_social_grounding']
            + culture_resonance * REINTERPRETATION_WEIGHTS['narrative_culture_resonance']
            + community_resonance * REINTERPRETATION_WEIGHTS['narrative_community_resonance']
            + ritual_memory * REINTERPRETATION_WEIGHTS['narrative_profile_ritual']
            + institutional_memory * REINTERPRETATION_WEIGHTS['narrative_profile_institution']
            - recent_strain * REINTERPRETATION_WEIGHTS['narrative_recent_strain_penalty']
            - transition_intensity * REINTERPRETATION_WEIGHTS['narrative_transition_penalty']
            - terrain_transition_roughness * REINTERPRETATION_WEIGHTS['narrative_roughness_penalty']
            - roughness_dwell * REINTERPRETATION_WEIGHTS['narrative_profile_roughness_penalty']
            - defensive_dwell * REINTERPRETATION_WEIGHTS['narrative_profile_roughness_penalty']
            - interaction_afterglow * REINTERPRETATION_WEIGHTS['narrative_afterglow_penalty']
            - anticipation_tension * REINTERPRETATION_WEIGHTS['narrative_anticipation_penalty']
            - replay_intensity * REINTERPRETATION_WEIGHTS['narrative_replay_penalty']
            + recovery_reopening * REINTERPRETATION_WEIGHTS['narrative_reopening']
            + object_attachment * REINTERPRETATION_WEIGHTS['narrative_attachment']
            - fragility_guard * REINTERPRETATION_WEIGHTS['narrative_fragility_penalty']
            - defensive_salience * REINTERPRETATION_WEIGHTS['narrative_defensive_penalty']
            + approach_confidence * REINTERPRETATION_WEIGHTS['narrative_approach_bonus']
        )

        grounding_deferral = (
            terrain_transition_roughness >= REINTERPRETATION_THRESHOLDS['grounding_deferral_roughness']
            and defensive_salience >= REINTERPRETATION_THRESHOLDS['grounding_deferral_defensive']
            and narrative_pull <= REINTERPRETATION_THRESHOLDS['grounding_deferral_narrative']
            and social_grounding <= REINTERPRETATION_THRESHOLDS['grounding_deferral_grounding']
        )

        if record_kind in {'experienced_sim', 'transferred_learning'}:
            mode = 'transfer_caution'
        elif record_kind == 'verified':
            mode = 'grounded_recall'
        elif grounding_deferral:
            mode = 'grounding_deferral'
        elif community_profile_pressure >= REINTERPRETATION_THRESHOLDS['community_profile_reframing'] and transition_intensity < REINTERPRETATION_THRESHOLDS['community_transition']:
            mode = 'community_profile_reframing'
        elif interaction_afterglow_intent == 'check_in' and interaction_afterglow >= REINTERPRETATION_THRESHOLDS['check_in_afterglow']:
            mode = 'relational_check_in_reframing'
        elif anticipation_tension >= REINTERPRETATION_THRESHOLDS['stabilizing_anticipation'] and meaning_inertia >= REINTERPRETATION_THRESHOLDS['stabilizing_inertia']:
            mode = 'stabilizing_recall'
        elif recovery_reopening >= REINTERPRETATION_THRESHOLDS['reopening_recovery'] and relational_clarity >= REINTERPRETATION_THRESHOLDS['reopening_clarity'] and meaning_inertia <= REINTERPRETATION_THRESHOLDS['reopening_inertia_ceiling']:
            mode = 'reopening_reframing'
        elif transition_intensity >= REINTERPRETATION_THRESHOLDS['community_transition']:
            mode = 'community_transition_reframing'
        elif social_self_pressure >= REINTERPRETATION_THRESHOLDS['social_reframing_pressure'] or role_present:
            mode = 'social_reframing'
        elif reflective_tension >= REINTERPRETATION_THRESHOLDS['reflective_reconsolidation']:
            mode = 'reflective_reconsolidation'
        else:
            mode = 'steady_recall'

        if grounding_deferral:
            meaning_push = min(
                meaning_push,
                _clamp(
                    0.42
                    - terrain_transition_roughness * 0.08
                    - defensive_salience * 0.06
                    + approach_confidence * 0.02
                ),
            )
            meaning_hold = 1.0 - meaning_push
            meaning_shift = meaning_push

        summary = self._summary(mode=mode, record_kind=record_kind, meaning_shift=meaning_shift, social_self_pressure=social_self_pressure)
        return ReinterpretationSnapshot(
            mode=mode,
            reflective_tension=reflective_tension,
            social_self_pressure=social_self_pressure,
            meaning_shift=meaning_shift,
            meaning_push=meaning_push,
            meaning_hold=meaning_hold,
            narrative_pull=narrative_pull,
            terrain_transition_roughness=terrain_transition_roughness,
            interaction_afterglow=interaction_afterglow,
            interaction_afterglow_intent=interaction_afterglow_intent,
            recovery_reopening=recovery_reopening,
            community_profile_pressure=community_profile_pressure,
            object_affordance_bias=object_affordance_bias,
            fragility_guard=fragility_guard,
            object_attachment=object_attachment,
            object_avoidance=object_avoidance,
            reachability=reachability,
            near_body_risk=near_body_risk,
            defensive_salience=defensive_salience,
            approach_confidence=approach_confidence,
            summary=summary,
        )

    def build_reconstructed_record(
        self,
        *,
        recall_payload: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]] = None,
        relational_world: Optional[Mapping[str, Any]] = None,
        environment_pressure: Optional[Mapping[str, Any]] = None,
        transition_signal: Optional[Mapping[str, Any]] = None,
        reply_text: str = '',
        user_text: str = '',
    ) -> Optional[Dict[str, Any]]:
        if not recall_payload:
            return None
        snapshot = self.snapshot(
            recall_payload=recall_payload,
            current_state=current_state,
            relational_world=relational_world,
            environment_pressure=environment_pressure,
            transition_signal=transition_signal,
        )
        record_kind = str(recall_payload.get('record_kind') or '').strip().lower()
        meaning_shift_floor = REINTERPRETATION_WEIGHTS['reconstructed_meaning_floor'] + snapshot.terrain_transition_roughness * REINTERPRETATION_WEIGHTS['reconstructed_meaning_roughness']
        reflective_tension_floor = REINTERPRETATION_WEIGHTS['reconstructed_reflective_floor'] + snapshot.terrain_transition_roughness * REINTERPRETATION_WEIGHTS['reconstructed_reflective_roughness']
        if (
            snapshot.meaning_shift < meaning_shift_floor
            and snapshot.reflective_tension < reflective_tension_floor
            and record_kind not in {'experienced_sim', 'transferred_learning'}
        ):
            return None
        if snapshot.mode == 'grounding_deferral' and record_kind not in {'experienced_sim', 'transferred_learning', 'verified'}:
            return None
        anchor = str(recall_payload.get('memory_anchor') or recall_payload.get('summary') or '').strip()
        if not anchor:
            return None
        base_summary = str(recall_payload.get('summary') or recall_payload.get('text') or anchor).strip()
        reply_fragment = (reply_text or user_text or '').strip()[:160]
        summary = f"{snapshot.summary}: {base_summary}".strip(': ')
        text = summary if not reply_fragment else f"{summary} | turn_echo={reply_fragment}"
        anticipation_tension = _float_from(current_state, 'anticipation_tension', 0.0)
        meaning_inertia = _float_from(current_state, 'meaning_inertia', 0.0)
        tentative_bias = _clamp01(
            snapshot.terrain_transition_roughness * REINTERPRETATION_WEIGHTS['tentative_roughness']
            + snapshot.reflective_tension * REINTERPRETATION_WEIGHTS['tentative_reflective']
            + (1.0 - snapshot.narrative_pull) * REINTERPRETATION_WEIGHTS['tentative_low_narrative']
        )
        confidence = _clamp(REINTERPRETATION_WEIGHTS['confidence_base'] + snapshot.meaning_shift * REINTERPRETATION_WEIGHTS['confidence_meaning'] + snapshot.narrative_pull * REINTERPRETATION_WEIGHTS['confidence_narrative'] - tentative_bias * REINTERPRETATION_WEIGHTS['confidence_tentative_penalty'] - snapshot.interaction_afterglow * REINTERPRETATION_WEIGHTS['confidence_afterglow_penalty'] - anticipation_tension * REINTERPRETATION_WEIGHTS['confidence_anticipation_penalty'] - meaning_inertia * REINTERPRETATION_WEIGHTS['confidence_inertia_penalty'] + snapshot.recovery_reopening * REINTERPRETATION_WEIGHTS['confidence_reopening'])
        return {
            'kind': 'reconstructed',
            'summary': summary[:220],
            'text': text[:280],
            'memory_anchor': anchor[:160],
            'source_episode_id': recall_payload.get('source_episode_id') or recall_payload.get('id'),
            'policy_hint': recall_payload.get('policy_hint'),
            'confidence': confidence,
            'culture_id': recall_payload.get('culture_id') or _text(relational_world, 'culture_id'),
            'community_id': recall_payload.get('community_id') or _text(relational_world, 'community_id'),
            'social_role': recall_payload.get('social_role') or _text(relational_world, 'social_role'),
            'reinterpretation_mode': snapshot.mode,
            'reflective_tension': round(snapshot.reflective_tension, 4),
            'social_self_pressure': round(snapshot.social_self_pressure, 4),
            'meaning_shift': round(snapshot.meaning_shift, 4),
            'meaning_push': round(snapshot.meaning_push, 4),
            'meaning_hold': round(snapshot.meaning_hold, 4),
            'terrain_transition_roughness': round(snapshot.terrain_transition_roughness, 4),
            'community_profile_pressure': round(snapshot.community_profile_pressure, 4),
            'tentative_bias': round(tentative_bias, 4),
            'recovery_reopening': round(snapshot.recovery_reopening, 4),
            'object_affordance_bias': round(snapshot.object_affordance_bias, 4),
            'fragility_guard': round(snapshot.fragility_guard, 4),
            'object_attachment': round(snapshot.object_attachment, 4),
            'object_avoidance': round(snapshot.object_avoidance, 4),
            'reachability': round(snapshot.reachability, 4),
            'near_body_risk': round(snapshot.near_body_risk, 4),
            'defensive_salience': round(snapshot.defensive_salience, 4),
            'approach_confidence': round(snapshot.approach_confidence, 4),
            'reinterpretation_summary': snapshot.summary,
            'environment_summary': _text(environment_pressure, 'summary') or snapshot.summary,
            'transition_intensity': round(_float_from(transition_signal, 'transition_intensity', 0.0), 4),
            'record_provenance': 'reconstruction',
        }

    def _summary(self, *, mode: str, record_kind: str, meaning_shift: float, social_self_pressure: float) -> str:
        if mode == 'transfer_caution':
            return 'simulation lesson held as cautious guidance'
        if mode == 'grounded_recall':
            return 'verified memory stays close to lived form'
        if mode == 'grounding_deferral':
            return 'memory is kept near grounded observation until the surrounding field settles'
        if mode == 'community_profile_reframing':
            return 'memory is being read through the longer communal pattern carried by this place'
        if mode == 'community_transition_reframing':
            return 'memory is being re-read while the social world is changing'
        if mode == 'relational_check_in_reframing':
            return 'memory is being held in a relational check-in before deeper meaning is assigned'
        if mode == 'stabilizing_recall':
            return 'memory is being held steady while anticipation settles before interpretation'
        if mode == 'reopening_reframing':
            return 'memory is reopening carefully after enough stability returns'
        if mode == 'social_reframing':
            return 'memory is reframed under social and cultural pressure'
        if mode == 'reflective_reconsolidation':
            return 'memory is being reconsidered through reflective tension'
        if meaning_shift >= REINTERPRETATION_THRESHOLDS['light_reinterpretation_meaning'] or social_self_pressure >= REINTERPRETATION_THRESHOLDS['light_reinterpretation_social']:
            return 'memory is lightly reinterpreted in the current context'
        return 'memory remains close to direct recall'


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _text(mapping: Optional[Mapping[str, Any]], key: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
