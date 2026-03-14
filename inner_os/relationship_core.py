from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


@dataclass
class RelationshipState:
    attachment: float = 0.42
    trust_memory: float = 0.45
    familiarity: float = 0.35
    role_alignment: float = 0.4
    rupture_sensitivity: float = 0.38

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RelationshipCore:
    """Track slow-changing interpersonal sediment inside the OS."""

    def snapshot(
        self,
        *,
        relational_world: Mapping[str, Any],
        sensor_input: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]] = None,
    ) -> RelationshipState:
        prev_attachment = _float_from(current_state, 'attachment', 0.42)
        prev_trust = _float_from(current_state, 'trust_memory', 0.45)
        prev_familiarity = _float_from(current_state, 'familiarity', 0.35)
        prev_role = _float_from(current_state, 'role_alignment', 0.4)
        prev_rupture = _float_from(current_state, 'rupture_sensitivity', 0.38)

        community_present = 1.0 if relational_world.get('community_id') else 0.0
        role_present = 1.0 if relational_world.get('social_role') else 0.0
        voice_level = _float_from(sensor_input, 'voice_level', 0.0)
        person_count = max(int(sensor_input.get('person_count', 0) or 0), 0)
        body_stress = _float_from(sensor_input, 'body_stress_index', 0.0)
        private_flag = 1.0 if str(sensor_input.get('body_state_flag') or '') == 'private_high_arousal' else 0.0

        attachment = _clamp(prev_attachment * 0.76 + 0.08 * community_present + 0.06 * min(voice_level, 1.0) + 0.04 * min(person_count, 2) - 0.03 * body_stress)
        trust_memory = _clamp(prev_trust * 0.78 + 0.08 * community_present + 0.05 * role_present - 0.04 * body_stress)
        familiarity = _clamp(prev_familiarity * 0.82 + 0.08 * community_present + 0.05 * min(person_count, 2) + 0.04 * min(voice_level, 1.0))
        role_alignment = _clamp(prev_role * 0.8 + 0.1 * role_present + 0.05 * community_present)
        rupture_sensitivity = _clamp(prev_rupture * 0.8 + 0.08 * body_stress + 0.05 * private_flag + 0.03 * max(0.0, 1.0 - trust_memory))

        return RelationshipState(
            attachment=attachment,
            trust_memory=trust_memory,
            familiarity=familiarity,
            role_alignment=role_alignment,
            rupture_sensitivity=rupture_sensitivity,
        )

    def post_turn(
        self,
        *,
        previous: Mapping[str, Any],
        relational_world: Mapping[str, Any],
        reply_present: bool,
        stress: float,
        recovery_need: float,
    ) -> RelationshipState:
        attachment = _float_from(previous, 'attachment', 0.42)
        trust_memory = _float_from(previous, 'trust_memory', 0.45)
        familiarity = _float_from(previous, 'familiarity', 0.35)
        role_alignment = _float_from(previous, 'role_alignment', 0.4)
        rupture_sensitivity = _float_from(previous, 'rupture_sensitivity', 0.38)

        success = 1.0 if reply_present else 0.0
        community_present = 1.0 if relational_world.get('community_id') else 0.0
        role_present = 1.0 if relational_world.get('social_role') else 0.0
        strain = max(stress, recovery_need)

        attachment = _clamp(attachment * 0.84 + 0.05 * success + 0.04 * community_present - 0.03 * strain)
        trust_memory = _clamp(trust_memory * 0.84 + 0.06 * success - 0.04 * strain)
        familiarity = _clamp(familiarity * 0.87 + 0.05 * success + 0.04 * community_present)
        role_alignment = _clamp(role_alignment * 0.85 + 0.05 * role_present + 0.03 * success)
        rupture_sensitivity = _clamp(rupture_sensitivity * 0.86 + 0.04 * strain - 0.03 * success)

        return RelationshipState(
            attachment=attachment,
            trust_memory=trust_memory,
            familiarity=familiarity,
            role_alignment=role_alignment,
            rupture_sensitivity=rupture_sensitivity,
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
