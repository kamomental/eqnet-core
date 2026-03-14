from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping


PERIPERSONAL_WEIGHTS = {
    "nearby_objects": 0.18,
    "private_mode_risk": 0.16,
    "person_density_risk": 0.14,
    "hazard_risk": 0.18,
    "fragility_risk": 0.12,
    "avoidance_risk": 0.18,
    "reachability_base": 0.24,
    "tool_extension": 0.18,
    "affordance": 0.14,
    "defensive_from_risk": 0.52,
    "defensive_from_private": 0.16,
    "approach_from_reachability": 0.34,
    "approach_from_affordance": 0.24,
    "approach_penalty": 0.28,
}


@dataclass
class PeripersonalSnapshot:
    reachability: float = 0.0
    near_body_risk: float = 0.0
    defensive_salience: float = 0.0
    approach_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PeripersonalCore:
    def snapshot(
        self,
        *,
        relational_world: Mapping[str, Any] | None = None,
        sensor_input: Mapping[str, Any] | None = None,
        current_state: Mapping[str, Any] | None = None,
        object_relation: Mapping[str, Any] | None = None,
    ) -> PeripersonalSnapshot:
        world = dict(relational_world or {})
        sensor = dict(sensor_input or {})
        state = dict(current_state or {})
        obj = dict(object_relation or {})

        nearby_objects = list(world.get("nearby_objects") or [])
        person_count = max(int(sensor.get("person_count", state.get("person_count", 0)) or 0), 0)
        privacy_tags = [str(tag).lower() for tag in (sensor.get("privacy_tags") or state.get("privacy_tags") or [])]
        private_mode = 1.0 if ("private" in privacy_tags or str(sensor.get("body_state_flag") or state.get("body_state_flag") or "") == "private_high_arousal") else 0.0
        hazard_level = _float_from(world, "hazard_level", 0.0)
        fragility_guard = _float_from(obj, "fragility_guard", _float_from(state, "fragility_guard", 0.0))
        object_avoidance = _float_from(obj, "object_avoidance", _float_from(state, "object_avoidance", 0.0))
        tool_extension_bias = _float_from(obj, "tool_extension_bias", _float_from(state, "tool_extension_bias", 0.0))
        object_affordance_bias = _float_from(obj, "object_affordance_bias", _float_from(state, "object_affordance_bias", 0.0))

        reachability = _clamp01(
            min(len(nearby_objects), 3) * PERIPERSONAL_WEIGHTS["reachability_base"]
            + tool_extension_bias * PERIPERSONAL_WEIGHTS["tool_extension"]
            + object_affordance_bias * PERIPERSONAL_WEIGHTS["affordance"]
        )
        near_body_risk = _clamp01(
            min(len(nearby_objects), 3) * PERIPERSONAL_WEIGHTS["nearby_objects"]
            + private_mode * PERIPERSONAL_WEIGHTS["private_mode_risk"]
            + min(person_count, 3) / 3.0 * PERIPERSONAL_WEIGHTS["person_density_risk"]
            + hazard_level * PERIPERSONAL_WEIGHTS["hazard_risk"]
            + fragility_guard * PERIPERSONAL_WEIGHTS["fragility_risk"]
            + object_avoidance * PERIPERSONAL_WEIGHTS["avoidance_risk"]
        )
        defensive_salience = _clamp01(
            near_body_risk * PERIPERSONAL_WEIGHTS["defensive_from_risk"]
            + private_mode * PERIPERSONAL_WEIGHTS["defensive_from_private"]
        )
        approach_confidence = _clamp01(
            reachability * PERIPERSONAL_WEIGHTS["approach_from_reachability"]
            + object_affordance_bias * PERIPERSONAL_WEIGHTS["approach_from_affordance"]
            - defensive_salience * PERIPERSONAL_WEIGHTS["approach_penalty"]
        )
        return PeripersonalSnapshot(
            reachability=round(reachability, 4),
            near_body_risk=round(near_body_risk, 4),
            defensive_salience=round(defensive_salience, 4),
            approach_confidence=round(approach_confidence, 4),
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_from(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default
