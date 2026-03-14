from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


@dataclass
class EnvironmentPressureSnapshot:
    resource_pressure: float = 0.0
    hazard_pressure: float = 0.0
    ritual_pressure: float = 0.0
    institutional_pressure: float = 0.0
    social_density: float = 0.0
    summary: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvironmentPressureCore:
    """Estimate how environment and institutions bias development and recall."""

    def snapshot(
        self,
        *,
        relational_world: Optional[Mapping[str, Any]] = None,
        sensor_input: Optional[Mapping[str, Any]] = None,
        current_state: Optional[Mapping[str, Any]] = None,
    ) -> EnvironmentPressureSnapshot:
        world = relational_world or {}
        sensor = sensor_input or {}
        state = current_state or {}

        world_type = str(world.get('world_type') or '').strip().lower()
        time_phase = str(world.get('time_phase') or '').strip().lower()
        weather = str(world.get('weather') or '').strip().lower()
        zone_id = str(world.get('zone_id') or '').strip().lower()
        mode = str(world.get('mode') or state.get('mode') or 'reality').strip().lower()

        scarcity = _float_from(world, 'resource_scarcity', _float_from(sensor, 'resource_scarcity', 0.0))
        hazard = _float_from(world, 'hazard_level', _float_from(sensor, 'hazard_level', 0.0))
        ritual = _float_from(world, 'ritual_signal', _float_from(sensor, 'ritual_signal', 0.0))
        institutional = _float_from(world, 'institutional_pressure', _float_from(sensor, 'institutional_pressure', 0.0))
        person_count = max(int(sensor.get('person_count', state.get('person_count', 0)) or 0), 0)
        motion_score = _float_from(sensor, 'motion_score', 0.0)
        body_stress = _float_from(sensor, 'body_stress_index', _float_from(state, 'stress', 0.0))

        resource_pressure = _clamp(scarcity * 0.65 + (0.08 if time_phase in {'night', 'dusk'} else 0.0) + (0.06 if zone_id in {'market', 'harbor'} else 0.0))
        hazard_pressure = _clamp(hazard * 0.7 + body_stress * 0.18 + motion_score * 0.12 + (0.08 if weather in {'storm', 'rain'} else 0.0))
        ritual_pressure = _clamp(ritual * 0.72 + (0.12 if zone_id in {'shrine', 'memorial', 'ceremony'} else 0.0) + (0.06 if mode == 'streaming' else 0.0))
        institutional_pressure = _clamp(institutional * 0.72 + (0.10 if world_type in {'institutional', 'civic'} else 0.0) + (0.05 if mode == 'simulation' else 0.0))
        social_density = _clamp(min(person_count, 6) / 6.0 * 0.72 + motion_score * 0.2 + (0.08 if mode == 'streaming' else 0.0))

        summary = self._summary(
            resource_pressure=resource_pressure,
            hazard_pressure=hazard_pressure,
            ritual_pressure=ritual_pressure,
            institutional_pressure=institutional_pressure,
            social_density=social_density,
        )
        return EnvironmentPressureSnapshot(
            resource_pressure=resource_pressure,
            hazard_pressure=hazard_pressure,
            ritual_pressure=ritual_pressure,
            institutional_pressure=institutional_pressure,
            social_density=social_density,
            summary=summary,
        )

    def _summary(
        self,
        *,
        resource_pressure: float,
        hazard_pressure: float,
        ritual_pressure: float,
        institutional_pressure: float,
        social_density: float,
    ) -> str:
        if hazard_pressure >= 0.5:
            return 'environment feels risky and narrows action'
        if ritual_pressure >= 0.45 or institutional_pressure >= 0.5:
            return 'environment carries shared rules and formal expectations'
        if resource_pressure >= 0.45:
            return 'environment feels scarce and increases caution'
        if social_density >= 0.45:
            return 'environment feels socially dense and self-monitoring rises'
        return 'environment pressure stays moderate'


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
