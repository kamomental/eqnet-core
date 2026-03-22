from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


FIELD_ESTIMATOR_WEIGHTS = {
    "level_observed": 0.6,
    "level_previous": 0.4,
    "momentum_velocity": 0.5,
    "momentum_previous": 0.5,
    "dwell_increment": 0.25,
    "dwell_decay": 0.15,
}

FIELD_ESTIMATOR_THRESHOLDS = {
    "roughness_active": 0.24,
    "defensive_active": 0.22,
}


@dataclass
class FieldEstimateSnapshot:
    roughness_level: float = 0.0
    roughness_velocity: float = 0.0
    roughness_momentum: float = 0.0
    roughness_dwell: float = 0.0
    defensive_level: float = 0.0
    defensive_velocity: float = 0.0
    defensive_momentum: float = 0.0
    defensive_dwell: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FieldEstimatorCore:
    """Lift instantaneous observables into a small time-aware latent state."""

    def snapshot(
        self,
        *,
        current_state: Mapping[str, Any] | None = None,
        observed_roughness: float = 0.0,
        observed_defensive_salience: float = 0.0,
    ) -> FieldEstimateSnapshot:
        state = dict(current_state or {})
        roughness = self._estimate_channel(
            previous=state,
            prefix="roughness",
            observed=_clamp01(observed_roughness),
            active_threshold=FIELD_ESTIMATOR_THRESHOLDS["roughness_active"],
        )
        defensive = self._estimate_channel(
            previous=state,
            prefix="defensive",
            observed=_clamp01(observed_defensive_salience),
            active_threshold=FIELD_ESTIMATOR_THRESHOLDS["defensive_active"],
        )
        return FieldEstimateSnapshot(
            roughness_level=roughness["level"],
            roughness_velocity=roughness["velocity"],
            roughness_momentum=roughness["momentum"],
            roughness_dwell=roughness["dwell"],
            defensive_level=defensive["level"],
            defensive_velocity=defensive["velocity"],
            defensive_momentum=defensive["momentum"],
            defensive_dwell=defensive["dwell"],
        )

    def _estimate_channel(
        self,
        *,
        previous: Mapping[str, Any],
        prefix: str,
        observed: float,
        active_threshold: float,
    ) -> Dict[str, float]:
        previous_level = _float_from(previous, f"{prefix}_level", _float_from(previous, f"{prefix}_salience", 0.0))
        previous_momentum = _float_from(previous, f"{prefix}_momentum", 0.0)
        previous_dwell = _float_from(previous, f"{prefix}_dwell", 0.0)

        level = _clamp01(
            observed * FIELD_ESTIMATOR_WEIGHTS["level_observed"]
            + previous_level * FIELD_ESTIMATOR_WEIGHTS["level_previous"]
        )
        velocity = _clamp_signed(observed - previous_level)
        momentum = _clamp_signed(
            velocity * FIELD_ESTIMATOR_WEIGHTS["momentum_velocity"]
            + previous_momentum * FIELD_ESTIMATOR_WEIGHTS["momentum_previous"]
        )
        if level >= active_threshold:
            dwell = _clamp01(previous_dwell + FIELD_ESTIMATOR_WEIGHTS["dwell_increment"])
        else:
            dwell = _clamp01(previous_dwell - FIELD_ESTIMATOR_WEIGHTS["dwell_decay"])
        return {
            "level": round(level, 4),
            "velocity": round(velocity, 4),
            "momentum": round(momentum, 4),
            "dwell": round(dwell, 4),
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _float_from(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default
