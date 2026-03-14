from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np


@dataclass
class TerrainSnapshot:
    valence: float
    arousal: float
    stress: float
    temporal_pressure: float
    danger_slope: float
    recovery_basin: float
    ignition_potential: float
    transition_roughness: float
    attractor: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AffectiveTerrainCore:
    """Minimal emotional terrain surface for reusable inner-life state."""

    def snapshot(
        self,
        *,
        valence: float,
        arousal: float,
        stress: float,
        temporal_pressure: float,
        memory_ignition: float = 0.0,
        transition_intensity: float = 0.0,
        social_grounding: float = 0.44,
        community_resonance: float = 0.0,
    ) -> TerrainSnapshot:
        val = float(np.clip(valence, -1.0, 1.0))
        aro = float(np.clip(arousal, 0.0, 1.0))
        stress_clamped = float(np.clip(stress, 0.0, 1.0))
        pressure = float(np.clip(temporal_pressure, 0.0, 1.0))
        ignition = float(np.clip(memory_ignition, 0.0, 1.0))
        transition = float(np.clip(transition_intensity, 0.0, 1.0))
        grounding = float(np.clip(social_grounding, 0.0, 1.0))
        resonance = float(np.clip(community_resonance, 0.0, 1.0))

        transition_roughness = float(
            np.clip(
                transition * 0.72
                + max(0.0, 0.5 - grounding) * 0.18
                + max(0.0, 0.45 - resonance) * 0.1,
                0.0,
                1.0,
            )
        )
        danger_slope = float(np.clip(0.6 * stress_clamped + 0.4 * max(aro - val, 0.0) + transition_roughness * 0.16, 0.0, 1.0))
        recovery_basin = float(np.clip((1.0 - stress_clamped) * (0.55 + 0.45 * max(val, 0.0)) - transition_roughness * 0.14, 0.0, 1.0))
        ignition_potential = float(np.clip(0.55 * ignition + 0.45 * pressure, 0.0, 1.0))
        attractor = self._attractor(val=val, aro=aro, stress=stress_clamped, pressure=pressure, transition_roughness=transition_roughness)
        return TerrainSnapshot(
            valence=val,
            arousal=aro,
            stress=stress_clamped,
            temporal_pressure=pressure,
            danger_slope=danger_slope,
            recovery_basin=recovery_basin,
            ignition_potential=ignition_potential,
            transition_roughness=transition_roughness,
            attractor=attractor,
        )

    def _attractor(self, *, val: float, aro: float, stress: float, pressure: float, transition_roughness: float) -> str:
        if transition_roughness >= 0.45 and stress < 0.72 and pressure < 0.78:
            return "unfamiliar_slope"
        if stress >= 0.72 or pressure >= 0.78:
            return "guarded_edge"
        if val >= 0.35 and aro <= 0.42:
            return "warm_rest"
        if val >= 0.2 and aro > 0.42:
            return "curious_approach"
        if val <= -0.25 and aro >= 0.45:
            return "pain_spike"
        if aro <= 0.2:
            return "low_tide"
        return "watchful_plain"
