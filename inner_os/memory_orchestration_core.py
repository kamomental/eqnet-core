from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

from .monument_query_adapter import MonumentQueryAdapter


MonumentQuery = Callable[[Mapping[str, Any]], Tuple[float, Optional[str]]]


@dataclass(frozen=True)
class MemoryOrchestrationSnapshot:
    monument_salience: float = 0.0
    monument_kind: Optional[str] = None
    conscious_mosaic_density: float = 0.0
    conscious_mosaic_recentness: float = 0.0
    reuse_trajectory: float = 0.0
    interference_pressure: float = 0.0
    consolidation_priority: float = 0.0
    prospective_memory_pull: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "monument_salience": round(self.monument_salience, 4),
            "monument_kind": self.monument_kind,
            "conscious_mosaic_density": round(self.conscious_mosaic_density, 4),
            "conscious_mosaic_recentness": round(self.conscious_mosaic_recentness, 4),
            "reuse_trajectory": round(self.reuse_trajectory, 4),
            "interference_pressure": round(self.interference_pressure, 4),
            "consolidation_priority": round(self.consolidation_priority, 4),
            "prospective_memory_pull": round(self.prospective_memory_pull, 4),
        }


class MemoryOrchestrationCore:
    def __init__(
        self,
        *,
        monument_query: Optional[MonumentQuery] = None,
        monument_adapter: Optional[MonumentQueryAdapter] = None,
    ) -> None:
        adapter = monument_adapter or MonumentQueryAdapter()
        self._monument_query = monument_query or adapter.query

    def snapshot(
        self,
        *,
        relational_world: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]],
        forgetting_snapshot: Optional[Mapping[str, Any]] = None,
        recall_active: bool = False,
    ) -> MemoryOrchestrationSnapshot:
        monument_salience, monument_kind = self._monument_query(relational_world)
        conscious_mosaic_density = _float_from(current_state, "conscious_mosaic_density", default=0.0)
        conscious_mosaic_recentness = _float_from(current_state, "conscious_mosaic_recentness", default=0.0)
        replay_intensity = _float_from(current_state, "replay_intensity", default=0.0)
        anticipation_tension = _float_from(current_state, "anticipation_tension", default=0.0)
        future_signal = _float_from(current_state, "future_signal", default=0.0)
        terrain_transition_roughness = _float_from(current_state, "terrain_transition_roughness", default=0.0)
        forgetting_pressure = _float_from(forgetting_snapshot, "forgetting_pressure", default=0.0)

        reuse_trajectory = _clamp01(
            conscious_mosaic_recentness * 0.28
            + conscious_mosaic_density * 0.16
            + monument_salience * 0.14
            + replay_intensity * 0.22
            + (0.12 if recall_active else 0.0)
        )
        interference_pressure = _clamp01(
            forgetting_pressure * 0.34
            + terrain_transition_roughness * 0.18
            + conscious_mosaic_density * 0.14
            + max(0.0, 1.0 - conscious_mosaic_recentness) * 0.08
        )
        consolidation_priority = _clamp01(
            monument_salience * 0.28
            + conscious_mosaic_density * 0.14
            + replay_intensity * 0.16
            + max(0.0, 1.0 - interference_pressure) * 0.14
            + (0.08 if recall_active else 0.0)
        )
        prospective_memory_pull = _clamp01(
            future_signal * 0.42
            + anticipation_tension * 0.24
            + monument_salience * 0.06
        )
        return MemoryOrchestrationSnapshot(
            monument_salience=monument_salience,
            monument_kind=monument_kind,
            conscious_mosaic_density=conscious_mosaic_density,
            conscious_mosaic_recentness=conscious_mosaic_recentness,
            reuse_trajectory=reuse_trajectory,
            interference_pressure=interference_pressure,
            consolidation_priority=consolidation_priority,
            prospective_memory_pull=prospective_memory_pull,
        )

def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, *, default: float = 0.0) -> float:
    if not mapping:
        return default
    try:
        return float(mapping.get(key, default) or 0.0)
    except (TypeError, ValueError, AttributeError):
        return default
