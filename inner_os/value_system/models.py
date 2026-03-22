from __future__ import annotations

from dataclasses import dataclass, field

from .emotional_dft import TerrainSnapshot


@dataclass
class ValueState:
    danger_score: float = 0.0
    trust_preservation_score: float = 0.0
    overload_risk: float = 0.0
    exploration_drive: float = 0.0
    repair_urgency: float = 0.0
    value_axes: dict[str, float] = field(default_factory=dict)
    terrain_snapshot: TerrainSnapshot | None = None
    terrain_energy: float = 0.0
