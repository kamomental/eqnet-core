from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldState:
    scene_graph: dict[str, list[str]] = field(default_factory=dict)
    spatial_map: dict[str, str] = field(default_factory=dict)
    object_states: dict[str, str] = field(default_factory=dict)
    task_states: dict[str, str] = field(default_factory=dict)
    social_relation_graph: dict[str, list[str]] = field(default_factory=dict)
    uncertainty: float = 1.0
