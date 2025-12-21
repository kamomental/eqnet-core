from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SceneAgent:
    """Agent participating in a reconstructed memory scene."""

    name: str
    role: str = "observer"
    perspective: str = "observer"
    certainty: float = 0.5
    traits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["certainty"] = float(max(0.0, min(1.0, self.certainty)))
        return payload


@dataclass
class SceneObject:
    """Object or place highlighted inside a reconstructed scene."""

    name: str
    description: str = ""
    salience: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["salience"] = float(max(0.0, min(1.0, self.salience)))
        return payload


@dataclass
class SceneConstraint:
    """Constraint or affordance that shaped the remembered action."""

    label: str
    intensity: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["intensity"] = float(max(0.0, min(1.0, self.intensity)))
        return payload


@dataclass
class AffectSnapshot:
    """Momentary affective image (e.g., father's stern face)."""

    label: str
    intensity: float
    emotion_tag: Optional[str] = None
    replay_source: str = "live"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["intensity"] = float(max(0.0, min(1.0, self.intensity)))
        return payload


@dataclass
class SceneFrame:
    """Derived artefact bundling agents/constraints for UI or dreams."""

    scene_id: str
    anchor: Optional[str]
    agents: List[SceneAgent]
    objects: List[SceneObject] = field(default_factory=list)
    constraints: List[SceneConstraint] = field(default_factory=list)
    norm_event: Optional[str] = None
    affect_snapshots: List[AffectSnapshot] = field(default_factory=list)
    replay_source: str = "derived"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "anchor": self.anchor,
            "agents": [agent.to_dict() for agent in self.agents],
            "objects": [obj.to_dict() for obj in self.objects],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "norm_event": self.norm_event,
            "affect_snapshots": [snap.to_dict() for snap in self.affect_snapshots],
            "replay_source": self.replay_source,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "SceneFrame":
        agents = [SceneAgent(**agent) for agent in payload.get("agents", [])]
        objects = [SceneObject(**obj) for obj in payload.get("objects", [])]
        constraints = [SceneConstraint(**item) for item in payload.get("constraints", [])]
        snapshots = [AffectSnapshot(**snap) for snap in payload.get("affect_snapshots", [])]
        return SceneFrame(
            scene_id=str(payload.get("scene_id", "scene")),
            anchor=payload.get("anchor"),
            agents=agents,
            objects=objects,
            constraints=constraints,
            norm_event=payload.get("norm_event"),
            affect_snapshots=snapshots,
            replay_source=str(payload.get("replay_source", "derived")),
        )


__all__ = [
    "SceneAgent",
    "SceneObject",
    "SceneConstraint",
    "AffectSnapshot",
    "SceneFrame",
]
