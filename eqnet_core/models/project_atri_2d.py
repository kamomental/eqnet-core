from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class ProjectATRI2DState:
    schema: str
    timestamp: str
    identity: Dict[str, Any]
    world: Dict[str, Any]
    body: Dict[str, Any]
    sensing: Dict[str, Any]
    activity: Dict[str, Any]
    social: Dict[str, Any]
    memory: Dict[str, Any]
    simulation: Dict[str, Any]
    last_event: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": self.schema,
            "timestamp": self.timestamp,
            "identity": dict(self.identity),
            "world": dict(self.world),
            "body": dict(self.body),
            "sensing": dict(self.sensing),
            "activity": dict(self.activity),
            "social": dict(self.social),
            "memory": dict(self.memory),
            "simulation": dict(self.simulation),
        }
        if self.last_event is not None:
            payload["last_event"] = dict(self.last_event)
        return payload


@dataclass
class ProjectATRI2DEvent:
    schema: str
    timestamp: str
    source: str
    event_type: str
    world_id: Optional[str]
    entity_id: Optional[str]
    payload: Dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProjectATRI2DEvent":
        event_type = str(raw.get("event_type") or "").strip()
        if not event_type:
            raise ValueError("event_type_required")
        payload = raw.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("payload_must_be_mapping")
        return cls(
            schema=str(raw.get("schema") or "project_atri_2d_event/v1"),
            timestamp=str(raw.get("timestamp") or ""),
            source=str(raw.get("source") or "2d_world"),
            event_type=event_type,
            world_id=_optional_text(raw.get("world_id")),
            entity_id=_optional_text(raw.get("entity_id")),
            payload=dict(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "timestamp": self.timestamp,
            "source": self.source,
            "event_type": self.event_type,
            "world_id": self.world_id,
            "entity_id": self.entity_id,
            "payload": dict(self.payload),
        }


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
