from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


@dataclass
class RelationalWorldState:
    mode: str = "reality"
    world_id: str = "harbor_town"
    world_type: str = "infrastructure"
    zone_id: str = "market"
    time_phase: str = "day"
    weather: str = "clear"
    simulation_enabled: bool = False
    simulation_episode_id: Optional[str] = None
    simulation_transfer_pending: bool = False
    world_source: str = "runtime"
    culture_id: str = "default"
    community_id: str = "local"
    social_role: str = "companion"
    person_id: Optional[str] = None
    resource_scarcity: float = 0.0
    hazard_level: float = 0.0
    ritual_signal: float = 0.0
    institutional_pressure: float = 0.0
    place_memory_anchor: Optional[str] = None
    nearby_objects: list[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RelationalWorldCore:
    """Keeps the lifeform situated in world / place / mode without owning UI."""

    def __init__(self, state: Optional[RelationalWorldState] = None) -> None:
        self.state = state or RelationalWorldState()

    def snapshot(self) -> Dict[str, Any]:
        payload = self.state.to_dict()
        if payload.get("nearby_objects") is None:
            payload["nearby_objects"] = []
        return payload

    def absorb_context(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        for key in ("culture_id", "community_id", "social_role", "place_memory_anchor", "person_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                setattr(self.state, key, value.strip())
        for key in ("resource_scarcity", "hazard_level", "ritual_signal", "institutional_pressure"):
            value = payload.get(key)
            try:
                setattr(self.state, key, max(0.0, min(1.0, float(value))))
            except (TypeError, ValueError):
                pass
        nearby_objects = payload.get("nearby_objects") or []
        if isinstance(nearby_objects, list):
            self.state.nearby_objects = [str(item).strip() for item in nearby_objects if str(item).strip()][:8]
        return self.snapshot()

    def mode(self) -> str:
        mode = str(self.state.mode or "reality").strip().lower()
        if mode not in {"reality", "streaming", "simulation"}:
            return "reality"
        return mode

    def ingest_surface_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        world_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if world_id:
            self.state.world_id = str(world_id)
        self.absorb_context(payload)
        zone_id = payload.get("zone_id")
        if isinstance(zone_id, str) and zone_id.strip():
            self.state.zone_id = zone_id.strip()
        world_type = payload.get("world_type")
        if isinstance(world_type, str) and world_type.strip():
            self.state.world_type = world_type.strip()
        time_phase = payload.get("time_phase")
        if isinstance(time_phase, str) and time_phase.strip():
            self.state.time_phase = time_phase.strip()
        weather = payload.get("weather")
        if isinstance(weather, str) and weather.strip():
            self.state.weather = weather.strip()

        if event_type == "stream_stage_enter":
            self.state.mode = "streaming"
        elif event_type == "stream_stage_exit":
            self.state.mode = "reality"
        elif event_type == "sim_episode_start":
            self.state.mode = "simulation"
            self.state.simulation_enabled = True
            self.state.simulation_episode_id = _text_or_none(
                payload.get("episode_id") or payload.get("sim_episode_id")
            )
            self.state.world_source = str(payload.get("world_source") or "mini_world")
        elif event_type == "sim_episode_end":
            self.state.mode = "reality"
            self.state.simulation_enabled = False
            self.state.simulation_transfer_pending = bool(payload.get("transfer_pending", False))
        elif event_type == "sim_transfer_candidate":
            self.state.simulation_transfer_pending = True
        elif event_type == "rest_enter":
            self.state.zone_id = str(payload.get("zone_id") or self.state.zone_id or "rest_place")
        elif event_type == "rest_exit" and self.state.zone_id == "rest_place":
            self.state.zone_id = "market"

        return self.snapshot()


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
