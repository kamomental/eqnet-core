from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ConsciousAccessSnapshot:
    talk_mode: str
    route: str
    surface_state: str
    intent: str
    recall_active: bool
    replay_active: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsciousAccessCore:
    """Small helper for mapping inner access state to outward surface labels."""

    def snapshot(
        self,
        *,
        talk_mode: str,
        route: str,
        mode: str,
        memory_anchor: Optional[str],
        replay_active: bool,
        body_state_flag: str = "normal",
        voice_level: float = 0.0,
        person_count: int = 0,
        autonomic_balance: float = 0.5,
    ) -> ConsciousAccessSnapshot:
        base_map = {
            "watch": "watch",
            "ask": "attend",
            "talk": "talk",
            "soothe": "sync",
            "presence": "rest",
        }
        surface_state = base_map.get(talk_mode, "idle")
        if mode == "streaming":
            surface_state = "stream"
        elif mode == "simulation":
            surface_state = "simulate"
        elif body_state_flag == "private_high_arousal":
            surface_state = "rest"
        elif replay_active:
            surface_state = "replay"
        elif memory_anchor:
            surface_state = "recall"
        elif voice_level >= 0.45 and person_count > 0:
            surface_state = "talk"

        if body_state_flag == "private_high_arousal":
            intent = "soften"
        elif autonomic_balance < 0.42:
            intent = "guard"
        elif route == "conscious":
            intent = "engage"
        elif route == "reflex":
            intent = "guard"
        elif memory_anchor:
            intent = "remember"
        elif voice_level >= 0.45 and person_count > 0:
            intent = "answer"
        else:
            intent = "listen"

        return ConsciousAccessSnapshot(
            talk_mode=talk_mode,
            route=route,
            surface_state=surface_state,
            intent=intent,
            recall_active=bool(memory_anchor),
            replay_active=bool(replay_active),
        )

