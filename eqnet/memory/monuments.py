"""Monument (L3) helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(slots=True)
class Monument:
    """Long-term semantic anchor (place × people × emotion)."""

    id: str
    place_id: str
    partner_ids: List[str]
    culture_tag: str
    core_emotion: str
    importance: float
    summary: str
    episodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "place_id": self.place_id,
            "partner_ids": self.partner_ids,
            "culture_tag": self.culture_tag,
            "core_emotion": self.core_emotion,
            "importance": float(self.importance),
            "summary": self.summary,
            "episodes": self.episodes,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Monument":
        return Monument(
            id=str(payload["id"]),
            place_id=str(payload.get("place_id", "unknown")),
            partner_ids=list(payload.get("partner_ids", [])),
            culture_tag=str(payload.get("culture_tag", "misc")),
            core_emotion=str(payload.get("core_emotion", "neutral")),
            importance=float(payload.get("importance", 0.0)),
            summary=str(payload.get("summary", "")),
            episodes=list(payload.get("episodes", [])),
        )
