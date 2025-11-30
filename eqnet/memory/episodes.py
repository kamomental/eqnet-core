"""Episode (L2) helpers for EQNet's heart OS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class Episode:
    """Serialized chunk of a day's worth of memories."""

    id: str
    date: date
    moments: List[str]
    place_ids: List[str]
    partner_ids: List[str]
    dominant_topic: str
    dominant_emotion: str
    impact: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "moments": self.moments,
            "place_ids": self.place_ids,
            "partner_ids": self.partner_ids,
            "dominant_topic": self.dominant_topic,
            "dominant_emotion": self.dominant_emotion,
            "impact": float(self.impact),
            "summary": self.summary,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Episode":
        return Episode(
            id=str(payload["id"]),
            date=date.fromisoformat(payload["date"]),
            moments=list(payload.get("moments", [])),
            place_ids=list(payload.get("place_ids", [])),
            partner_ids=list(payload.get("partner_ids", [])),
            dominant_topic=str(payload.get("dominant_topic", "misc")),
            dominant_emotion=str(payload.get("dominant_emotion", "neutral")),
            impact=float(payload.get("impact", 0.0)),
            summary=str(payload.get("summary", "")),
        )


def most_common(values: Sequence[str] | None, default: str = "misc") -> str:
    if not values:
        return default
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts, key=counts.get)


def extract_ids(payloads: Iterable[Optional[str]]) -> List[str]:
    seen: Dict[str, None] = {}
    for item in payloads:
        if not item:
            continue
        seen.setdefault(str(item), None)
    return list(seen.keys())
