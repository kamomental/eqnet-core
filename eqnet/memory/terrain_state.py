"""Terrain state helpers for nightly promotion / retrieval."""

from __future__ import annotations

from typing import List, Optional

from .episodes import Episode
from .monuments import Monument


class TerrainState:
    """Lightweight container tracking episodes/monuments."""

    def __init__(
        self,
        *,
        episodes: Optional[List[Episode]] = None,
        monuments: Optional[List[Monument]] = None,
    ) -> None:
        self.episodes: List[Episode] = episodes or []
        self.monuments: List[Monument] = monuments or []

    # --- registration -------------------------------------------------

    def register_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def register_monument(self, monument: Monument) -> None:
        self.monuments.append(monument)

    # --- sensitivity / novelty hooks (very rough) ---------------------

    def similarity_to_existing_themes(self, topic: str | None) -> float:
        if not topic:
            return 0.0
        matches = sum(1 for ep in self.episodes if ep.dominant_topic == topic)
        total = max(1, len(self.episodes))
        return min(1.0, matches / total)

    def sensitivity_for_topic(self, topic: str | None) -> float:
        if not topic:
            return 0.5
        important = sum(1 for mon in self.monuments if mon.culture_tag == topic)
        total = max(1, len(self.monuments))
        return min(1.0, 0.3 + 0.7 * important / total)

    # --- monument lookup ---------------------------------------------

    def should_create_monument(self, moment: object) -> bool:
        place_id = getattr(moment, "place_id", None) or ""
        culture_tag = getattr(moment, "culture_tag", getattr(moment, "topic", ""))
        for mon in self.monuments:
            if mon.place_id == place_id and mon.culture_tag == culture_tag:
                return False
        return True

    def find_monuments_by_place(self, place_id: str, k: int = 3) -> List[Monument]:
        matches = [m for m in self.monuments if m.place_id == place_id]
        matches.sort(key=lambda m: m.importance, reverse=True)
        return matches[: max(1, k)]
