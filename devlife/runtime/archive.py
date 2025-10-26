"""Simple archive writing developmental episodes to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class EpisodeArchive:
    base_dir: Path = Path("logs/episodes")
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 10_000

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, episode: Dict[str, Any]) -> None:
        self.buffer.append(episode)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]
        path = self._episode_path(episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(episode, fh, ensure_ascii=False, indent=2)

    def sample_episodes(self, query: Dict[str, Any] | None = None, limit: int = 5) -> Sequence[Dict[str, Any]]:
        if not query:
            return self.buffer[-limit:]
        filtered = [ep for ep in self.buffer if all(ep.get(k) == v for k, v in query.items())]
        return filtered[-limit:]

    def _episode_path(self, episode: Dict[str, Any]) -> Path:
        stage = episode.get("stage", "unknown")
        timestamp = episode.get("timestamp", "na")
        fname = f"{timestamp.replace(':', '').replace('-', '')}_{stage}.json"
        return self.base_dir / stage / fname
