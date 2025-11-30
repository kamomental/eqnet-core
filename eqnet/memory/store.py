"""Simple JSONL memory store for episodes and monuments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from .episodes import Episode
from .monuments import Monument


class MemoryStore:
    """Minimal persistence helper."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._episode_path = self.base_dir / "episodes.jsonl"
        self._monument_path = self.base_dir / "monuments.jsonl"

    def save_episode(self, episode: Episode) -> None:
        self._append_jsonl(self._episode_path, episode.to_dict())

    def save_monument(self, monument: Monument) -> None:
        self._append_jsonl(self._monument_path, monument.to_dict())

    def load_all(self) -> Tuple[List[Episode], List[Monument]]:
        return self.load_episodes(), self.load_monuments()

    def load_episodes(self) -> List[Episode]:
        return [Episode.from_dict(data) for data in self._iter_jsonl(self._episode_path)]

    def load_monuments(self) -> List[Monument]:
        return [Monument.from_dict(data) for data in self._iter_jsonl(self._monument_path)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict]:
        if not path.exists():
            return []
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
