from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from eqnet_core.models.conscious import ConsciousEpisode


class MemoryMosaic:
    """Append-only conscious episode log for downstream replay/analysis."""

    def __init__(self, path: str | Path = "logs/conscious_episodes.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_conscious_episode(self, episode: ConsciousEpisode) -> None:
        self._append_jsonl(episode.to_dict())

    def iter_dicts(self) -> Iterator[Dict[str, object]]:
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def tail(self, n: int = 10) -> Iterable[Dict[str, object]]:
        buffer: list[Dict[str, object]] = []
        for payload in self.iter_dicts():
            buffer.append(payload)
            if len(buffer) > n:
                buffer.pop(0)
        return buffer

    def _append_jsonl(self, payload: Dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
