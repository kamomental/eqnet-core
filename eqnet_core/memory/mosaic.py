from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from eqnet_core.models.conscious import ConsciousEpisode
from .working_memory_seed import (
    extract_long_term_theme_from_context_tags,
    extract_working_memory_seed_from_context_tags,
)


class MemoryMosaic:
    """Append-only conscious episode log for downstream replay/analysis."""

    def __init__(self, path: str | Path = "logs/conscious_episodes.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_conscious_episode(self, episode: ConsciousEpisode) -> None:
        payload = episode.to_dict()
        payload["working_memory_seed"] = extract_working_memory_seed_from_context_tags(
            episode.world_state.context_tags
        )
        payload["long_term_theme"] = extract_long_term_theme_from_context_tags(
            episode.world_state.context_tags
        )
        self._append_jsonl(payload)

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

    def latest_working_memory_seed(self, n: int = 12) -> Dict[str, object]:
        seeds: list[Dict[str, str]] = []
        for payload in self.tail(n):
            seed = payload.get("working_memory_seed")
            if isinstance(seed, dict) and (seed.get("focus") or seed.get("anchor")):
                seeds.append(
                    {
                        "focus": str(seed.get("focus") or "").strip(),
                        "anchor": str(seed.get("anchor") or "").strip(),
                    }
                )
        if not seeds:
            return {}
        focus_counts = Counter(seed["focus"] for seed in seeds if seed["focus"])
        anchor_counts = Counter(seed["anchor"] for seed in seeds if seed["anchor"])
        focus = focus_counts.most_common(1)[0][0] if focus_counts else ""
        anchor = anchor_counts.most_common(1)[0][0] if anchor_counts else ""
        strength = min(len(seeds) / max(n, 1), 1.0)
        if not focus and not anchor:
            return {}
        return {
            "focus": focus,
            "anchor": anchor,
            "strength": round(strength, 4),
        }

    def latest_long_term_theme(self, n: int = 12) -> Dict[str, object]:
        themes: list[Dict[str, str]] = []
        for payload in self.tail(n):
            theme = payload.get("long_term_theme")
            if not isinstance(theme, dict):
                continue
            focus = str(theme.get("focus") or "").strip()
            anchor = str(theme.get("anchor") or "").strip()
            kind = str(theme.get("kind") or "").strip()
            summary = str(theme.get("summary") or "").strip()
            if not focus and not anchor and not kind and not summary:
                continue
            themes.append(
                {
                    "focus": focus,
                    "anchor": anchor,
                    "kind": kind,
                    "summary": summary,
                }
            )
        if not themes:
            return {}
        focus_counts = Counter(theme["focus"] for theme in themes if theme["focus"])
        anchor_counts = Counter(theme["anchor"] for theme in themes if theme["anchor"])
        kind_counts = Counter(theme["kind"] for theme in themes if theme["kind"])
        focus = focus_counts.most_common(1)[0][0] if focus_counts else ""
        anchor = anchor_counts.most_common(1)[0][0] if anchor_counts else ""
        kind = kind_counts.most_common(1)[0][0] if kind_counts else ""
        summary = ""
        for theme in reversed(themes):
            if theme["summary"]:
                summary = theme["summary"]
                break
        strength = min(len(themes) / max(n, 1), 1.0)
        if not focus and not anchor and not kind and not summary:
            return {}
        return {
            "focus": focus,
            "anchor": anchor,
            "kind": kind,
            "summary": summary,
            "strength": round(strength, 4),
        }

    def _append_jsonl(self, payload: Dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
