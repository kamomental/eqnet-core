"""Minimal episodic story graph stub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class StoryGraph:
    """Keeps a rolling list of episodes and basic summaries."""

    buffer: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 1000
    _last_tokens: List[int] = field(default_factory=list)

    def update(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        self.buffer.append(episode)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]
        summary = {
            "count": len(self.buffer),
            "last_stage": episode.get("stage"),
            "last_tokens": episode.get("tokens", [])[:5],
        }
        # keep last tokens for quick self-consistency proxy
        toks = episode.get("tokens", [])
        if isinstance(toks, list):
            self._last_tokens = list(toks)
        return summary

    def query_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.buffer[-limit:]

    # --------------------------- simple self-consistency proxy
    def self_consistency_error(self, new_tokens: Sequence[int] | None) -> float:
        """Return a simple Jaccard-distance proxy between last and current tokens.
        Higher means more inconsistency.
        """
        if not new_tokens or not self._last_tokens:
            return 0.0
        a = set(int(t) for t in self._last_tokens[:16])
        b = set(int(t) for t in list(new_tokens)[:16])
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = max(1, len(a | b))
        jaccard = 1.0 - (inter / union)
        return float(max(0.0, min(1.0, jaccard)))
