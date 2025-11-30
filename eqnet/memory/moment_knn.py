"""KNN scaffolding for MomentLog-derived memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class Neighbor:
    """Lightweight neighbor descriptor."""

    moment_id: str
    episode_id: Optional[str]
    monument_id: Optional[str]
    topic: Optional[str]
    similarity: float
    importance: float
    emotion_tag: Optional[str] = None
    summary: Optional[str] = None


class MomentKNNIndex:
    """Simple in-memory cosine index for conversational moments."""

    def __init__(self, capacity: int = 256) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: List[Dict[str, Any]] = []
        self._id_index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        self._id_index = {
            entry["meta"].get("moment_id"): idx
            for idx, entry in enumerate(self._entries)
            if entry["meta"].get("moment_id")
        }

    def _maybe_trim(self) -> None:
        while len(self._entries) > self._capacity:
            removed = self._entries.pop(0)
            moment_id = removed["meta"].get("moment_id")
            if moment_id in self._id_index:
                self._id_index.pop(moment_id)
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        embedding: Sequence[float],
        *,
        moment_id: str,
        topic: Optional[str],
        summary: Optional[str],
        emotion_tag: Optional[str],
        episode_id: Optional[str] = None,
        monument_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> None:
        """Register a brand-new L1 moment."""

        if moment_id in self._id_index:
            return
        vec = np.asarray(embedding, dtype=float).reshape(-1)
        if vec.size == 0:
            return
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm == 0:
            return
        normalized = vec / norm
        meta = {
            "moment_id": moment_id,
            "topic": topic,
            "summary": summary,
            "emotion_tag": emotion_tag,
            "episode_id": episode_id,
            "monument_id": monument_id,
            "importance": float(importance),
        }
        entry = {"vec": normalized, "meta": meta, "strength": 1.0}
        self._entries.append(entry)
        self._rebuild_index()
        self._maybe_trim()

    def build_from_moment_logs(
        self,
        moment_logs: Sequence[Tuple[Sequence[float], Dict[str, Optional[str]]]],
    ) -> None:
        self._entries = []
        self._id_index = {}
        for embedding, meta in moment_logs:
            self.add(embedding, **meta)

    # ------------------------------------------------------------------
    # Strength dynamics
    # ------------------------------------------------------------------

    def decay_all(self, factor: float = 0.98) -> None:
        factor = float(max(0.0, min(1.0, factor)))
        for entry in self._entries:
            entry["strength"] *= factor

    def reinforce(self, moment_id: str, alpha: float = 0.2) -> None:
        idx = self._id_index.get(moment_id)
        if idx is None:
            return
        alpha = float(max(0.0, min(1.0, alpha)))
        entry = self._entries[idx]
        strength = entry["strength"]
        entry["strength"] = strength + (1.0 - strength) * alpha

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, embedding: Sequence[float], k: int = 3) -> List[Neighbor]:
        if not self._entries:
            return []
        query = np.asarray(embedding, dtype=float).reshape(-1)
        if query.size == 0:
            return []
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            return []
        normalized_query = query / norm
        neighbors: List[Neighbor] = []
        for entry in self._entries:
            vec = entry["vec"]
            if vec.shape != normalized_query.shape:
                continue
            sim = float(np.dot(normalized_query, vec))
            weighted = sim * entry["strength"]
            meta = entry["meta"]
            neighbors.append(
                Neighbor(
                    moment_id=meta.get("moment_id") or "",
                    episode_id=meta.get("episode_id"),
                    monument_id=meta.get("monument_id"),
                    topic=meta.get("topic"),
                    similarity=weighted,
                    importance=float(meta.get("importance", 0.5)),
                    emotion_tag=meta.get("emotion_tag"),
                    summary=meta.get("summary"),
                )
            )
        neighbors.sort(key=lambda n: n.similarity, reverse=True)
        return neighbors[: max(1, k)]