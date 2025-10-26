# -*- coding: utf-8 -*-
"""
Replay memory persistence for internal simulations.

Records replay traces to a JSONL file so downstream analytics and learning
passes can consume the history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping
import json
import re


DEFAULT_PATH = Path("state/replay_memory.jsonl")


@dataclass
class ReplayTrace:
    trace_id: str
    episode_id: str
    timestamp: float
    source: str
    horizon: int
    uncertainty: float
    mood: Dict[str, float]
    value: Dict[str, float]
    controls: Dict[str, Any]
    imagined: Dict[str, Any]
    meta: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    tau: float = 0.0


class ReplayMemory:
    """Append-only store for replay traces."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def store(self, trace: ReplayTrace) -> None:
        payload = json.dumps(asdict(trace), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def iter_recent(self, limit: int = 500) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-limit:]

    def load_all(self) -> List[Dict[str, Any]]:
        """Return every stored trace; primarily for nightly maintenance."""
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def rewrite(self, events: List[Dict[str, Any]]) -> None:
        """Rewrite the entire replay memory with the provided events."""
        with self.path.open("w", encoding="utf-8") as handle:
            for event in events:
                try:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                except Exception:
                    continue

    def sample_prioritized(
        self,
        *,
        k: int = 3,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return top-k traces by uncertainty/|ΔU|/normリスクの重み付け."""
        rows = self.iter_recent(limit=limit)
        if not rows:
            return []

        def score(row: Dict[str, Any]) -> float:
            uncertainty = float(row.get("uncertainty", 0.0))
            value = row.get("value", {})
            delta_u = abs(float(value.get("total", 0.0)))
            meta = row.get("meta", {})
            norm_risk = float(meta.get("norm_risk", 0.0))
            novelty = float(meta.get("novelty", 0.0))
            return 0.4 * uncertainty + 0.3 * delta_u + 0.2 * norm_risk + 0.1 * novelty

        ranked = sorted(rows, key=score, reverse=True)
        return ranked[:k]

    def topk(self, cue: Mapping[str, Any], *, k: int = 16, limit: int = 512) -> List[Dict[str, Any]]:
        """Return recent traces with a naive token-overlap similarity for metamemory."""
        rows = self.iter_recent(limit=limit)
        if not rows:
            return []

        text = str(cue.get("text", "")).lower()
        tokens = set(re.findall(r"\w+", text))
        if not tokens:
            tokens = set(text.split())

        results: List[Dict[str, Any]] = []
        for row in rows:
            imagined = row.get("imagined", {}) or {}
            candidate = str(imagined.get("best_action") or imagined.get("excerpt") or "")
            cand_lower = candidate.lower()
            cand_tokens = set(re.findall(r"\w+", cand_lower)) or set(cand_lower.split())
            if not cand_tokens:
                similarity = 0.0
            else:
                intersection = tokens & cand_tokens
                union = tokens | cand_tokens
                similarity = len(intersection) / max(1, len(union))
            partial: Dict[str, Any] = {}
            if candidate:
                words = candidate.split()
                if words:
                    partial["initial"] = words[0][0]
                digits = re.findall(r"\d{4}", candidate)
                if digits:
                    partial["year"] = digits[0]
                if "-" in candidate:
                    partial["phonology"] = candidate.split("-")[0]
            results.append({"sim": float(similarity), "partial": partial, "trace": row})

        results.sort(key=lambda item: item["sim"], reverse=True)
        return results[:k]

    def decay(self, factor: float, *, min_total: float = 0.05) -> None:
        """Apply exponential forgetting to stored traces."""
        factor = float(max(0.0, min(1.0, factor)))
        if factor >= 0.999 or not self.path.exists():
            return
        surviving: List[str] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                value = record.get("value", {})
                total = float(value.get("total", 0.0)) * factor
                value["total"] = total
                record["value"] = value
                record["uncertainty"] = float(record.get("uncertainty", 0.0)) * factor
                if abs(total) >= min_total:
                    surviving.append(json.dumps(record, ensure_ascii=False))
        with self.path.open("w", encoding="utf-8") as handle:
            for row in surviving:
                handle.write(row + "\n")


__all__ = ["ReplayMemory", "ReplayTrace", "DEFAULT_PATH"]
