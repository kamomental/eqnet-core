# -*- coding: utf-8 -*-
"""
Replay memory persistence for internal simulations.

Records replay traces to a JSONL file so downstream analytics and learning
passes can consume the history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping
import json
import re


DEFAULT_PATH = Path("state/replay_memory.jsonl")
DEFAULT_THINK_LOG_PATH = Path("logs/think_log.jsonl")
DEFAULT_ACT_LOG_PATH = Path("logs/act_log.jsonl")


class MemoryKind(str, Enum):
    EXPERIENCE = "experience"
    IMAGERY = "imagery"
    HYPOTHESIS = "hypothesis"
    BORROWED_IDEA = "borrowed_idea"
    DISCUSSION = "discussion"
    UNKNOWN = "unknown"


_LEGACY_KIND_MAP = {
    "episodic": MemoryKind.EXPERIENCE.value,
}


def normalize_memory_kind(value: Optional[str]) -> tuple[str, bool]:
    if value is None:
        return MemoryKind.UNKNOWN.value, False
    text = str(value).strip().lower()
    if not text:
        return MemoryKind.UNKNOWN.value, False
    if text in _LEGACY_KIND_MAP:
        return _LEGACY_KIND_MAP[text], True
    allowed = {item.value for item in MemoryKind}
    if text in allowed:
        return text, False
    return MemoryKind.UNKNOWN.value, True


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
    memory_kind: Optional[str] = None
    novelty_score: Optional[float] = None
    social_weight: Optional[float] = None
    constraint_weight: Optional[float] = None
    emotion_modulation: Optional[float] = None
    conf_internal: Optional[float] = None
    conf_external: Optional[float] = None
    replay_source: Optional[str] = None
    activation_trace_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    tau: float = 0.0


class ReplayMemory:
    """Append-only store for replay traces."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        think_log_path: Optional[Path] = None,
        act_log_path: Optional[Path] = None,
    ) -> None:
        self.path = path or DEFAULT_PATH
        self.think_log_path = think_log_path or DEFAULT_THINK_LOG_PATH
        self.act_log_path = act_log_path or DEFAULT_ACT_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.think_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.act_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _route_for_kind(self, memory_kind: str) -> Path:
        if memory_kind == MemoryKind.EXPERIENCE.value:
            return self.act_log_path
        return self.think_log_path

    def store(self, trace: ReplayTrace) -> None:
        payload = asdict(trace)
        normalized_kind, changed = normalize_memory_kind(trace.memory_kind)
        payload["memory_kind"] = normalized_kind
        route = self._route_for_kind(normalized_kind)
        if changed:
            payload.setdefault("meta", {})
            meta = payload.get("meta")
            if isinstance(meta, dict):
                meta.setdefault("audit_event", "MEMORY_KIND_NORMALIZED")
                meta.setdefault("memory_kind_original", trace.memory_kind)
            payload["meta"] = meta
        payload.setdefault("turn_id", payload.get("trace_id"))
        self._append_jsonl(route, payload)
        # Backward-compat mirror until all readers move to think/act logs.
        self._append_jsonl(self.path, payload)

    def _find_latest_in_log(self, path: Path, trace_id: str) -> Optional[Dict[str, Any]]:
        if not trace_id:
            return None
        rows = self._iter_path(path)
        for row in reversed(rows):
            if str(row.get("trace_id", "")) == trace_id:
                return row
        return None

    def _write_promotion_audit(
        self,
        source: Dict[str, Any],
        *,
        audit_event: str,
        reason: Optional[str] = None,
        evidence_event_id: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "trace_id": source.get("trace_id"),
            "turn_id": source.get("turn_id") or source.get("trace_id"),
            "episode_id": source.get("episode_id"),
            "timestamp": source.get("timestamp"),
            "memory_kind": source.get("memory_kind"),
            "meta": {"audit_event": audit_event},
        }
        if reason:
            payload["meta"]["reason"] = reason
        if evidence_event_id:
            payload["meta"]["evidence_event_id"] = evidence_event_id
        self._append_jsonl(self.think_log_path, payload)

    def promote_with_evidence(
        self,
        thought_trace_id: str,
        *,
        evidence_event_id: Optional[str],
    ) -> Dict[str, Any]:
        """Promote thought-side memory into experience only with evidence."""
        source = self._find_latest_in_log(self.think_log_path, str(thought_trace_id))
        if source is None:
            return {"promoted": False, "reason": "not_found"}
        if not evidence_event_id or not str(evidence_event_id).strip():
            self._write_promotion_audit(
                source,
                audit_event="PROMOTION_GUARD_BLOCKED",
                reason="missing_evidence_event_id",
            )
            return {"promoted": False, "reason": "missing_evidence_event_id"}

        promoted = dict(source)
        promoted["memory_kind"] = MemoryKind.EXPERIENCE.value
        promoted.setdefault("meta", {})
        meta = promoted.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["audit_event"] = "PROMOTION_GUARD_PASSED"
        meta["promotion_evidence_event_id"] = str(evidence_event_id)
        meta["promoted_from_memory_kind"] = source.get("memory_kind")
        promoted["meta"] = meta
        promoted.setdefault("turn_id", promoted.get("trace_id"))

        self._append_jsonl(self.act_log_path, promoted)
        self._append_jsonl(self.path, promoted)
        self._write_promotion_audit(
            source,
            audit_event="PROMOTION_GUARD_PASSED",
            evidence_event_id=str(evidence_event_id),
        )
        return {"promoted": True, "reason": "ok", "trace_id": promoted.get("trace_id")}

    def _iter_path(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def iter_recent(self, limit: int = 500) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._iter_path(self.act_log_path))
        rows.extend(self._iter_path(self.think_log_path))
        if not rows:
            rows = self._iter_path(self.path)
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


__all__ = [
    "ReplayMemory",
    "ReplayTrace",
    "MemoryKind",
    "normalize_memory_kind",
    "DEFAULT_PATH",
    "DEFAULT_THINK_LOG_PATH",
    "DEFAULT_ACT_LOG_PATH",
]
