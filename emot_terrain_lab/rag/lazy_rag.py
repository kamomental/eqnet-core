# -*- coding: utf-8 -*-
"""
LazyRAG: Graphで候補IDを絞り、JSONLから該当IDだけ拾って context を作る。

- Graph: JSON (nodes/edges)
- Store: JSONL (memory log)
- Retrieval: graph candidate ids -> fetch matching rows -> rank -> format
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import logging
import os
import re
from rag.assoc_safety import calc_saturation_stats, clamp_score, sanitize_weights

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]+")


def _tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _WORD_RE.findall(text)


def _char_bigrams(text: str) -> set[str]:
    s = (text or "").strip()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _overlap_score(query: str, doc: str) -> float:
    qt = set(_tokenize(query))
    dt = set(_tokenize(doc))
    tok = len(qt & dt) / (len(qt) + 1e-9)
    qb = _char_bigrams(query)
    db = _char_bigrams(doc)
    bi = len(qb & db) / (len(qb) + 1e-9)
    return 0.55 * tok + 0.45 * bi


@dataclass
class LazyRAGConfig:
    graph_path: Path
    memory_jsonl_path: Path
    topk_candidates: int = 50
    topk_context: int = 6
    max_chars_per_item: int = 420
    header: str = "参考（過去ログ/メモ）"
    include_meta: bool = True
    score_weights: Optional[Dict[str, float]] = None
    score_clamp_min: float = 0.0
    score_clamp_max: float = 1.0
    score_diag_enabled: bool = False
    score_diag_warn_sat_ratio: float = 0.6
    score_diag_every_n: int = 0

    @staticmethod
    def from_env() -> "LazyRAGConfig":
        gp = Path(os.getenv("EQNET_LAZY_RAG_GRAPH", "data/state/memory_graph.json"))
        mp = Path(os.getenv("EQNET_LAZY_RAG_JSONL", "data/logs.jsonl"))
        token_w = float(os.getenv("EQNET_LAZY_RAG_SCORE_TOKEN", "0.55"))
        bigram_w = float(os.getenv("EQNET_LAZY_RAG_SCORE_BIGRAM", "0.45"))
        clamp_min = float(os.getenv("EQNET_LAZY_RAG_SCORE_CLAMP_MIN", "0.0"))
        clamp_max = float(os.getenv("EQNET_LAZY_RAG_SCORE_CLAMP_MAX", "1.0"))
        diag_enabled = (os.getenv("EQNET_LAZY_RAG_SCORE_DIAG", "0") or "0").lower() in {
            "1",
            "true",
            "on",
        }
        diag_warn = float(os.getenv("EQNET_LAZY_RAG_SCORE_DIAG_WARN_SAT_RATIO", "0.6"))
        diag_every_n = int(os.getenv("EQNET_LAZY_RAG_SCORE_DIAG_EVERY_N", "0"))
        return LazyRAGConfig(
            graph_path=gp,
            memory_jsonl_path=mp,
            topk_candidates=int(os.getenv("EQNET_LAZY_RAG_TOPK_CAND", "50")),
            topk_context=int(os.getenv("EQNET_LAZY_RAG_TOPK_CTX", "6")),
            score_weights={"token": token_w, "bigram": bigram_w},
            score_clamp_min=clamp_min,
            score_clamp_max=clamp_max,
            score_diag_enabled=diag_enabled,
            score_diag_warn_sat_ratio=diag_warn,
            score_diag_every_n=diag_every_n,
        )


class LazyRAG:
    def __init__(self, cfg: Optional[LazyRAGConfig] = None) -> None:
        self.cfg = cfg or LazyRAGConfig.from_env()
        self._logger = logging.getLogger(__name__)
        self._diag_calls = 0
        self._last_score_diag: Dict[str, float] = {}
        raw_weights = self.cfg.score_weights or {"token": 0.55, "bigram": 0.45}
        self._score_weights = sanitize_weights(
            {
                "token": float(raw_weights.get("token", 0.55)),
                "bigram": float(raw_weights.get("bigram", 0.45)),
            },
            normalize=True,
            fallback_key="token",
        )
        self._score_clamp_min = float(self.cfg.score_clamp_min)
        self._score_clamp_max = float(self.cfg.score_clamp_max)
        self._graph: Optional[Dict[str, Any]] = None
        self._node_text: Dict[str, str] = {}
        self._edges: Dict[str, List[str]] = {}

    def _score(self, query: str, doc: str) -> float:
        qt = set(_tokenize(query))
        dt = set(_tokenize(doc))
        tok = len(qt & dt) / (len(qt) + 1e-9)
        qb = _char_bigrams(query)
        db = _char_bigrams(doc)
        bi = len(qb & db) / (len(qb) + 1e-9)
        raw = (self._score_weights["token"] * tok) + (self._score_weights["bigram"] * bi)
        return clamp_score(raw, self._score_clamp_min, self._score_clamp_max)

    def _score_detail(self, query: str, doc: str) -> Tuple[float, float]:
        qt = set(_tokenize(query))
        dt = set(_tokenize(doc))
        tok = len(qt & dt) / (len(qt) + 1e-9)
        qb = _char_bigrams(query)
        db = _char_bigrams(doc)
        bi = len(qb & db) / (len(qb) + 1e-9)
        raw = (self._score_weights["token"] * tok) + (self._score_weights["bigram"] * bi)
        clamped = clamp_score(raw, self._score_clamp_min, self._score_clamp_max)
        return raw, clamped

    def _maybe_log_score_diag(self, clamped_scores: List[float]) -> None:
        sat_min, sat_max, n = calc_saturation_stats(
            clamped_scores, self._score_clamp_min, self._score_clamp_max
        )
        sat_ratio = float((sat_min + sat_max) / n) if n > 0 else 0.0
        self._last_score_diag = {
            "n": float(n),
            "sat_min": float(sat_min),
            "sat_max": float(sat_max),
            "sat_ratio": float(sat_ratio),
            "clamp_min": float(self._score_clamp_min),
            "clamp_max": float(self._score_clamp_max),
        }
        if not self.cfg.score_diag_enabled:
            return
        self._diag_calls += 1
        warn_threshold = float(self.cfg.score_diag_warn_sat_ratio)
        every_n = int(self.cfg.score_diag_every_n)
        should_warn = sat_ratio >= warn_threshold
        should_debug = every_n > 0 and (self._diag_calls % every_n == 0)
        if not should_warn and not should_debug:
            return
        msg = (
            "lazy_rag.score_diag n=%d sat_min=%d sat_max=%d sat_ratio=%.3f clamp=[%.3f,%.3f]"
        )
        args = (
            n,
            sat_min,
            sat_max,
            sat_ratio,
            self._score_clamp_min,
            self._score_clamp_max,
        )
        if should_warn:
            self._logger.warning(msg, *args)
        else:
            self._logger.debug(msg, *args)

    @property
    def last_score_diag(self) -> Dict[str, float]:
        return dict(self._last_score_diag)

    def _load_graph(self) -> None:
        if self._graph is not None:
            return
        path = self.cfg.graph_path
        if not path.exists():
            self._graph = {"nodes": [], "edges": []}
            return
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        self._graph = data
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        for n in nodes:
            nid = str(n.get("id") or n.get("node_id") or "")
            if not nid:
                continue
            t = str(n.get("title") or n.get("text") or n.get("summary") or n.get("label") or "")
            self._node_text[nid] = t
        adj: Dict[str, List[str]] = {}
        for e in edges:
            s = str(e.get("src") or e.get("source") or "")
            d = str(e.get("dst") or e.get("target") or "")
            if not s or not d:
                continue
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, []).append(s)
        self._edges = adj

    def candidate_ids(self, query: str) -> List[str]:
        self._load_graph()
        if not self._node_text:
            return []
        scored: List[Tuple[float, str]] = []
        for nid, text in self._node_text.items():
            scored.append((self._score(query, text), nid))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [nid for _, nid in scored[: self.cfg.topk_candidates]]
        expanded: List[str] = []
        seen = set()
        for nid in top:
            if nid not in seen:
                expanded.append(nid)
                seen.add(nid)
            for nb in self._edges.get(nid, [])[:5]:
                if nb not in seen:
                    expanded.append(nb)
                    seen.add(nb)
        return expanded[: self.cfg.topk_candidates]

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            return []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

    def _row_id(self, row: Dict[str, Any]) -> Optional[str]:
        for key in ("id", "node_id", "turn_id", "event_id", "trace_id", "uuid"):
            value = row.get(key)
            if value:
                return str(value)
        data = row.get("data")
        if isinstance(data, dict):
            for key in ("id", "node_id"):
                value = data.get(key)
                if value:
                    return str(value)
        return None

    def _row_text(self, row: Dict[str, Any]) -> str:
        for key in ("text", "message", "content", "summary"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = row.get("data")
        if isinstance(data, dict):
            for key in ("text", "message", "content", "summary", "note"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return json.dumps(row, ensure_ascii=False)

    def fetch_rows_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        want = set(ids)
        out: List[Dict[str, Any]] = []
        for row in self._iter_jsonl(self.cfg.memory_jsonl_path):
            rid = self._row_id(row)
            if rid and rid in want:
                out.append(row)
        return out

    def build_context(self, query: str) -> Optional[str]:
        ids = self.candidate_ids(query)
        if not ids:
            return None
        rows = self.fetch_rows_by_ids(ids)
        if not rows:
            return None
        scored: List[Tuple[float, Dict[str, Any]]] = []
        diag_scores: List[float] = []
        for row in rows:
            text = self._row_text(row)
            _, clamped = self._score_detail(query, text)
            scored.append((clamped, row))
            diag_scores.append(clamped)
        self._maybe_log_score_diag(diag_scores)
        scored.sort(reverse=True, key=lambda x: x[0])
        picked = [row for _, row in scored[: self.cfg.topk_context]]
        lines: List[str] = [f"## {self.cfg.header}"]
        for row in picked:
            text = self._row_text(row).replace("\n", " ").strip()
            if len(text) > self.cfg.max_chars_per_item:
                text = text[: self.cfg.max_chars_per_item] + "…"
            prefix = ""
            if self.cfg.include_meta:
                ts = row.get("timestamp") or row.get("timestamp_ms") or row.get("time") or ""
                src = row.get("source") or row.get("event") or ""
                rid = self._row_id(row) or ""
                meta = " ".join([x for x in [str(ts), str(src), str(rid)] if x])
                if meta:
                    prefix = f"[{meta}] "
            lines.append(f"- {prefix}{text}")
        return "\n".join(lines)
