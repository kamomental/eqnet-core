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
import os
import re

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

    @staticmethod
    def from_env() -> "LazyRAGConfig":
        gp = Path(os.getenv("EQNET_LAZY_RAG_GRAPH", "data/state/memory_graph.json"))
        mp = Path(os.getenv("EQNET_LAZY_RAG_JSONL", "data/logs.jsonl"))
        return LazyRAGConfig(
            graph_path=gp,
            memory_jsonl_path=mp,
            topk_candidates=int(os.getenv("EQNET_LAZY_RAG_TOPK_CAND", "50")),
            topk_context=int(os.getenv("EQNET_LAZY_RAG_TOPK_CTX", "6")),
        )


class LazyRAG:
    def __init__(self, cfg: Optional[LazyRAGConfig] = None) -> None:
        self.cfg = cfg or LazyRAGConfig.from_env()
        self._graph: Optional[Dict[str, Any]] = None
        self._node_text: Dict[str, str] = {}
        self._edges: Dict[str, List[str]] = {}

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
            scored.append((_overlap_score(query, text), nid))
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
        for row in rows:
            text = self._row_text(row)
            scored.append((_overlap_score(query, text), row))
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
