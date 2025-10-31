# -*- coding: utf-8 -*-
"""Loaders for policy invariants and cache settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

_BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AckPolicy:
    allow_intents: List[str] = field(default_factory=lambda: ["listen", "clarify", "plan", "empathize"])
    forbid_memory_write: bool = True
    max_speculative_tokens: int = 40


@dataclass(frozen=True)
class KvPrefixPolicy:
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    invalidate_on: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RagPolicy:
    max_topk: int = 5
    high_influence_radius_threshold: float = 0.7
    defer_high_influence_commits: bool = True


@dataclass(frozen=True)
class MemoryPolicy:
    commit_phase: str = "post_final"
    require_flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class KpiTargets:
    tone_match_min: float = 0.9
    culture_violation_per_100_max: float = 0.5
    ack_latency_ms_max: float = 300.0
    p50_final_s_max: float = 1.6
    p95_final_s_max: float = 3.0


@dataclass(frozen=True)
class Invariants:
    ack: AckPolicy = AckPolicy()
    kv_prefix: KvPrefixPolicy = KvPrefixPolicy()
    rag: RagPolicy = RagPolicy()
    memory: MemoryPolicy = MemoryPolicy()
    kpi_targets: KpiTargets = KpiTargets()


@dataclass(frozen=True)
class CachePaths:
    prefix_kv: Dict[str, Any] = field(default_factory=dict)
    rag: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)
    terrain: Dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_invariants(path: str | Path = _BASE_DIR / "invariants.yaml") -> Invariants:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return Invariants()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Invariants(
        ack=AckPolicy(**(data.get("ack", {}) or {})),
        kv_prefix=KvPrefixPolicy(**(data.get("kv_prefix", {}) or {})),
        rag=RagPolicy(**(data.get("rag", {}) or {})),
        memory=MemoryPolicy(**(data.get("memory", {}) or {})),
        kpi_targets=KpiTargets(**(data.get("kpi_targets", {}) or {})),
    )


@lru_cache(maxsize=1)
def load_cache_config(path: str | Path = _BASE_DIR / "cache.yaml") -> CachePaths:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return CachePaths()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return CachePaths(
        prefix_kv=data.get("prefix_kv", {}) or {},
        rag=data.get("rag", {}) or {},
        embeddings=data.get("embeddings", {}) or {},
        terrain=data.get("terrain", {}) or {},
    )
