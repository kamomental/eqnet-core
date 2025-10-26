# -*- coding: utf-8 -*-
"""Shared/private thought clustering helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))


@dataclass
class ShareGraph:
    adjacency: Dict[Tuple[str, str], float]
    digest: str


class ThoughtAssigner:
    """Approximate assigner tagging thoughts as shared/private across agents."""

    def __init__(self, *, sim_th: float = 0.75) -> None:
        self.sim_th = float(sim_th)

    def assign(self, packets_by_agent: Dict[str, List[Dict]]) -> Tuple[Dict[str, List[Dict]], ShareGraph]:
        adjacency: Dict[Tuple[str, str], float] = {}
        agents = sorted(packets_by_agent.keys())
        for idx, a in enumerate(agents):
            for b in agents[idx + 1 :]:
                sim = self._best_similarity(packets_by_agent[a], packets_by_agent[b])
                adjacency[(a, b)] = sim
        for agent, packets in packets_by_agent.items():
            for packet in packets:
                packet["share"] = "private"
                for (a, b), sim in adjacency.items():
                    if agent not in (a, b):
                        continue
                    other = b if agent == a else a
                    if sim >= self.sim_th and self._kind_exists(packets_by_agent.get(other, []), packet.get("kind")):
                        packet["share"] = "shared"
                        break
        edges: List[Tuple[str, str, float]] = [
            (min(a, b), max(a, b), round(val, 3)) for (a, b), val in adjacency.items()
        ]
        edges.sort()
        digest_src = "|".join(f"{a}-{b}:{w:.3f}" for a, b, w in edges)
        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:16]
        return packets_by_agent, ShareGraph(adjacency=adjacency, digest=digest)

    def _best_similarity(self, pa: List[Dict], pb: List[Dict]) -> float:
        score = 0.0
        for a in pa:
            vec_a = a.get("vec")
            if not isinstance(vec_a, list):
                continue
            for b in pb:
                if a.get("kind") != b.get("kind"):
                    continue
                vec_b = b.get("vec")
                if not isinstance(vec_b, list):
                    continue
                score = max(score, _cosine(vec_a, vec_b))
        return score

    def _kind_exists(self, packets: List[Dict], kind: str | None) -> bool:
        if kind is None:
            return False
        return any(pkt.get("kind") == kind for pkt in packets)


__all__ = ["ThoughtAssigner", "ShareGraph"]
