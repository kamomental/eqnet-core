from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .association_graph import AssociationGraph, AssociationLink
from .dot_seed import DotSeed, DotSeedSet


@dataclass(frozen=True)
class InsightScore:
    total: float
    link_weight: float
    source_diversity: float
    novelty_gain: float
    tension_relief: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": float(self.total),
            "link_weight": float(self.link_weight),
            "source_diversity": float(self.source_diversity),
            "novelty_gain": float(self.novelty_gain),
            "tension_relief": float(self.tension_relief),
        }


@dataclass(frozen=True)
class InsightEvent:
    triggered: bool
    score: InsightScore
    link_key: str
    connected_seed_ids: tuple[str, ...] = ()
    connected_seed_keys: tuple[str, ...] = ()
    dominant_seed_label: str = ""
    summary: str = ""
    orient_bias: float = 0.0
    stabilizing_bias: float = 0.0
    cues: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "score": self.score.to_dict(),
            "link_key": self.link_key,
            "connected_seed_ids": list(self.connected_seed_ids),
            "connected_seed_keys": list(self.connected_seed_keys),
            "dominant_seed_label": self.dominant_seed_label,
            "summary": self.summary,
            "orient_bias": float(self.orient_bias),
            "stabilizing_bias": float(self.stabilizing_bias),
            "cues": list(self.cues),
            "reasons": list(self.reasons),
        }


@dataclass
class BasicInsightDetector:
    trigger_threshold: float = 0.42

    def detect(
        self,
        *,
        dot_seeds: DotSeedSet | Sequence[DotSeed],
        association_graph: AssociationGraph | Mapping[str, Any] | None,
        qualia_trust: float = 1.0,
    ) -> InsightEvent:
        seeds = tuple(dot_seeds.seeds if isinstance(dot_seeds, DotSeedSet) else dot_seeds)
        graph = _coerce_graph(association_graph)
        top_link = graph.edges[0] if graph.edges else None
        if top_link is None or len(seeds) < 2:
            return _neutral_event()
        left = _find_seed(seeds, top_link.left_seed_id)
        right = _find_seed(seeds, top_link.right_seed_id)
        source_diversity = top_link.source_diversity
        novelty_gain = top_link.novelty_gain
        tension_relief = top_link.unresolved_relief
        total = _clamp01(
            top_link.weight * 0.52
            + source_diversity * 0.18
            + novelty_gain * 0.16
            + tension_relief * 0.14
        ) * _clamp01(0.25 + 0.75 * qualia_trust)
        triggered = bool(total >= self.trigger_threshold and source_diversity > 0.0)
        summary = ""
        dominant_seed_label = ""
        if left is not None and right is not None:
            dominant_seed_label = left.label if left.strength >= right.strength else right.label
            summary = f"{left.label} <-> {right.label}"
        cues = ["insight_candidate"]
        if triggered:
            cues.append("insight_triggered")
        reasons = list(top_link.reasons)
        if triggered:
            reasons.append("coherent_link")
        score = InsightScore(
            total=total,
            link_weight=top_link.weight,
            source_diversity=source_diversity,
            novelty_gain=novelty_gain,
            tension_relief=tension_relief,
        )
        return InsightEvent(
            triggered=triggered,
            score=score,
            link_key=top_link.link_key,
            connected_seed_ids=(top_link.left_seed_id, top_link.right_seed_id),
            connected_seed_keys=(top_link.left_seed_key, top_link.right_seed_key),
            dominant_seed_label=dominant_seed_label,
            summary=summary,
            orient_bias=_clamp01(total * 0.82),
            stabilizing_bias=_clamp01(tension_relief * 0.4 + total * 0.2),
            cues=tuple(cues),
            reasons=tuple(dict.fromkeys(reasons)),
        )


def _coerce_graph(value: AssociationGraph | Mapping[str, Any] | None) -> AssociationGraph:
    if isinstance(value, AssociationGraph):
        return value
    if not isinstance(value, Mapping):
        return AssociationGraph()
    raw_edges = value.get("edges") or []
    edges: list[AssociationLink] = []
    for item in raw_edges:
        if not isinstance(item, Mapping):
            continue
        try:
            edges.append(
                AssociationLink(
                    link_id=str(item.get("link_id") or ""),
                    link_key=str(item.get("link_key") or ""),
                    left_seed_id=str(item.get("left_seed_id") or ""),
                    right_seed_id=str(item.get("right_seed_id") or ""),
                    left_seed_key=str(item.get("left_seed_key") or ""),
                    right_seed_key=str(item.get("right_seed_key") or ""),
                    weight=_clamp01(item.get("weight")),
                    source_diversity=_clamp01(item.get("source_diversity")),
                    anchor_overlap=_clamp01(item.get("anchor_overlap")),
                    novelty_gain=_clamp01(item.get("novelty_gain")),
                    unresolved_relief=_clamp01(item.get("unresolved_relief")),
                    reasons=tuple(str(reason) for reason in item.get("reasons") or [] if str(reason).strip()),
                )
            )
        except Exception:
            continue
    edges = sorted(edges, key=lambda item: item.weight, reverse=True)
    return AssociationGraph(edges=tuple(edges), dominant_link_id=edges[0].link_id if edges else "", dominant_weight=edges[0].weight if edges else 0.0)


def _find_seed(seeds: Sequence[DotSeed], seed_id: str) -> DotSeed | None:
    for seed in seeds:
        if seed.seed_id == seed_id:
            return seed
    return None


def _neutral_event() -> InsightEvent:
    return InsightEvent(
        triggered=False,
        score=InsightScore(
            total=0.0,
            link_weight=0.0,
            source_diversity=0.0,
            novelty_gain=0.0,
            tension_relief=0.0,
        ),
        link_key="",
    )


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
