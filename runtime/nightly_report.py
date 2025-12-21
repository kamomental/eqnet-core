from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from eqnet_core.models.activation_trace import ActivationTrace, load_traces_for_day


@dataclass
class NightlyRecallReport:
    """Audit payload derived from activation traces."""

    schema_version: str
    trace_count: int
    anchors: Dict[str, int]
    confidence: Dict[str, float]
    ignitions: List[Dict[str, Any]]
    dream_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_count": self.trace_count,
            "anchors": dict(self.anchors),
            "confidence": dict(self.confidence),
            "ignitions": list(self.ignitions),
            "dream_prompt": self.dream_prompt,
        }


def generate_recall_report(
    trace_path: str | Path,
    *,
    day: Optional[date] = None,
) -> NightlyRecallReport:
    traces = load_traces_for_day(trace_path, day=day)
    anchors: Dict[str, int] = {}
    ignitions: List[Dict[str, Any]] = []
    internal_samples: List[float] = []
    external_samples: List[float] = []
    for trace in traces:
        anchor = trace.anchor_hit or "unknown"
        anchors[anchor] = anchors.get(anchor, 0) + 1
        internal, external = _confidence_tail(trace)
        internal_samples.append(internal)
        external_samples.append(external)
        ignitions.append(
            {
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp,
                "anchor": trace.anchor_hit,
                "activation_total": round(sum(node.activation for node in trace.activation_chain), 3),
                "conf_internal_final": round(internal, 3),
                "conf_external_final": round(external, 3),
                "scene_frames": [frame.to_dict() for frame in trace.scene_frames],
            }
        )
    confidence_summary = {
        "avg_internal": round(mean(internal_samples), 3) if internal_samples else 0.0,
        "avg_external": round(mean(external_samples), 3) if external_samples else 0.0,
        "max_internal": round(max(internal_samples), 3) if internal_samples else 0.0,
        "min_internal": round(min(internal_samples), 3) if internal_samples else 0.0,
    }
    prompt = _dream_prompt(traces)
    return NightlyRecallReport(
        schema_version="activation_nightly_v1",
        trace_count=len(traces),
        anchors=anchors,
        confidence=confidence_summary,
        ignitions=ignitions,
        dream_prompt=prompt,
    )


def _confidence_tail(trace: ActivationTrace) -> tuple[float, float]:
    if not trace.confidence_curve:
        return 0.0, 0.0
    sample = trace.confidence_curve[-1]
    return float(sample.conf_internal), float(sample.conf_external)


def _dream_prompt(traces: List[ActivationTrace]) -> str:
    if not traces:
        return "No ignition events captured today; skip dream prompt."
    anchor = traces[0].anchor_hit or "the remembered place"
    tags = traces[0].trigger_context.get("context_tags") if traces[0].trigger_context else None
    route = " -> ".join(tags) if isinstance(tags, list) and tags else "landmarks"
    return (
        "Draw a 3-panel dream map about walking from {anchor} along {route}. "
        "Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. "
        "Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. "
        "Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation."
    ).format(anchor=anchor, route=route)


__all__ = ["NightlyRecallReport", "generate_recall_report"]
