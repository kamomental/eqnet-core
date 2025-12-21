from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .scene_frame import SceneFrame


@dataclass
class ActivationNode:
    """Single node inside an ignition chain."""

    node_id: str
    activation: float
    cue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "activation": float(self.activation),
            "cue": self.cue,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ActivationNode":
        return ActivationNode(
            node_id=str(payload.get("node_id", "unknown")),
            activation=float(payload.get("activation", 0.0)),
            cue=payload.get("cue"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ConfidenceSample:
    """Confidence value at a discrete time during recall."""

    step: int
    conf_internal: float
    conf_external: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "conf_internal": float(self.conf_internal),
            "conf_external": float(self.conf_external),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ConfidenceSample":
        return ConfidenceSample(
            step=int(payload.get("step", 0)),
            conf_internal=float(payload.get("conf_internal", 0.0)),
            conf_external=float(payload.get("conf_external", 0.0)),
        )


@dataclass
class ReplayEvent:
    """Record of replayed scene selection during ignition."""

    scene_id: str
    replay_source: str = "replay"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "replay_source": self.replay_source,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ReplayEvent":
        return ReplayEvent(
            scene_id=str(payload.get("scene_id", "scene")),
            replay_source=str(payload.get("replay_source", "replay")),
            payload=dict(payload.get("payload", {})),
        )


@dataclass
class ActivationTrace:
    """Structured log describing how a memory ignition unfolded."""

    trace_id: str
    timestamp: float
    trigger_context: Dict[str, Any]
    anchor_hit: Optional[str]
    activation_chain: List[ActivationNode]
    confidence_curve: List[ConfidenceSample]
    replay_events: List[ReplayEvent]
    notes: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    scene_frames: List[SceneFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": float(self.timestamp),
            "trigger_context": dict(self.trigger_context),
            "anchor_hit": self.anchor_hit,
            "activation_chain": [node.to_dict() for node in self.activation_chain],
            "confidence_curve": [sample.to_dict() for sample in self.confidence_curve],
            "replay_events": [event.to_dict() for event in self.replay_events],
            "notes": self.notes,
            "metadata": dict(self.metadata),
            "scene_frames": [frame.to_dict() for frame in self.scene_frames],
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ActivationTrace":
        return ActivationTrace(
            trace_id=str(payload.get("trace_id", "trace")),
            timestamp=float(payload.get("timestamp", 0.0)),
            trigger_context=dict(payload.get("trigger_context", {})),
            anchor_hit=payload.get("anchor_hit"),
            activation_chain=[
                ActivationNode.from_dict(item)
                for item in payload.get("activation_chain", [])
            ],
            confidence_curve=[
                ConfidenceSample.from_dict(item)
                for item in payload.get("confidence_curve", [])
            ],
            replay_events=[
                ReplayEvent.from_dict(item)
                for item in payload.get("replay_events", [])
            ],
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
            scene_frames=[
                SceneFrame.from_dict(item)
                for item in payload.get("scene_frames", [])
            ],
        )


class ActivationTraceLogger:
    """JSONL logger for :class:`ActivationTrace`."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: ActivationTrace) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def iter_traces(self) -> Iterator[ActivationTrace]:
        if self.path is None or not self.path.exists():
            return iter(())

        def _reader() -> Iterator[ActivationTrace]:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield ActivationTrace.from_dict(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return _reader()


def load_traces_for_day(
    path: str | Path,
    *,
    day: Optional[date] = None,
) -> List[ActivationTrace]:
    """Utility helper for nightly passes."""

    logger = ActivationTraceLogger(path)
    traces: List[ActivationTrace] = []
    for trace in logger.iter_traces():
        if day is None:
            traces.append(trace)
            continue
        ts = datetime.utcfromtimestamp(trace.timestamp)
        if ts.date() == day:
            traces.append(trace)
    return traces


__all__ = [
    "ActivationNode",
    "ConfidenceSample",
    "ReplayEvent",
    "ActivationTrace",
    "ActivationTraceLogger",
    "load_traces_for_day",
]
