from dataclasses import dataclass
from typing import Any, Dict, Optional, Mapping


@dataclass
class RuntimeAffectSummary:
    valence: float
    arousal: float
    confidence: float
    timestamp: float

    @classmethod
    def from_sample(cls, sample: Any) -> Optional["RuntimeAffectSummary"]:
        if sample is None:
            return None
        return cls(
            valence=float(getattr(sample, "valence", 0.0)),
            arousal=float(getattr(sample, "arousal", 0.0)),
            confidence=float(getattr(sample, "confidence", 0.0)),
            timestamp=float(getattr(sample, "timestamp", 0.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class RuntimeResponseSummary:
    text: Optional[str]
    latency_ms: Optional[float]
    safety: Optional[Dict[str, Any]]
    controls: Optional[Dict[str, Any]]
    retrieval_summary: Optional[Dict[str, Any]]
    perception_summary: Optional[Dict[str, Any]]

    @classmethod
    def from_response(cls, response: Any) -> Optional["RuntimeResponseSummary"]:
        if response is None:
            return None
        latency = getattr(response, "latency_ms", None)
        return cls(
            text=getattr(response, "text", None),
            latency_ms=None if latency is None else float(latency),
            safety=getattr(response, "safety", None),
            controls=getattr(response, "controls_used", None),
            retrieval_summary=getattr(response, "retrieval_summary", None),
            perception_summary=getattr(response, "perception_summary", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "safety": self.safety,
            "controls": self.controls,
            "retrieval_summary": self.retrieval_summary,
            "perception_summary": self.perception_summary,
        }


@dataclass
class RuntimeTurnResult:
    talk_mode: Optional[str]
    response_route: Optional[str]
    metrics: Dict[str, Any]
    persona_meta: Dict[str, Any]
    heart: Dict[str, Any]
    shadow: Optional[Dict[str, Any]]
    qualia_gate: Dict[str, Any]
    affect: Optional[RuntimeAffectSummary]
    response: Optional[RuntimeResponseSummary]
    memory_reference: Optional[Dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeTurnResult":
        return cls(
            talk_mode=payload.get("talk_mode"),
            response_route=payload.get("response_route"),
            metrics=dict(payload.get("metrics") or {}),
            persona_meta=dict(payload.get("persona_meta") or {}),
            heart=dict(payload.get("heart") or {}),
            shadow=payload.get("shadow"),
            qualia_gate=dict(payload.get("qualia_gate") or {}),
            affect=RuntimeAffectSummary.from_sample(payload.get("affect")),
            response=RuntimeResponseSummary.from_response(payload.get("response")),
            memory_reference=payload.get("memory_reference"),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "talk_mode": self.talk_mode,
            "response_route": self.response_route,
            "metrics": self.metrics,
            "persona_meta": self.persona_meta,
            "heart": self.heart,
            "shadow": self.shadow,
            "qualia_gate": self.qualia_gate,
            "response": self.response.to_dict() if self.response is not None else None,
        }
        if self.affect is not None:
            payload["affect"] = self.affect.to_dict()
        if self.memory_reference is not None:
            payload["memory_reference"] = self.memory_reference
        return payload


