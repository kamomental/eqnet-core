from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Mapping, Tuple

from .emotion import EmotionVector, ValueGradient
from .talk_mode import TalkMode


class ResponseRoute(str, Enum):
    """Marks which route (reflex/habit/conscious) produced a response."""

    REFLEX = "reflex"
    HABIT = "habit"
    CONSCIOUS = "conscious"


class SelfLayer(str, Enum):
    """Which layer of self led the behaviour for a turn."""

    REFLEX = "reflex"
    AFFECTIVE = "affective"
    NARRATIVE = "narrative"


def _coerce_force_value(
    payload: Mapping[str, Any] | None, key: str, default: float = 0.0
) -> float:
    if not payload or key not in payload:
        return default
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return default


@dataclass
class LayerForceRow:
    survival: float = 0.0
    physiological: float = 0.0
    social: float = 0.0
    exploration: float = 0.0
    attachment: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "LayerForceRow":
        return cls(
            survival=_coerce_force_value(payload, "survival"),
            physiological=_coerce_force_value(payload, "physiological"),
            social=_coerce_force_value(payload, "social"),
            exploration=_coerce_force_value(payload, "exploration"),
            attachment=_coerce_force_value(payload, "attachment"),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "survival": float(self.survival),
            "physiological": float(self.physiological),
            "social": float(self.social),
            "exploration": float(self.exploration),
            "attachment": float(self.attachment),
        }

    def scaled(self, multiplier: float) -> "LayerForceRow":
        factor = float(multiplier)
        return LayerForceRow(
            survival=self.survival * factor,
            physiological=self.physiological * factor,
            social=self.social * factor,
            exploration=self.exploration * factor,
            attachment=self.attachment * factor,
        )


@dataclass
class ForceMatrix:
    reflex: LayerForceRow = field(default_factory=LayerForceRow)
    affective: LayerForceRow = field(default_factory=LayerForceRow)
    narrative: LayerForceRow = field(default_factory=LayerForceRow)

    @staticmethod
    def _normalize_layer_key(name: Any) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().lower()
        if key in {SelfLayer.REFLEX.value, "reflex"}:
            return "reflex"
        if key in {SelfLayer.AFFECTIVE.value, "affective"}:
            return "affective"
        if key in {SelfLayer.NARRATIVE.value, "narrative"}:
            return "narrative"
        return None

    @staticmethod
    def _normalize_axis_key(name: Any) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().lower()
        mapping = {
            "survival": "survival",
            "survival_bias": "survival",
            "physiological": "physiological",
            "physio": "physiological",
            "physiological_bias": "physiological",
            "social": "social",
            "social_bias": "social",
            "exploration": "exploration",
            "exploration_bias": "exploration",
            "attachment": "attachment",
            "attachment_bias": "attachment",
        }
        return mapping.get(key)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ForceMatrix":
        rows = {
            "reflex": LayerForceRow(),
            "affective": LayerForceRow(),
            "narrative": LayerForceRow(),
        }
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                layer_key = cls._normalize_layer_key(key)
                if not layer_key or not isinstance(value, Mapping):
                    continue
                rows[layer_key] = LayerForceRow.from_mapping(value)
        return cls(
            reflex=rows["reflex"],
            affective=rows["affective"],
            narrative=rows["narrative"],
        )

    def merge(self, overrides: Mapping[str, Any] | None) -> "ForceMatrix":
        if not overrides or not isinstance(overrides, Mapping):
            return ForceMatrix.from_mapping(self.to_dict())
        base = {
            "reflex": dict(self.reflex.to_dict()),
            "affective": dict(self.affective.to_dict()),
            "narrative": dict(self.narrative.to_dict()),
        }
        for layer_key, layer_payload in overrides.items():
            norm_layer = self._normalize_layer_key(layer_key)
            if not norm_layer or not isinstance(layer_payload, Mapping):
                continue
            row = base.setdefault(norm_layer, {})
            for axis, value in layer_payload.items():
                norm_axis = self._normalize_axis_key(axis)
                if not norm_axis:
                    continue
                row[norm_axis] = _coerce_force_value(layer_payload, axis, row.get(norm_axis, 0.0))
        return ForceMatrix.from_mapping(base)

    def row_for(self, layer: SelfLayer) -> LayerForceRow:
        if layer == SelfLayer.AFFECTIVE:
            return self.affective
        if layer == SelfLayer.NARRATIVE:
            return self.narrative
        return self.reflex

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "reflex": self.reflex.to_dict(),
            "affective": self.affective.to_dict(),
            "narrative": self.narrative.to_dict(),
        }


@dataclass
class SelfForceSnapshot:
    reflex: float
    affective: float
    narrative: float
    winner: Optional[SelfLayer] = None
    winner_margin: Optional[float] = None
    is_tie: Optional[bool] = None

    def _ranked(self) -> List[Tuple[SelfLayer, float]]:
        values = [
            (SelfLayer.REFLEX, float(self.reflex)),
            (SelfLayer.AFFECTIVE, float(self.affective)),
            (SelfLayer.NARRATIVE, float(self.narrative)),
        ]
        return sorted(values, key=lambda kv: kv[1], reverse=True)

    def with_winner(self, *, tie_eps: float = 1e-6) -> "SelfForceSnapshot":
        if (
            self.winner is not None
            and self.winner_margin is not None
            and self.is_tie is not None
        ):
            return self
        ranked = self._ranked()
        top_layer, top_val = ranked[0]
        _, second_val = ranked[1]
        margin = float(top_val - second_val)
        is_tie = bool(margin <= float(tie_eps))
        self.winner_margin = margin
        self.is_tie = is_tie
        self.winner = None if is_tie else top_layer
        return self

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reflex": float(self.reflex),
            "affective": float(self.affective),
            "narrative": float(self.narrative),
        }
        if self.winner is not None:
            payload["winner"] = self.winner.value
        if self.winner_margin is not None:
            payload["winner_margin"] = float(self.winner_margin)
        if self.is_tie is not None:
            payload["is_tie"] = bool(self.is_tie)
        return payload


@dataclass
class BoundarySignal:
    """Represents a detected boundary transition strength and its contributors."""

    score: float
    sources: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "sources": {k: float(v) for k, v in self.sources.items()},
        }


BOUNDARY_SOURCE_KEYS: Tuple[str, ...] = (
    "pred_error_delta",
    "force_delta",
    "winner_flip",
    "raw_damped_gap",
    "latency_penalty",
    "tag_bump",
)


@dataclass
class ResetEvent:
    """Lightweight record of a runtime reset performed around a boundary."""

    reason: str
    targets: List[str]
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reason": self.reason,
            "targets": list(self.targets),
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


RESET_TARGET_SCRATCHPAD = "scratchpad"
RESET_TARGET_AFFECTIVE_ECHO = "affective_echo"
RESET_TARGET_FORCE_CACHE = "force_cache"
RESET_TARGET_RECENT_TAG_BUMPS = "recent_tag_bumps"


@dataclass
class ReflexTraits:
    safety_reflex_bias: float = 0.5
    startle_reactivity: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "safety_reflex_bias": float(self.safety_reflex_bias),
            "startle_reactivity": float(self.startle_reactivity),
        }


@dataclass
class EmotionTraits:
    baseline_valence: float = 0.5
    baseline_arousal: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline_valence": float(self.baseline_valence),
            "baseline_arousal": float(self.baseline_arousal),
        }


@dataclass
class NarrativeTraits:
    self_story_tags: List[str]
    identity_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_story_tags": list(self.self_story_tags),
            "identity_confidence": float(self.identity_confidence),
        }


@dataclass
class LayeredSelf:
    reflex_self: ReflexTraits
    affective_self: EmotionTraits
    narrative_self: NarrativeTraits

    @staticmethod
    def default() -> "LayeredSelf":
        return LayeredSelf(
            reflex_self=ReflexTraits(safety_reflex_bias=0.6, startle_reactivity=0.5),
            affective_self=EmotionTraits(baseline_valence=0.5, baseline_arousal=0.5),
            narrative_self=NarrativeTraits(self_story_tags=["neutral"], identity_confidence=0.5),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflex_self": self.reflex_self.to_dict(),
            "affective_self": self.affective_self.to_dict(),
            "narrative_self": self.narrative_self.to_dict(),
        }


@dataclass
class ImplementationContext:
    hardware_profile: str
    latency_ms: float
    memory_load: float
    sensor_fidelity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_profile": self.hardware_profile,
            "latency_ms": float(self.latency_ms),
            "memory_load": float(self.memory_load),
            "sensor_fidelity": float(self.sensor_fidelity),
        }


@dataclass
class SelfModel:
    """Persisted notion of "self as an object" (traits, energy, ties)."""

    role_labels: List[str]
    long_term_traits: Dict[str, float]
    current_mode: TalkMode
    current_energy: float
    attachment_to_user: float
    layered_self: Optional[LayeredSelf] = None

    def snapshot(self) -> "SelfModelSnapshot":
        layered = self.layered_self or LayeredSelf.default()
        self.layered_self = layered
        return SelfModelSnapshot(
            role_labels=list(self.role_labels),
            long_term_traits=dict(self.long_term_traits),
            current_mode=self.current_mode,
            current_energy=float(self.current_energy),
            attachment_to_user=float(self.attachment_to_user),
            layered_self=layered,
        )


@dataclass
class SelfModelSnapshot:
    role_labels: List[str]
    long_term_traits: Dict[str, float]
    current_mode: TalkMode
    current_energy: float
    attachment_to_user: float
    layered_self: LayeredSelf

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["current_mode"] = self.current_mode.value
        payload["layered_self"] = self.layered_self.to_dict()
        return payload


@dataclass
class WorldStateSnapshot:
    """Lightweight summary of "how the world looked" in that moment."""

    summary_text: str
    salient_entities: List[str]
    context_tags: List[str]
    prediction_error: float
    hazard_score: float = 0.0
    hazard_sources: Optional[List[str]] = None
    flags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.hazard_sources is None:
            self.hazard_sources = []
        if self.flags is None:
            self.flags = []

    def importance(self) -> float:
        base = float(max(self.prediction_error, 0.0))
        return base + float(max(self.hazard_score, 0.0)) + 0.1 * len(self.salient_entities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "salient_entities": list(self.salient_entities),
            "context_tags": list(self.context_tags),
            "prediction_error": float(self.prediction_error),
            "hazard_score": float(self.hazard_score),
            "hazard_sources": list(self.hazard_sources or []),
            "flags": list(self.flags or []),
        }

    @staticmethod
    def from_context(context: Mapping[str, Any] | None, *, prediction_error: float) -> "WorldStateSnapshot":
        ctx = context or {}
        return WorldStateSnapshot(
            summary_text=str(ctx.get("world_summary", "")),
            salient_entities=list(ctx.get("salient_entities", []) or []),
            context_tags=list(ctx.get("context_tags", []) or []),
            prediction_error=prediction_error,
            hazard_score=float(ctx.get("hazard_score", 0.0) or 0.0),
            hazard_sources=list(ctx.get("hazard_sources", []) or []),
            flags=list(ctx.get("flags", []) or []),
        )


@dataclass
class ConsciousEpisode:
    id: str
    timestamp: datetime
    self_state: SelfModelSnapshot
    world_state: WorldStateSnapshot
    qualia: EmotionVector
    narrative: Optional[str] = None
    route: ResponseRoute = ResponseRoute.CONSCIOUS
    value_gradient: Optional[ValueGradient] = None
    dominant_self_layer: Optional[SelfLayer] = None
    implementation: Optional[ImplementationContext] = None
    self_force: Optional[SelfForceSnapshot] = None
    raw_self_force: Optional[SelfForceSnapshot] = None
    boundary_signal: Optional[BoundarySignal] = None
    reset_event: Optional[ResetEvent] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "self_state": self.self_state.to_dict(),
            "world_state": self.world_state.to_dict(),
            "qualia": self.qualia.to_dict(),
            "narrative": self.narrative,
            "route": self.route.value,
            "value_gradient": self.value_gradient.to_dict() if self.value_gradient else None,
            "dominant_self_layer": self.dominant_self_layer.value if self.dominant_self_layer else None,
            "implementation": self.implementation.to_dict() if self.implementation else None,
            "self_force": self.self_force.to_dict() if self.self_force else None,
            "raw_self_force": self.raw_self_force.to_dict() if self.raw_self_force else None,
            "boundary_signal": self.boundary_signal.to_dict() if self.boundary_signal else None,
            "reset_event": self.reset_event.to_dict() if self.reset_event else None,
        }

    def salience(self) -> float:
        return self.world_state.importance() + self.qualia.salience_score()

