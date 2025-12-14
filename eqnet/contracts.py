from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Literal

try:
    from eqnet.runtime.policy import PolicyPrior
except Exception:  # pragma: no cover - fallback during partial installs
    PolicyPrior = Any  # type: ignore

try:
    from eqnet.runtime.state import QualiaState
except Exception:  # pragma: no cover - fallback during partial installs
    QualiaState = Any  # type: ignore

ContextMode = Literal[
    "casual",
    "business",
    "crisis",
    "call_center",
    "political_speech",
    "stage_performance",
]

ActionType = Literal["none", "speak", "suggest", "ask", "escalate", "pause"]


@dataclass
class SomaticSignals:
    arousal_hint: float = 0.0
    stress_hint: float = 0.0
    fatigue_hint: float = 0.0
    jitter: float = 0.0
    proximity: float = 1.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialContext:
    mode: ContextMode = "casual"
    cultural_pressure: float = 0.0
    offer_requested: bool = False
    disclosure_budget: float = 1.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldSummary:
    hazard_level: float = 0.0
    ambiguity: float = 0.0
    npc_affect: float = 0.0
    social_pressure: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptPacket:
    turn_id: str
    timestamp_ms: int
    user_text: Optional[str] = None
    somatic: SomaticSignals = field(default_factory=SomaticSignals)
    context: SocialContext = field(default_factory=SocialContext)
    world: WorldSummary = field(default_factory=WorldSummary)
    seed: Optional[int] = None
    scenario_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SpeechPlan:
    text: str
    tone: str = "neutral"
    directiveness: float = 0.0
    disclosure: float = 0.0
    fillers_enabled: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionIntent:
    action: ActionType = "speak"
    confidence: float = 0.5
    rationale: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryDelta:
    should_write_diary: bool = False
    diary_tags: List[str] = field(default_factory=list)
    salience: float = 0.0
    qualia_delta: Dict[str, float] = field(default_factory=dict)


@dataclass
class TraceSchema:
    schema_version: str = "trace_v1"
    source_loop: Optional[str] = None
    scenario_id: Optional[str] = None
    turn_id: Optional[str] = None
    seed: Optional[int] = None
    timestamp_ms: Optional[int] = None
    boundary: Dict[str, Any] = field(default_factory=dict)
    self_state: Dict[str, Any] = field(default_factory=dict)
    prospection: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    qualia: Dict[str, Any] = field(default_factory=dict)
    invariants: Dict[str, Any] = field(default_factory=dict)

    def ensure_minimums(self) -> None:
        self.boundary = dict(self.boundary or {})
        self.self_state = dict(self.self_state or {})
        self.prospection = dict(self.prospection or {})
        self.policy = dict(self.policy or {})
        self.qualia = dict(self.qualia or {})
        self.invariants = dict(self.invariants or {})

    def to_dict(self) -> Dict[str, Any]:
        self.ensure_minimums()
        return {
            "schema_version": self.schema_version,
            "source_loop": self.source_loop,
            "scenario_id": self.scenario_id,
            "turn_id": self.turn_id,
            "seed": self.seed,
            "timestamp_ms": self.timestamp_ms,
            "boundary": self.boundary,
            "self": self.self_state,
            "prospection": self.prospection,
            "policy": self.policy,
            "qualia": self.qualia,
            "invariants": self.invariants,
        }


@dataclass
class TurnResult:
    turn_id: str
    intent: ActionIntent
    speech: Optional[SpeechPlan] = None
    memory: MemoryDelta = field(default_factory=MemoryDelta)
    trace: TraceSchema = field(default_factory=TraceSchema)


__all__ = [
    "PerceptPacket",
    "SomaticSignals",
    "SocialContext",
    "WorldSummary",
    "ActionIntent",
    "SpeechPlan",
    "MemoryDelta",
    "TurnResult",
    "TraceSchema",
]
