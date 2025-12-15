from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

WORKSPACE_EVENT_NAME = "workspace.snapshot_v1"
INTERVENTION_EVENT_NAME = "intervention.applied_v1"

TelemetryEmitter = Callable[..., Mapping[str, Any] | None]


class FieldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    S_norm: float = Field(..., ge=0.0, le=1.0)
    H_norm: float = Field(..., ge=0.0, le=1.0)
    rho_norm: float = Field(..., ge=0.0, le=1.0)
    Ignition: float = Field(..., ge=0.0)
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    gate_state: str = Field(..., description="forward/reverse/idle")
    gate_level: int = Field(0, ge=0)
    S_raw: Optional[float] = None
    H_raw: Optional[float] = None


class RhythmSingle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R: float = Field(..., ge=0.0, le=1.0)
    psi: Optional[float] = None
    rho: Optional[float] = None
    I: Optional[float] = None
    q: Optional[float] = None


class RhythmDual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dpsi: Optional[float] = None
    mismatch: Optional[float] = None


class RhythmSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    self: RhythmSingle
    other: Optional[RhythmSingle] = None
    eff: RhythmSingle
    dual: Optional[RhythmDual] = None


class MemoryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replayed: bool = False
    memory_id: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None


class PerceptionBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    vision: Dict[str, Any] = Field(default_factory=dict)
    audio: Dict[str, Any] = Field(default_factory=dict)
    interoception: Dict[str, Any] = Field(default_factory=dict)


class GoalWeight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float


class WorkspaceValueState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goals: Sequence[GoalWeight] = Field(default_factory=list)
    risk: Optional[float] = None


class WorkspaceActionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float
    reason_codes: Sequence[str] = Field(default_factory=list)


class WorkspaceActionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    expected_effect: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perception: PerceptionBlock = Field(default_factory=PerceptionBlock)
    value: WorkspaceValueState = Field(default_factory=WorkspaceValueState)
    action_candidates: Sequence[WorkspaceActionCandidate] = Field(default_factory=list)
    selected_action: Optional[WorkspaceActionSelection] = None


class ControlOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    pause_ms: Optional[float] = None
    priority: Optional[float] = None


class InterventionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: bool = False
    ids: Sequence[str] = Field(default_factory=list)


class WorkspaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = Field(default=WORKSPACE_EVENT_NAME, frozen=True)
    schema_version: int = Field(default=1, ge=1)
    timestamp_ms: int
    stage: str = "runtime"
    step: int = 0
    source_loop: Optional[str] = None
    pid: Optional[int] = None
    field: FieldState
    rhythm: RhythmSnapshot
    memory_ref: Optional[MemoryReference] = None
    workspace: WorkspaceState
    control_out: Optional[ControlOutputs] = None
    intervention: Optional[InterventionStatus] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceSnapshot(BaseModel):
    event: str = Field(default=WORKSPACE_EVENT_NAME, frozen=True)
    schema_version: int = Field(default=1, ge=1)
    timestamp_ms: int
    turn_id: str
    stage: str = "runtime"
    step: int = 0
    source_loop: Optional[str] = None
    pid: Optional[int] = None
    field: FieldState
    rhythm: RhythmSnapshot
    memory_ref: Optional[MemoryReference] = None
    workspace: WorkspaceState
    control_out: Optional[ControlOutputs] = None
    intervention: Optional[InterventionStatus] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class InterventionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = Field(default=INTERVENTION_EVENT_NAME, frozen=True)
    schema_version: int = Field(default=1, ge=1)
    timestamp_ms: int
    id: str
    type: str
    name: str
    target: Optional[str] = None
    magnitude: Optional[float] = None
    duration_ms: Optional[int] = None
    seed: Optional[int] = None
    reason: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


def _resolve_emitter(telemetry_hook: Optional[TelemetryEmitter]) -> TelemetryEmitter:
    if telemetry_hook is not None:
        return telemetry_hook
    try:
        from telemetry.event import event as telemetry_event  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("telemetry_event is not available; pass telemetry_hook explicitly") from exc
    return telemetry_event


def emit_workspace_snapshot(
    snapshot: WorkspaceSnapshot,
    *,
    telemetry_hook: Optional[TelemetryEmitter] = None,
    log_path: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    emitter = _resolve_emitter(telemetry_hook)
    payload = snapshot.model_dump(mode="json")
    try:
        return emitter(snapshot.event, payload, log_path=log_path)  # type: ignore[arg-type]
    except TypeError:
        return emitter(snapshot.event, payload)


def emit_intervention_event(
    event_model: InterventionEvent,
    *,
    telemetry_hook: Optional[TelemetryEmitter] = None,
    log_path: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    emitter = _resolve_emitter(telemetry_hook)
    payload = event_model.model_dump(mode="json")
    try:
        return emitter(event_model.event, payload, log_path=log_path)  # type: ignore[arg-type]
    except TypeError:
        return emitter(event_model.event, payload)
