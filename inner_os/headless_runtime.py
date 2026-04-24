from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_BASE_WAIT_MS = {
    "brief": 80,
    "measured": 180,
    "extended": 320,
    "held": 460,
}

_TIMING_BIAS_WAIT_MS = {
    "gentle_overlap": 40,
    "turn_entry": 60,
    "just_after_turn": 70,
    "near_turn_end": 90,
    "wait": 420,
}


@dataclass(frozen=True)
class TurnTimingHint:
    response_channel: str = "speak"
    wait_before_action: str = "brief"
    timing_bias: str = ""
    entry_window: str = "ready"
    pause_profile: str = "none"
    overlap_policy: str = "follow_turn"
    interruptibility: str = "medium"
    minimum_wait_ms: int = 0
    interrupt_guard_ms: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "response_channel": self.response_channel,
            "wait_before_action": self.wait_before_action,
            "timing_bias": self.timing_bias,
            "entry_window": self.entry_window,
            "pause_profile": self.pause_profile,
            "overlap_policy": self.overlap_policy,
            "interruptibility": self.interruptibility,
            "minimum_wait_ms": int(self.minimum_wait_ms),
            "interrupt_guard_ms": int(self.interrupt_guard_ms),
        }


@dataclass(frozen=True)
class EmitTimingContract:
    response_channel: str = ""
    entry_window: str = ""
    pause_profile: str = ""
    overlap_policy: str = ""
    interruptibility: str = ""
    minimum_wait_ms: float = 0.0
    interrupt_guard_ms: float = 0.0
    effective_emit_delay_ms: float = 0.0
    effective_latency_ms: float = 0.0
    emit_not_before_ms: float = 0.0
    interrupt_guard_until_ms: float = 0.0
    wait_applied: bool = False
    wait_applied_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "response_channel": self.response_channel,
            "entry_window": self.entry_window,
            "pause_profile": self.pause_profile,
            "overlap_policy": self.overlap_policy,
            "interruptibility": self.interruptibility,
            "minimum_wait_ms": round(self.minimum_wait_ms, 4),
            "interrupt_guard_ms": round(self.interrupt_guard_ms, 4),
            "effective_emit_delay_ms": round(self.effective_emit_delay_ms, 4),
            "effective_latency_ms": round(self.effective_latency_ms, 4),
            "emit_not_before_ms": round(self.emit_not_before_ms, 4),
            "interrupt_guard_until_ms": round(self.interrupt_guard_until_ms, 4),
            "wait_applied": self.wait_applied,
            "wait_applied_ms": round(self.wait_applied_ms, 4),
        }


@dataclass(frozen=True)
class TimingGuardState:
    active: bool = False
    reason: str = "idle"
    response_channel: str = ""
    overlap_policy: str = ""
    emit_not_before_ms: float = 0.0
    interrupt_guard_until_ms: float = 0.0
    voice_conflict: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "reason": self.reason,
            "response_channel": self.response_channel,
            "overlap_policy": self.overlap_policy,
            "emit_not_before_ms": round(self.emit_not_before_ms, 4),
            "interrupt_guard_until_ms": round(self.interrupt_guard_until_ms, 4),
            "voice_conflict": self.voice_conflict,
        }


def coerce_turn_timing_hint(value: Mapping[str, Any] | TurnTimingHint | None) -> TurnTimingHint:
    if isinstance(value, TurnTimingHint):
        return value
    payload = dict(value or {})
    return TurnTimingHint(
        response_channel=str(payload.get("response_channel") or "speak").strip().lower() or "speak",
        wait_before_action=str(payload.get("wait_before_action") or "brief").strip().lower() or "brief",
        timing_bias=str(payload.get("timing_bias") or "").strip().lower(),
        entry_window=str(payload.get("entry_window") or "ready").strip() or "ready",
        pause_profile=str(payload.get("pause_profile") or "none").strip() or "none",
        overlap_policy=str(payload.get("overlap_policy") or "follow_turn").strip() or "follow_turn",
        interruptibility=str(payload.get("interruptibility") or "medium").strip() or "medium",
        minimum_wait_ms=int(max(float(payload.get("minimum_wait_ms") or 0.0), 0.0)),
        interrupt_guard_ms=int(max(float(payload.get("interrupt_guard_ms") or 0.0), 0.0)),
    )


def coerce_emit_timing_contract(
    value: Mapping[str, Any] | EmitTimingContract | None,
) -> EmitTimingContract:
    if isinstance(value, EmitTimingContract):
        return value
    payload = dict(value or {})
    return EmitTimingContract(
        response_channel=str(payload.get("response_channel") or "").strip().lower(),
        entry_window=str(payload.get("entry_window") or "").strip(),
        pause_profile=str(payload.get("pause_profile") or "").strip(),
        overlap_policy=str(payload.get("overlap_policy") or "").strip(),
        interruptibility=str(payload.get("interruptibility") or "").strip(),
        minimum_wait_ms=max(float(payload.get("minimum_wait_ms") or 0.0), 0.0),
        interrupt_guard_ms=max(float(payload.get("interrupt_guard_ms") or 0.0), 0.0),
        effective_emit_delay_ms=max(float(payload.get("effective_emit_delay_ms") or 0.0), 0.0),
        effective_latency_ms=max(float(payload.get("effective_latency_ms") or 0.0), 0.0),
        emit_not_before_ms=max(float(payload.get("emit_not_before_ms") or 0.0), 0.0),
        interrupt_guard_until_ms=max(float(payload.get("interrupt_guard_until_ms") or 0.0), 0.0),
        wait_applied=bool(payload.get("wait_applied", False)),
        wait_applied_ms=max(float(payload.get("wait_applied_ms") or 0.0), 0.0),
    )


def coerce_timing_guard_state(
    value: Mapping[str, Any] | TimingGuardState | None,
) -> TimingGuardState:
    if isinstance(value, TimingGuardState):
        return value
    payload = dict(value or {})
    return TimingGuardState(
        active=bool(payload.get("active", False)),
        reason=str(payload.get("reason") or "idle").strip() or "idle",
        response_channel=str(payload.get("response_channel") or "").strip().lower(),
        overlap_policy=str(payload.get("overlap_policy") or "").strip(),
        emit_not_before_ms=max(float(payload.get("emit_not_before_ms") or 0.0), 0.0),
        interrupt_guard_until_ms=max(float(payload.get("interrupt_guard_until_ms") or 0.0), 0.0),
        voice_conflict=bool(payload.get("voice_conflict", False)),
    )


def derive_turn_timing_hint(*, actuation_plan: Mapping[str, Any] | None) -> TurnTimingHint:
    plan = dict(actuation_plan or {})
    response_channel = str(plan.get("response_channel") or "speak").strip().lower()
    wait_before_action = str(plan.get("wait_before_action") or "brief").strip().lower()
    nonverbal_response_state = dict(plan.get("nonverbal_response_state") or {})
    timing_bias = str(nonverbal_response_state.get("timing_bias") or "").strip().lower()

    base_wait_ms = int(_BASE_WAIT_MS.get(wait_before_action, _BASE_WAIT_MS["brief"]))
    bias_wait_ms = int(_TIMING_BIAS_WAIT_MS.get(timing_bias, 0))
    entry_window = {
        "brief": "ready",
        "measured": "measured",
        "extended": "held",
        "held": "held",
    }.get(wait_before_action, "ready")
    if timing_bias in {"gentle_overlap", "turn_entry", "just_after_turn", "near_turn_end"}:
        entry_window = "ready"
    elif timing_bias == "wait":
        entry_window = "held"

    pause_profile = "soft_pause" if wait_before_action in {"extended", "held"} else "none"
    overlap_policy = "follow_turn"
    interruptibility = "medium"
    minimum_wait_ms = base_wait_ms
    interrupt_guard_ms = max(80, base_wait_ms)

    if response_channel == "backchannel":
        pause_profile = "none"
        overlap_policy = "allow_soft_overlap" if timing_bias == "gentle_overlap" else "yield_to_user_release"
        interruptibility = "high"
        minimum_wait_ms = min(base_wait_ms, bias_wait_ms or 110)
        interrupt_guard_ms = 90 if timing_bias == "gentle_overlap" else 140
    elif response_channel == "hold":
        pause_profile = "soft_pause"
        overlap_policy = "wait_for_release"
        interruptibility = "low"
        minimum_wait_ms = max(base_wait_ms, bias_wait_ms or 280)
        interrupt_guard_ms = max(240, minimum_wait_ms)
    elif response_channel == "defer":
        pause_profile = "soft_pause"
        overlap_policy = "wait_for_release"
        interruptibility = "low"
        minimum_wait_ms = max(base_wait_ms, bias_wait_ms or 220)
        interrupt_guard_ms = max(180, minimum_wait_ms)
    elif timing_bias == "gentle_overlap":
        overlap_policy = "allow_soft_overlap"
        interruptibility = "high"
        minimum_wait_ms = min(base_wait_ms, bias_wait_ms or 60)
        interrupt_guard_ms = 80
    elif timing_bias == "wait":
        pause_profile = "soft_pause"
        overlap_policy = "wait_for_release"
        interruptibility = "low"
        minimum_wait_ms = max(base_wait_ms, bias_wait_ms or 320)
        interrupt_guard_ms = max(180, minimum_wait_ms)

    return TurnTimingHint(
        response_channel=response_channel,
        wait_before_action=wait_before_action,
        timing_bias=timing_bias,
        entry_window=entry_window,
        pause_profile=pause_profile,
        overlap_policy=overlap_policy,
        interruptibility=interruptibility,
        minimum_wait_ms=minimum_wait_ms,
        interrupt_guard_ms=interrupt_guard_ms,
    )


@dataclass
class HeadlessTurnResult:
    execution_mode: str
    primary_action: str
    action_queue: list[str] = field(default_factory=list)
    response_channel: str = "speak"
    response_channel_score: float = 0.0
    reply_permission: str = "speak"
    wait_before_action: str = "brief"
    repair_window_commitment: str = "soft"
    outcome_goal: str = ""
    boundary_mode: str = ""
    attention_target: str = ""
    memory_write_priority: str = ""
    nonverbal_response_state: dict[str, object] = field(default_factory=dict)
    presence_hold_state: dict[str, object] = field(default_factory=dict)
    turn_timing_hint: TurnTimingHint | Mapping[str, Any] = field(default_factory=TurnTimingHint)
    do_not_cross: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "primary_action": self.primary_action,
            "action_queue": list(self.action_queue),
            "response_channel": self.response_channel,
            "response_channel_score": float(self.response_channel_score),
            "reply_permission": self.reply_permission,
            "wait_before_action": self.wait_before_action,
            "repair_window_commitment": self.repair_window_commitment,
            "outcome_goal": self.outcome_goal,
            "boundary_mode": self.boundary_mode,
            "attention_target": self.attention_target,
            "memory_write_priority": self.memory_write_priority,
            "nonverbal_response_state": dict(self.nonverbal_response_state),
            "presence_hold_state": dict(self.presence_hold_state),
            "turn_timing_hint": coerce_turn_timing_hint(self.turn_timing_hint).to_dict(),
            "do_not_cross": list(self.do_not_cross),
        }


class HeadlessInnerOSRuntime:
    """LLM を使わずに actuation plan を実行用の最小状態へ落とす。"""

    def step(self, *, actuation_plan: Mapping[str, Any] | None) -> HeadlessTurnResult:
        plan = dict(actuation_plan or {})
        turn_timing_hint = derive_turn_timing_hint(actuation_plan=plan)
        return HeadlessTurnResult(
            execution_mode=str(plan.get("execution_mode") or "attuned_contact"),
            primary_action=str(plan.get("primary_action") or "hold_presence"),
            action_queue=[str(item) for item in plan.get("action_queue") or [] if str(item).strip()],
            response_channel=str(plan.get("response_channel") or "speak"),
            response_channel_score=float(plan.get("response_channel_score") or 0.0),
            reply_permission=str(plan.get("reply_permission") or "speak"),
            wait_before_action=str(plan.get("wait_before_action") or "brief"),
            repair_window_commitment=str(plan.get("repair_window_commitment") or "soft"),
            outcome_goal=str(plan.get("outcome_goal") or ""),
            boundary_mode=str(plan.get("boundary_mode") or ""),
            attention_target=str(plan.get("attention_target") or ""),
            memory_write_priority=str(plan.get("memory_write_priority") or ""),
            nonverbal_response_state=dict(plan.get("nonverbal_response_state") or {}),
            presence_hold_state=dict(plan.get("presence_hold_state") or {}),
            turn_timing_hint=turn_timing_hint,
            do_not_cross=[str(item) for item in plan.get("do_not_cross") or [] if str(item).strip()],
        )
