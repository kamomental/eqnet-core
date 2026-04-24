from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import emot_terrain_lab.hub.runtime as runtime_module
import numpy as np
from emot_terrain_lab.terrain import llm as terrain_llm

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, _apply_inner_os_expression_controls
from inner_os.action_posture import coerce_action_posture_contract
from inner_os.actuation_plan import coerce_actuation_plan_contract
from inner_os.expression import build_expression_hints_from_gate_result
from inner_os.expression_hint_bundles import (
    FieldRegulationHintBundleContract,
    InteractionAuditHintBundleContract,
    InteractionReasoningHintBundleContract,
    QualiaHintBundleContract,
    SceneHintBundleContract,
    TerrainInsightHintBundleContract,
    WorkspaceHintBundleContract,
)
from inner_os.expression.surface_context_packet import coerce_surface_context_packet
from inner_os.headless_runtime import TimingGuardState
from inner_os.integration_hooks import IntegrationHooks
from inner_os.headless_runtime import HeadlessInnerOSRuntime, HeadlessTurnResult
from inner_os.memory_core import MemoryCore
from inner_os.policy_packet import coerce_interaction_policy_packet


@dataclass
class _SensorSnapshot:
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _SensorHarness:
    snapshot: _SensorSnapshot


@dataclass
class _SelfHarness:
    current_energy: float = 0.8


@dataclass
class _ProcessTurnHarness:
    _integration_hooks: IntegrationHooks
    _inner_os_headless_runtime: HeadlessInnerOSRuntime = field(default_factory=HeadlessInnerOSRuntime)
    _runtime_sensors: _SensorHarness = field(
        default_factory=lambda: _SensorHarness(_SensorSnapshot({"body_stress_index": 0.25, "activity_level": 0.5}))
    )
    _last_shadow_estimate: Optional[Dict[str, Any]] = None
    _last_gate_context: Dict[str, Any] = field(default_factory=dict)
    _last_inner_os_timing_guard: TimingGuardState = field(default_factory=TimingGuardState)
    _inner_os_emit_not_before_ms: float = 0.0
    _inner_os_interrupt_guard_until_ms: float = 0.0
    _inner_os_emit_overlap_policy: str = ""
    _inner_os_emit_response_channel: str = ""
    _self_model: _SelfHarness = field(default_factory=_SelfHarness)
    _last_step_context: Optional[str] = None
    _surface_user_history: Any = field(default_factory=lambda: deque(maxlen=4))
    _surface_world_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "zone_id": "market",
            "mode": "reality",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        }
    )

    process_turn = EmotionalHubRuntime.process_turn
    _normalize_perception_summary = EmotionalHubRuntime._normalize_perception_summary
    _visual_reflection_line = EmotionalHubRuntime._visual_reflection_line
    _attach_visual_reflection = EmotionalHubRuntime._attach_visual_reflection
    _inner_os_relational_context = EmotionalHubRuntime._inner_os_relational_context
    _inner_os_working_memory_seed = EmotionalHubRuntime._inner_os_working_memory_seed
    _merge_conscious_long_term_theme = EmotionalHubRuntime._merge_conscious_long_term_theme
    _latest_nightly_long_term_theme_summary = EmotionalHubRuntime._latest_nightly_long_term_theme_summary
    _latest_nightly_identity_arc_summary = EmotionalHubRuntime._latest_nightly_identity_arc_summary
    _latest_nightly_identity_arc_registry_summary = EmotionalHubRuntime._latest_nightly_identity_arc_registry_summary
    _latest_nightly_relation_arc_summary = EmotionalHubRuntime._latest_nightly_relation_arc_summary
    _latest_nightly_relation_arc_registry_summary = EmotionalHubRuntime._latest_nightly_relation_arc_registry_summary
    _latest_nightly_group_relation_arc_summary = EmotionalHubRuntime._latest_nightly_group_relation_arc_summary
    _latest_nightly_partner_relation_summary = EmotionalHubRuntime._latest_nightly_partner_relation_summary
    _latest_nightly_partner_relation_registry_summary = EmotionalHubRuntime._latest_nightly_partner_relation_registry_summary
    _apply_inner_os_surface_policy = EmotionalHubRuntime._apply_inner_os_surface_policy
    _apply_inner_os_surface_profile = EmotionalHubRuntime._apply_inner_os_surface_profile
    _build_inner_os_llm_guidance = EmotionalHubRuntime._build_inner_os_llm_guidance
    _apply_surface_context_packet_to_content_sequence = EmotionalHubRuntime._apply_surface_context_packet_to_content_sequence
    _shape_inner_os_surface_text = EmotionalHubRuntime._shape_inner_os_surface_text
    _apply_inner_os_actuation_timing_profile = EmotionalHubRuntime._apply_inner_os_actuation_timing_profile
    _inner_os_emit_clock_ms = EmotionalHubRuntime._inner_os_emit_clock_ms
    _inner_os_emit_wait_enabled = EmotionalHubRuntime._inner_os_emit_wait_enabled
    _apply_inner_os_turn_timing_guard = EmotionalHubRuntime._apply_inner_os_turn_timing_guard
    _apply_inner_os_emit_timing = EmotionalHubRuntime._apply_inner_os_emit_timing
    _shape_inner_os_surface_profile_text = EmotionalHubRuntime._shape_inner_os_surface_profile_text
    _render_fast_ack = EmotionalHubRuntime._render_fast_ack
    _render_inner_os_response_channel_text = EmotionalHubRuntime._render_inner_os_response_channel_text
    _shape_inner_os_content_sequence = EmotionalHubRuntime._shape_inner_os_content_sequence
    _select_short_inner_os_sequence = EmotionalHubRuntime._select_short_inner_os_sequence
    _compact_inner_os_sequence_text = EmotionalHubRuntime._compact_inner_os_sequence_text
    _inner_os_stream_state_from_hints = EmotionalHubRuntime._inner_os_stream_state_from_hints
    _inner_os_live_followup_signals = EmotionalHubRuntime._inner_os_live_followup_signals
    _inner_os_live_style_feedback = EmotionalHubRuntime._inner_os_live_style_feedback
    _inner_os_style_alignment = EmotionalHubRuntime._inner_os_style_alignment
    _inner_os_stream_converged = EmotionalHubRuntime._inner_os_stream_converged
    _run_inner_os_live_response_loop = EmotionalHubRuntime._run_inner_os_live_response_loop
    _compose_inner_os_surface_text = EmotionalHubRuntime._compose_inner_os_surface_text
    _inner_os_surface_probe = EmotionalHubRuntime._inner_os_surface_probe
    _inner_os_surface_reopening_line = EmotionalHubRuntime._inner_os_surface_reopening_line
    _inner_os_surface_contract_state = EmotionalHubRuntime._inner_os_surface_contract_state
    _inner_os_surface_prefix = EmotionalHubRuntime._inner_os_surface_prefix
    _inner_os_surface_closing = EmotionalHubRuntime._inner_os_surface_closing
    _inner_os_surface_policy_level = EmotionalHubRuntime._inner_os_surface_policy_level
    _inner_os_surface_policy_intent = EmotionalHubRuntime._inner_os_surface_policy_intent
    _recent_surface_user_history = EmotionalHubRuntime._recent_surface_user_history
    _recent_surface_response_history = EmotionalHubRuntime._recent_surface_response_history
    _recent_dialogue_thread_history = EmotionalHubRuntime._recent_dialogue_thread_history
    _build_context_payload = EmotionalHubRuntime._build_context_payload
    _serialize_response_meta = EmotionalHubRuntime._serialize_response_meta
    _emit_inner_os_distillation_record = EmotionalHubRuntime._emit_inner_os_distillation_record
    _build_inner_os_dashboard_snapshot = EmotionalHubRuntime._build_inner_os_dashboard_snapshot
    export_inner_os_transfer_package = EmotionalHubRuntime.export_inner_os_transfer_package
    build_inner_os_model_swap_bundle = EmotionalHubRuntime.build_inner_os_model_swap_bundle
    load_inner_os_transfer_package = EmotionalHubRuntime.load_inner_os_transfer_package
    warm_start_from_transfer_package = EmotionalHubRuntime.warm_start_from_transfer_package
    build_inner_os_state_seed_from_transfer_package = EmotionalHubRuntime.build_inner_os_state_seed_from_transfer_package
    apply_inner_os_transfer_package = EmotionalHubRuntime.apply_inner_os_transfer_package
    _load_inner_os_transfer_package_from_disk = EmotionalHubRuntime._load_inner_os_transfer_package_from_disk
    _persist_inner_os_transfer_package = EmotionalHubRuntime._persist_inner_os_transfer_package
    _persist_inner_os_dashboard_snapshot = EmotionalHubRuntime._persist_inner_os_dashboard_snapshot

    def _surface_safety_bias(self) -> float:
        return 0.2

    def _surface_mode(self) -> str:
        return "reality"

    def _dominant_memory_anchor(self) -> str:
        return "harbor slope"

    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am trying to stay with what is grounded. I will avoid overreading the scene.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": [{"id": "vision-1"}]},
            perception_summary={"text": "soft evening slope and signboard"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": {
                "reply": "tentative memory reply",
                "fidelity": 0.88,
                "meta": {"source_class": "self", "audit_event": "OK"},
                "candidate": {"label": "harbor slope"},
            },
        }


@dataclass
class _DegradedHarness(_ProcessTurnHarness):
    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am keeping the reading tentative.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": []},
            perception_summary={"error": "vision_error", "detail": "500 Server Error"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": None,
        }


def test_apply_inner_os_expression_controls_softens_llm_controls() -> None:
    base = {"temperature": 0.48, "top_p": 0.92, "directness": 0.22}
    adjusted = _apply_inner_os_expression_controls(
        base,
        {"tentative_bias": 0.64, "assertiveness_cap": 0.58, "question_bias": 0.35},
    )
    assert adjusted["directness"] < base["directness"]
    assert adjusted["temperature"] < base["temperature"]
    assert adjusted["top_p"] < base["top_p"]
    assert adjusted["inner_os_tentative_bias"] == 0.64
    assert adjusted["inner_os_assertiveness_cap"] == 0.58
    assert adjusted["inner_os_question_bias"] == 0.35


def test_shape_inner_os_surface_text_shortens_check_in_text() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    shaped = harness._shape_inner_os_surface_text(
        "I am trying to stay with what is grounded. I will avoid overreading the scene.",
        intent="check_in",
    )
    assert shaped == "I am trying to stay with what is grounded."


def test_inner_os_surface_closing_depends_on_question_bias() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.12) == ""
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.34) == "Then I can answer a little more cleanly."
    assert harness._inner_os_surface_closing(intent="check_in", question_bias=0.34) == "I want to stay with this gently first."


def test_apply_inner_os_surface_policy_prefixes_clarifying_line() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert updated.text.endswith("Then I can answer a little more cleanly.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"


def test_apply_inner_os_surface_policy_keeps_opening_only_when_question_bias_is_low() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.12},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert not updated.text.endswith("Then I can answer a little more cleanly.")


def test_apply_inner_os_surface_policy_uses_check_in_prefix() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check in gently before I go further.")
    assert "I will avoid overreading the scene." not in updated.text
    assert updated.text.endswith("I want to stay with this gently first.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:check_in"


def test_process_turn_uses_response_history_for_discussion_anchor() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._surface_response_history = deque(maxlen=3)
    harness._surface_response_history.append(
        "前に触れていた「港での約束」のところから、いま話せる分だけ戻れば十分です。"
    )

    result = harness.process_turn(
        user_text="前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
        intent="check_in",
    )

    inner_os = dict(result.persona_meta.get("inner_os") or {})
    packet = dict(inner_os.get("interaction_policy_packet") or {})
    recent_dialogue_state = dict(packet.get("recent_dialogue_state") or {})
    discussion_thread_state = dict(packet.get("discussion_thread_state") or {})

    assert recent_dialogue_state.get("recent_anchor") == "港での約束"
    assert discussion_thread_state.get("topic_anchor") == "港での約束"


def test_apply_inner_os_surface_policy_can_use_interaction_policy_packet() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "response_strategy": "respectful_wait",
                "opening_move": "acknowledge_and_wait",
                "closing_move": "leave_space",
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None


def test_apply_inner_os_emit_timing_records_absolute_windows() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._inner_os_emit_wait_enabled = lambda: False  # type: ignore[assignment]
    harness._inner_os_emit_clock_ms = lambda: 1000.0  # type: ignore[assignment]
    response = SimpleNamespace(latency_ms=12.0, controls_used={}, controls={})

    emit_timing = harness._apply_inner_os_emit_timing(
        response,
        actuation_plan={
            "response_channel": "hold",
            "turn_timing_hint": {
                "response_channel": "hold",
                "entry_window": "held",
                "pause_profile": "soft_pause",
                "overlap_policy": "wait_for_release",
                "interruptibility": "low",
                "minimum_wait_ms": 420,
                "interrupt_guard_ms": 420,
            },
        },
    )

    assert response.latency_ms == 420.0
    assert emit_timing.effective_emit_delay_ms == 408.0
    assert emit_timing.emit_not_before_ms == 1408.0
    assert emit_timing.interrupt_guard_until_ms == 1828.0
    assert emit_timing.wait_applied is False
    assert emit_timing.wait_applied_ms == 0.0
    assert response.controls_used["inner_os_emit_timing"]["entry_window"] == "held"


def test_inner_os_emit_wait_enabled_uses_streaming_surface_mode() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness.config = SimpleNamespace(latency=SimpleNamespace(enable_loose=True))
    harness._surface_mode = lambda: "streaming"  # type: ignore[assignment]

    assert harness._inner_os_emit_wait_enabled() is True

    harness._surface_mode = lambda: "reality"  # type: ignore[assignment]
    assert harness._inner_os_emit_wait_enabled() is False


def test_apply_inner_os_emit_timing_can_apply_wait(monkeypatch: Any) -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._inner_os_emit_wait_enabled = lambda: True  # type: ignore[assignment]
    harness._inner_os_emit_clock_ms = lambda: 2000.0  # type: ignore[assignment]
    response = SimpleNamespace(latency_ms=5.0, controls_used={}, controls={})
    sleep_calls: list[float] = []
    monkeypatch.setattr(runtime_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    emit_timing = harness._apply_inner_os_emit_timing(
        response,
        actuation_plan={
            "response_channel": "backchannel",
            "turn_timing_hint": {
                "response_channel": "backchannel",
                "entry_window": "ready",
                "pause_profile": "none",
                "overlap_policy": "allow_soft_overlap",
                "interruptibility": "high",
                "minimum_wait_ms": 40,
                "interrupt_guard_ms": 90,
            },
        },
    )

    assert sleep_calls == [0.035]
    assert emit_timing.effective_emit_delay_ms == 35.0
    assert emit_timing.wait_applied is True
    assert emit_timing.wait_applied_ms == 35.0
    assert emit_timing.emit_not_before_ms == 2035.0
    assert emit_timing.interrupt_guard_until_ms == 2125.0


def test_apply_inner_os_emit_timing_applies_wait_in_streaming_mode(monkeypatch: Any) -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness.config = SimpleNamespace(latency=SimpleNamespace(enable_loose=True))
    harness._surface_mode = lambda: "streaming"  # type: ignore[assignment]
    harness._inner_os_emit_clock_ms = lambda: 3000.0  # type: ignore[assignment]
    response = SimpleNamespace(latency_ms=5.0, controls_used={}, controls={})
    sleep_calls: list[float] = []
    monkeypatch.setattr(runtime_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    emit_timing = harness._apply_inner_os_emit_timing(
        response,
        actuation_plan={
            "response_channel": "hold",
            "turn_timing_hint": {
                "response_channel": "hold",
                "entry_window": "held",
                "pause_profile": "soft_pause",
                "overlap_policy": "wait_for_release",
                "interruptibility": "low",
                "minimum_wait_ms": 40,
                "interrupt_guard_ms": 90,
            },
        },
    )

    assert sleep_calls == [0.035]
    assert emit_timing.wait_applied is True
    assert emit_timing.wait_applied_ms == 35.0
    assert emit_timing.emit_not_before_ms == 3035.0


def test_apply_inner_os_turn_timing_guard_blocks_until_emit_window() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._inner_os_emit_response_channel = "hold"  # type: ignore[attr-defined]
    harness._inner_os_emit_overlap_policy = "wait_for_release"  # type: ignore[attr-defined]
    harness._inner_os_emit_not_before_ms = 1500.0  # type: ignore[attr-defined]
    harness._inner_os_interrupt_guard_until_ms = 1900.0  # type: ignore[attr-defined]

    gate_ctx = runtime_module.GateContext(
        engaged=True,
        face_motion=0.08,
        blink=0.0,
        voice_energy=0.0,
        delta_m=0.0,
        jerk=0.0,
        text_input=False,
        since_last_user_ms=640.0,
        force_listen=False,
    )

    updated = harness._apply_inner_os_turn_timing_guard(gate_ctx, now_ts_ms=1400.0)

    assert updated.force_listen is True
    assert harness._last_inner_os_timing_guard.active is True
    assert harness._last_inner_os_timing_guard.reason == "emit_delay"


def test_apply_inner_os_turn_timing_guard_blocks_interrupt_overlap_when_user_still_speaking() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._inner_os_emit_response_channel = "backchannel"  # type: ignore[attr-defined]
    harness._inner_os_emit_overlap_policy = "wait_for_release"  # type: ignore[attr-defined]
    harness._inner_os_emit_not_before_ms = 1000.0  # type: ignore[attr-defined]
    harness._inner_os_interrupt_guard_until_ms = 1600.0  # type: ignore[attr-defined]

    gate_ctx = runtime_module.GateContext(
        engaged=True,
        face_motion=0.08,
        blink=0.0,
        voice_energy=0.09,
        delta_m=0.0,
        jerk=0.0,
        text_input=False,
        since_last_user_ms=510.0,
        force_listen=False,
    )

    updated = harness._apply_inner_os_turn_timing_guard(gate_ctx, now_ts_ms=1200.0)

    assert updated.force_listen is True
    assert harness._last_inner_os_timing_guard.active is True
    assert harness._last_inner_os_timing_guard.reason == "interrupt_guard"


def test_apply_inner_os_turn_timing_guard_allows_soft_overlap() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._inner_os_emit_response_channel = "backchannel"  # type: ignore[attr-defined]
    harness._inner_os_emit_overlap_policy = "allow_soft_overlap"  # type: ignore[attr-defined]
    harness._inner_os_emit_not_before_ms = 1000.0  # type: ignore[attr-defined]
    harness._inner_os_interrupt_guard_until_ms = 1600.0  # type: ignore[attr-defined]

    gate_ctx = runtime_module.GateContext(
        engaged=True,
        face_motion=0.08,
        blink=0.0,
        voice_energy=0.09,
        delta_m=0.0,
        jerk=0.0,
        text_input=False,
        since_last_user_ms=510.0,
        force_listen=False,
    )

    updated = harness._apply_inner_os_turn_timing_guard(gate_ctx, now_ts_ms=1200.0)

    assert updated.force_listen is False
    assert harness._last_inner_os_timing_guard.active is False
    assert harness._last_inner_os_timing_guard.reason == "idle"


def test_serialize_response_meta_keeps_inner_os_timing_contract() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        model="qwen3.5-4b",
        model_source="lmstudio",
        trace_id="turn-123",
        latency_ms=84.0,
        controls_used={
            "temperature": 0.4,
            "inner_os_emit_timing": {
                "response_channel": "hold",
                "entry_window": "held",
                "overlap_policy": "wait_for_release",
                "minimum_wait_ms": 420.0,
                "interrupt_guard_ms": 420.0,
                "emit_not_before_ms": 1408.0,
                "interrupt_guard_until_ms": 1828.0,
                "wait_applied": False,
            },
            "inner_os_timing_guard": {
                "active": True,
                "reason": "interrupt_guard",
                "response_channel": "hold",
                "overlap_policy": "wait_for_release",
                "emit_not_before_ms": 1408.0,
                "interrupt_guard_until_ms": 1828.0,
                "voice_conflict": True,
            },
            "inner_os_gate_force_listen": True,
        },
        safety={"status": "ok"},
        confidence=0.73,
        uncertainty_reason=("none",),
        perception_summary=None,
        retrieval_summary=None,
    )

    meta = harness._serialize_response_meta(response)

    assert meta is not None
    assert meta["controls_used"] == {"temperature": 0.4, "inner_os_gate_force_listen": 1.0}
    assert meta["actuation_emit_timing"]["response_channel"] == "hold"
    assert meta["actuation_emit_timing"]["emit_not_before_ms"] == 1408.0
    assert meta["timing_guard"]["reason"] == "interrupt_guard"
    assert meta["timing_guard"]["voice_conflict"] is True
    assert meta["gate_force_listen"] is True


def test_apply_inner_os_surface_policy_can_use_object_operation_and_effects_without_strategy() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "opening_move": "acknowledge_without_probe",
                "closing_move": "",
                "primary_object_operation": {
                    "operation_kind": "hold_without_probe",
                },
                "object_operation_kinds": ["hold_without_probe", "protect_unfinished_part"],
                "interaction_effect_kinds": ["avoid_forced_reopening"],
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_apply_inner_os_surface_policy_can_use_ordered_contract_fields_without_strategy() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "primary_object_operation": {
                    "operation_kind": "hold_without_probe",
                },
                "ordered_operation_kinds": ["hold_without_probe"],
                "ordered_effect_kinds": ["preserve_self_pacing"],
                "deferred_object_labels": ["deeper meaning"],
                "question_budget": 0,
                "question_pressure": 0.62,
                "defer_dominance": 0.74,
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_apply_inner_os_surface_policy_can_use_conversation_contract_aliases_only() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I can stay with what is visible first.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "interaction_policy_packet": {
                "conversation_contract": {
                    "focus_now": "what is visible first",
                    "leave_closed_for_now": ["deeper meaning"],
                    "response_action_now": {
                        "primary_operation": "hold_without_probe",
                        "question_budget": 0,
                        "question_pressure": 0.62,
                        "defer_dominance": 0.74,
                        "ordered_operations": ["hold_without_probe"],
                    },
                    "ordered_effects": ["preserve_self_pacing"],
                    "wanted_effect_on_other": [
                        {"effect": "preserve_self_pacing", "target": "what is visible first", "intensity": 0.68}
                    ],
                },
            },
        },
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me give this a little more room before I press further.")
    assert updated.text.endswith("We do not need to open the rest right now.")


def test_inner_os_surface_closing_can_follow_ordered_effects_without_legacy_kind_lists() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    closing = harness._inner_os_surface_closing(
        intent="check_in",
        question_bias=0.12,
        interaction_policy={
            "primary_object_operation": {"operation_kind": "offer_small_next_step"},
            "ordered_effect_kinds": ["enable_small_next_step"],
            "question_pressure": 0.18,
            "defer_dominance": 0.12,
        },
    )
    assert closing == "We can keep the next step connected from here."


def test_inner_os_surface_closing_can_follow_conversation_contract_aliases_only() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    closing = harness._inner_os_surface_closing(
        intent="check_in",
        question_bias=0.12,
        interaction_policy={
            "conversation_contract": {
                "response_action_now": {
                    "primary_operation": "offer_small_next_step",
                    "question_pressure": 0.18,
                    "defer_dominance": 0.12,
                },
                "ordered_effects": ["enable_small_next_step"],
                "wanted_effect_on_other": [
                    {"effect": "enable_small_next_step", "target": "next step", "intensity": 0.61}
                ],
            },
        },
    )
    assert closing == "We can keep the next step connected from here."


def test_build_inner_os_llm_guidance_uses_policy_packet_and_sequence() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "clarify",
                "response_strategy": "repair_then_attune",
                "dialogue_order": [
                    "open:name_overreach_and_reduce_force",
                    "follow:invite_visible_state",
                    "close:hold_space",
                ],
                "do_not_cross": ["overinterpret"],
                "shell_guidance": ["opening:held", "return:careful_return"],
            },
            "surface_opening_delay": "long",
            "surface_response_length": "short",
            "surface_sentence_temperature": "steady",
            "surface_pause_insertion": "visible_pause",
            "surface_certainty_style": "careful",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
            "partner_utterance_stance": "measured_check_in",
            "interaction_judgement_view": {
                "observed_signals": [{"signal_id": "observed:0", "signal_kind": "user_text", "text": "Can you stay with what is visible first?", "confidence": 1.0}],
                "inferred_signals": [{"signal_id": "inferred:0", "signal_kind": "detail_room", "statement": "system estimates that the other person has low room for detail right now", "strength": 0.24}],
                "selected_object_labels": ["what is visible first"],
                "deferred_object_labels": ["deeper meaning"],
                "active_operation_labels": ["hold_without_probe:what is visible first"],
                "intended_effect_labels": ["preserve_self_pacing:what is visible first"],
                "cues": ["observed:1", "inferred:1"],
            },
            "interaction_judgement_summary": {
                "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                "compact_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
            },
            "interaction_condition_report": {
                "scene_lines": ["system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。"],
                "relation_lines": ["system は、この相手には丁寧さを強めた方がよいと見ています。"],
                "memory_lines": ["system は、「harbor slope」を今回の記憶の足場にしています。"],
                "integration_lines": ["そのため system は、いまは詳しく聞くより、相手に話す余地を残す方がよいと見ています。"],
                "report_lines": [
                    "場面が効いていること: system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。",
                    "相手との関係が効いていること: system は、この相手には丁寧さを強めた方がよいと見ています。",
                ],
            },
            "conversation_contract": {
                "primary_object": "what is visible first",
                "focus_now": "what is visible first",
                "do_not_open_yet": ["deeper meaning"],
                "response_action_now": {
                    "primary_operation": "hold_without_probe",
                    "question_budget": 0,
                    "question_pressure": 0.72,
                },
                "wanted_effect_on_other": [
                    {"effect": "preserve_self_pacing", "target": "what is visible first", "intensity": 0.6}
                ],
            },
            "interaction_inspection_report": {
                "shared_observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "changed_sections": [],
                "case_reports": [
                    {
                        "case_id": "current_case",
                        "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                        "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                        "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                        "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                        "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                    }
                ],
                "report_lines": [
                    "共通して観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                    "current_case の確認結果です。",
                    "観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                ],
            },
            "interaction_audit_bundle": {
                "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                "inferred_lines": ["system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。（強さ 0.24）"],
                "selected_object_lines": ["今回 system が扱う対象は「what is visible first」です。"],
                "deferred_object_lines": ["今回 system は「deeper meaning」にはまだ深く触れません。"],
                "operation_lines": ["system は「what is visible first」について、相手に詳しい説明を求めずに受け止めます。（強さ 0.78 / 相手の負担見積り 0.12）"],
                "intended_effect_lines": ["system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。（強さ 0.60）"],
                "scene_lines": ["system は、いまは距離や言い方を急に近づけない方がよい場面だと見ています。"],
                "relation_lines": ["system は、この相手には丁寧さを強めた方がよいと見ています。"],
                "memory_lines": ["system は、「harbor slope」を今回の記憶の足場にしています。"],
                "integration_lines": ["そのため system は、いまは詳しく聞くより、相手に話す余地を残す方がよいと見ています。"],
                "inspection_lines": ["current_case の確認結果です。"],
                "key_metrics": {"question_budget": 0, "question_pressure": 0.72},
                "report_lines": ["観測したこと: 相手は「Can you stay with what is visible first?」と言いました。"],
            },
            "interaction_audit_casebook": {
                "cases": [
                    {
                        "observed_text": "Can you stay with what is visible first?",
                        "condition_key": "attuned_presence|familiar|harbor slope",
                        "scene_family": "attuned_presence",
                        "relation_hint": "familiar",
                        "memory_anchor": "harbor slope",
                        "judgement_summary": {
                            "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        },
                        "audit_bundle": {
                            "observed_lines": ["相手は「Can you stay with what is visible first?」と言いました。"],
                        },
                    }
                ]
            },
            "interaction_audit_report": {
                "changed_sections": ["inferred_lines"],
                "metric_differences": [],
                "report_lines": [
                    "current_case の監査結果です。",
                    "観測したこと: 相手は「Can you stay with what is visible first?」と言いました。",
                ],
            },
            "interaction_audit_reference_case_ids": ["reference_1"],
            "interaction_audit_reference_case_meta": {
                "reference_1": {
                    "scene_family": "attuned_presence",
                    "relation_hint": "familiar",
                    "memory_anchor": "harbor slope",
                    "condition_key": "attuned_presence|familiar|harbor slope",
                }
            },
            "contact_reflection_state": {
                "reflection_style": "reflect_only",
            },
            "green_kernel_composition": {
                "field": {
                    "guardedness": 0.62,
                    "reopening_pull": 0.41,
                    "affective_charge": 0.58,
                }
            },
        },
        user_text="Can you stay with what is visible first?",
        intent="clarify",
    )
    assert guidance["interaction_policy"]["response_strategy"] == "repair_then_attune"
    assert guidance["utterance_stance"] == "measured_check_in"
    assert guidance["surface_profile"]["response_length"] == "short"
    assert guidance["interaction_judgement_view"]["observed_signals"]
    assert guidance["interaction_judgement_view"]["selected_object_labels"] == ["what is visible first"]
    assert guidance["interaction_judgement_summary"]["observed_lines"]
    assert guidance["interaction_judgement_summary"]["selected_object_lines"] == ["今回 system が扱う対象は「what is visible first」です。"]
    assert guidance["interaction_condition_report"]["scene_lines"]
    assert guidance["interaction_condition_report"]["report_lines"]
    assert guidance["conversation_contract"]["focus_now"] == "what is visible first"
    assert guidance["conversation_contract"]["response_action_now"]["question_budget"] == 0
    assert guidance["interaction_inspection_report"]["case_reports"]
    assert any("current_case" in line for line in guidance["interaction_inspection_report"]["report_lines"])
    assert guidance["interaction_audit_bundle"]["report_lines"]
    assert guidance["interaction_audit_bundle"]["key_metrics"]["question_budget"] == 0
    assert guidance["interaction_audit_casebook"]["cases"]
    assert guidance["interaction_audit_report"]["report_lines"]
    assert guidance["interaction_audit_reference_case_ids"] == ["reference_1"]
    assert guidance["interaction_audit_reference_case_meta"]["reference_1"]["scene_family"] == "attuned_presence"
    assert guidance["surface_context_packet"]["conversation_phase"] == "clarify"
    assert guidance["surface_context_packet"]["constraints"]["no_generic_clarification"] is True
    assert guidance["surface_context_packet"]["shared_core"]["already_shared"] == ["Can you stay with what is visible first?"]
    assert guidance["surface_context_packet"]["response_role"]["secondary"] == "reflect_only"
    assert guidance["surface_context_packet"]["surface_profile"]["heartbeat_reaction"] in {"steady", "attune", "contain", "recover", "bounce"}
    assert guidance["surface_context_packet"]["source_state"]["heartbeat_response_tempo"] >= 0.0
    assert guidance["reaction_contract"]["question_budget"] == 0
    assert guidance["reaction_contract"]["response_channel"] in {"speak", "backchannel", "hold", "defer", ""}
    assert guidance["reaction_contract"]["continuity_mode"] in {"fresh", "continue", "reopen", "open"}
    sequence = guidance["content_sequence"]
    assert isinstance(sequence, list)
    assert len(sequence) >= 2
    assert sequence[0]["act"] == "acknowledge_overreach"


def test_build_inner_os_llm_guidance_prunes_thread_reopening_sequence_with_surface_context_packet() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "recent_dialogue_state": {"state": "reopening_thread", "recent_anchor": "あの約束"},
                "discussion_thread_state": {"state": "revisit_issue", "topic_anchor": "あの約束"},
                "issue_state": {"state": "pausing_issue", "issue_anchor": "あの約束"},
            },
            "surface_response_length": "short",
            "surface_cultural_register": "soft_companion",
            "surface_group_register": "one_to_one",
            "surface_sentence_temperature": "gentle",
            "turn_delta": {
                "kind": "issue_pause",
                "preferred_act": "leave_return_point_from_anchor",
                "anchor_hint": "あの約束",
            },
            "surface_context_packet": {
                "conversation_phase": "issue_pause",
                "shared_core": {
                    "anchor": "あの約束",
                    "already_shared": ["あの約束"],
                    "not_yet_shared": [],
                },
                "response_role": {
                    "primary": "leave_return_point_from_anchor",
                    "secondary": "reflect_only",
                },
                "constraints": {
                    "no_generic_clarification": True,
                    "no_advice": True,
                    "max_questions": 0,
                    "keep_thread_visible": True,
                    "prefer_return_point": True,
                    "boundary_style": "soft_hold",
                },
                "surface_profile": {
                    "response_length": "short",
                    "cultural_register": "soft_companion",
                    "group_register": "one_to_one",
                    "sentence_temperature": "gentle",
                    "surface_mode": "",
                },
                "source_state": {
                    "recent_dialogue_state": "reopening_thread",
                    "discussion_thread_state": "revisit_issue",
                    "issue_state": "pausing_issue",
                    "turn_delta_kind": "issue_pause",
                    "green_guardedness": 0.4,
                    "green_reopening_pull": 0.6,
                    "green_affective_charge": 0.3,
                },
            },
        },
        user_text="前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
        intent="check_in",
    )

    sequence = guidance["content_sequence"]
    acts = [item["act"] for item in sequence]
    assert acts == ["leave_return_point_from_anchor"]


def test_build_inner_os_llm_guidance_prunes_deep_disclosure_sequence_when_questions_are_blocked() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "recent_dialogue_state": {"state": "continuing_thread"},
                "discussion_thread_state": {"state": "discussion_unresolved"},
                "issue_state": {"state": "exploring_issue"},
            },
            "surface_response_length": "short",
            "surface_cultural_register": "soft_companion",
            "surface_group_register": "one_to_one",
            "surface_sentence_temperature": "gentle",
            "turn_delta": {
                "kind": "discussion_unresolved",
                "preferred_act": "stay_with_present_need",
            },
            "surface_context_packet": {
                "conversation_phase": "discussion_unresolved",
                "shared_core": {
                    "anchor": "",
                    "already_shared": ["本当は、あのとき助けてほしかったって、まだ言えていないんです。"],
                    "not_yet_shared": ["助けてほしかった"],
                },
                "response_role": {
                    "primary": "stay_with_present_need",
                    "secondary": "reflect_only",
                },
                "constraints": {
                    "no_generic_clarification": True,
                    "no_advice": True,
                    "max_questions": 0,
                    "keep_thread_visible": True,
                    "prefer_return_point": False,
                    "boundary_style": "soft_hold",
                },
                "surface_profile": {
                    "response_length": "short",
                    "cultural_register": "soft_companion",
                    "group_register": "one_to_one",
                    "sentence_temperature": "gentle",
                    "surface_mode": "",
                },
                "source_state": {
                    "recent_dialogue_state": "continuing_thread",
                    "discussion_thread_state": "discussion_unresolved",
                    "issue_state": "exploring_issue",
                    "turn_delta_kind": "discussion_unresolved",
                    "green_guardedness": 0.7,
                    "green_reopening_pull": 0.2,
                    "green_affective_charge": 0.8,
                },
            },
        },
        user_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        intent="check_in",
    )

    sequence = guidance["content_sequence"]
    acts = [item["act"] for item in sequence]
    assert acts == ["reflect_hidden_need", "stay_with_present_need"]


def test_build_inner_os_llm_guidance_includes_bright_discourse_shape() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "recent_dialogue_state": {"state": "continuing_thread"},
                "discussion_thread_state": {"state": "discussion_open"},
                "issue_state": {"state": "open_thread"},
                "live_engagement_state": {"state": "pickup_comment"},
                "lightness_budget_state": {"state": "open_play"},
            },
            "surface_response_length": "short",
            "surface_cultural_register": "soft_companion",
            "surface_group_register": "one_to_one",
            "surface_sentence_temperature": "warm",
            "surface_voice_texture": "measured",
            "turn_delta": {
                "kind": "bright_continuity",
                "preferred_act": "light_bounce",
            },
            "organism_state": {
                "dominant_posture": "play",
                "attunement": 0.68,
                "coherence": 0.62,
                "grounding": 0.57,
                "protective_tension": 0.2,
                "expressive_readiness": 0.74,
                "play_window": 0.76,
                "relation_pull": 0.64,
                "social_exposure": 0.16,
            },
            "surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "shared_core": {
                    "anchor": "",
                    "already_shared": ["ちょっと笑えることがあった"],
                    "not_yet_shared": [],
                },
                "response_role": {
                    "primary": "light_bounce",
                    "secondary": "shared_delight",
                },
                "constraints": {
                    "max_questions": 0,
                    "keep_thread_visible": True,
                    "allow_small_next_step": True,
                },
                "surface_profile": {
                    "response_length": "short",
                    "voice_texture": "measured",
                    "playfulness": 0.36,
                    "tempo": 0.44,
                    "organism_posture": "play",
                    "organism_play_window": 0.76,
                },
                "source_state": {
                    "live_engagement_state": "pickup_comment",
                    "lightness_budget_state": "open_play",
                    "organism_posture": "play",
                    "organism_expressive_readiness": 0.74,
                },
            },
        },
        user_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        intent="check_in",
    )

    assert guidance["discourse_shape"]["shape_id"] == "bright_bounce"
    assert guidance["surface_context_packet"]["surface_profile"]["organism_posture"] == "play"
    acts = [item["act"] for item in guidance["content_sequence"]]
    assert acts == ["shared_delight", "light_bounce"]


def test_build_inner_os_llm_guidance_accepts_typed_contract_inputs() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    harness._last_gate_context["inner_os_heartbeat_structure_state"] = {}
    guidance = harness._build_inner_os_llm_guidance(
        expression_hint={
            "interaction_policy_packet": coerce_interaction_policy_packet(
                {
                    "dialogue_act": "check_in",
                    "recent_dialogue_state": {"state": "continuing_thread"},
                    "discussion_thread_state": {"state": "discussion_open"},
                    "issue_state": {"state": "open_thread"},
                    "live_engagement_state": {"state": "pickup_comment", "score": 0.62},
                    "lightness_budget_state": {"state": "open_play", "banter_room": 0.48},
                    "shared_moment_state": {"moment_kind": "laugh", "afterglow": 0.52},
                    "listener_action_state": {
                        "state": "warm_laugh_ack",
                        "token_profile": "soft_laugh",
                    },
                    "joint_state": {"dominant_mode": "shared_delight", "common_ground": 0.64},
                }
            ),
            "action_posture": coerce_action_posture_contract(
                {
                    "engagement_mode": "co_move",
                    "primary_operation_kind": "share_small_shift",
                }
            ),
            "actuation_plan": coerce_actuation_plan_contract(
                {
                    "execution_mode": "shared_progression",
                    "primary_action": "riff_current_comment",
                    "response_channel": "speak",
                }
            ),
            "surface_context_packet": coerce_surface_context_packet(
                {
                    "conversation_phase": "bright_continuity",
                    "shared_core": {
                        "anchor": "あのあとちょっと笑えることもあって。",
                        "already_shared": ["あのあとちょっと笑えることもあって。"],
                    },
                    "response_role": {"primary": "shared_delight"},
                    "constraints": {"max_questions": 0},
                    "surface_profile": {"voice_texture": "light_playful"},
                    "source_state": {"utterance_reason_offer": "brief_shared_smile"},
                }
            ),
            "surface_response_length": "short",
            "surface_voice_texture": "light_playful",
            "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
        },
        user_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        intent="check_in",
    )

    assert guidance["interaction_policy"]["dialogue_act"] == "check_in"
    assert guidance["action_posture"]["engagement_mode"] == "co_move"
    assert guidance["actuation_plan"]["response_channel"] == "speak"
    assert guidance["surface_context_packet"]["conversation_phase"] == "bright_continuity"
    assert guidance["surface_context_packet"]["surface_profile"]["voice_texture"] == "light_playful"
    assert guidance["discourse_shape"]["shape_id"] == "bright_bounce"


def test_apply_inner_os_surface_profile_shapes_text_and_controls() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        text="I am trying to stay with what is grounded. I will avoid overreading the scene.",
        controls_used={"mode": "watch"},
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "long",
            "surface_response_length": "short",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "visible_pause",
            "surface_certainty_style": "tentative",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "soft_return",
        },
    )
    assert updated is not None
    assert (
        updated.text.startswith("... Gently,")
        or updated.text.startswith("... I think")
        or updated.text.startswith("... gently,")
        or updated.text.startswith("... I am")
        or updated.text.startswith("... I'm here.")
    )
    assert "I will avoid overreading the scene." not in updated.text
    assert updated.controls_used["inner_os_surface_profile"]["response_length"] == "short"
    assert updated.controls_used["inner_os_surface_profile"]["return_gaze_expectation"] == "soft_return"
    assert updated.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1


def test_apply_inner_os_surface_profile_uses_live_mismatch_to_shorten_and_carefully_shape() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        text="I can describe what I notice and then follow the thread a little further.",
        controls_used={"mode": "watch"},
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "brief",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "measured",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "live_opening_pace_mismatch": 0.66,
            "live_return_gaze_mismatch": 0.62,
            "stream_shared_attention_window_mean": 0.2,
        },
    )
    assert updated is not None
    assert updated.controls_used["inner_os_surface_profile"]["response_length"] == "short"
    assert updated.controls_used["inner_os_surface_profile"]["certainty_style"] == "careful"
    assert updated.controls_used["inner_os_surface_profile"]["pause_insertion"] == "visible_pause"
    assert updated.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1


def test_apply_inner_os_surface_profile_uses_backchannel_timing_from_actuation_plan() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    runtime = EmotionalHubRuntime(runtime_module.RuntimeConfig(use_eqnet_core=True))
    harness._ack_cfg = runtime._ack_cfg  # type: ignore[attr-defined]
    harness._runtime_cfg = runtime._runtime_cfg  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="I can stay with the thread for a moment longer.",
        controls_used={
            "mode": "talk",
            "inner_os_planned_content_sequence": [
                {"act": "light_bounce", "text": "That stays a little open."},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "soft_return",
            "actuation_plan": {
                "response_channel": "backchannel",
                "wait_before_action": "brief",
                "reply_permission": "hold_or_brief",
                "nonverbal_response_state": {
                    "timing_bias": "just_after_turn",
                },
            },
        },
    )
    assert updated is not None
    profile = updated.controls_used["inner_os_surface_profile"]
    assert profile["actuation_response_channel"] == "backchannel"
    assert profile["opening_pace_windowed"] == "ready"
    assert profile["pause_insertion"] == "none"
    assert profile["response_length"] == "short"


def test_apply_inner_os_surface_profile_uses_hold_timing_from_actuation_plan() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    runtime = EmotionalHubRuntime(runtime_module.RuntimeConfig(use_eqnet_core=True))
    harness._ack_cfg = runtime._ack_cfg  # type: ignore[attr-defined]
    harness._runtime_cfg = runtime._runtime_cfg  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="I can leave the space open.",
        controls_used={
            "mode": "watch",
            "inner_os_planned_content_sequence": [
                {"act": "quiet_presence", "text": "This stays quiet."},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "brief",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "actuation_plan": {
                "response_channel": "hold",
                "wait_before_action": "extended",
                "reply_permission": "hold_or_brief",
                "nonverbal_response_state": {
                    "timing_bias": "wait",
                },
            },
        },
    )
    assert updated is not None
    profile = updated.controls_used["inner_os_surface_profile"]
    assert profile["actuation_response_channel"] == "hold"
    assert profile["opening_pace_windowed"] == "held"
    assert profile["pause_insertion"] == "soft_pause"
    assert profile["response_length"] == "short"



def test_apply_inner_os_surface_profile_prefers_bright_short_sequence_when_voice_is_light_playful() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "watch",
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
                {"act": "leave_unfinished_closed", "text": "また話せそうなときに、そこからで大丈夫です。"},
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "surface_voice_texture": "light_playful",
        },
    )
    assert updated is not None
    assert "それは、ちょっといい感じだね。" in updated.text
    assert "大丈夫です" not in updated.text
    assert "話せるところから" not in updated.text
    assert not updated.text.startswith("いまは、")
    assert not updated.text.startswith("…")
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]


def test_apply_inner_os_surface_profile_prefers_bright_short_sequence_when_turn_delta_is_bright() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "watch",
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "surface_voice_texture": "measured",
            "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
        },
    )
    assert updated is not None
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]


def test_apply_inner_os_surface_profile_keeps_small_bright_progression_balanced() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    harness._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "talk",
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それ、ちょっと笑えるやつだね。"},
                {"act": "light_bounce", "text": "そういうのあると、ちょっと楽になるよね。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.62,
                "playfulness": 0.54,
                "tempo": 0.48,
            },
            "inner_os_surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "shared_core": {
                    "already_shared": [
                        "あのあとちょっと笑えることもあって。",
                    ],
                },
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                    "listener_action_state": "warm_laugh_ack",
                },
            },
        },
    )

    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "brief",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "warm",
            "surface_pause_insertion": "none",
            "surface_certainty_style": "direct",
            "surface_voice_texture": "light_playful",
            "surface_context_packet": response.controls_used["inner_os_surface_context_packet"],
            "actuation_plan": {
                "execution_mode": "shared_progression",
                "reply_permission": "speak_briefly",
                "primary_action": "riff_current_comment",
            },
            "discourse_shape": response.controls_used["inner_os_discourse_shape"],
        },
    )

    assert updated is not None
    assert "I think we can take one next step from here." not in updated.text
    assert "笑えるやつ" in updated.text
    assert "ちょっと楽になるよね" in updated.text
    assert updated.controls_used["inner_os_surface_profile"]["response_length"] == "balanced"
    planned = updated.controls_used["inner_os_planned_content_sequence"]
    assert "笑えるやつ" in str(planned[0]["text"])
    assert "ちょっと楽になるよね" in str(planned[1]["text"])


def test_apply_inner_os_surface_profile_prefers_bright_sequence_from_discourse_shape_even_when_voice_is_measured() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "watch",
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": "いまは、ちょっと良かったね。",
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "surface_voice_texture": "measured",
            "discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.52,
                "playfulness": 0.34,
                "tempo": 0.44,
            },
        },
    )

    assert updated is not None
    assert "それは、ちょっといい感じだね。" in updated.text
    assert "大丈夫です" not in updated.text
    assert "話せるところから" not in updated.text
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]


def test_apply_inner_os_surface_profile_overrides_stale_reflect_shape_when_bright_signal_is_structural() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "watch",
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": "いま見えているところだけに絞って、一緒に見ていきます。",
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "reflect_step",
                "primary_move": "reflect",
                "secondary_move": "gentle_close",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "soft_close",
                "energy": "neutral",
                "brightness": 0.0,
                "playfulness": 0.0,
                "tempo": 0.0,
            },
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "surface_voice_texture": "light_playful",
            "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
            "surface_context_packet": coerce_surface_context_packet(
                {
                    "conversation_phase": "bright_continuity",
                    "surface_profile": {
                        "voice_texture": "light_playful",
                        "utterance_reason_offer": "brief_shared_smile",
                        "shared_moment_kind": "laugh",
                    },
                    "source_state": {
                        "utterance_reason_offer": "brief_shared_smile",
                        "shared_moment_kind": "laugh",
                        "live_engagement_state": "riff_with_comment",
                        "lightness_budget_state": "open_play",
                    },
                }
            ),
            "actuation_plan": {
                "execution_mode": "shared_progression",
                "reply_permission": "speak_briefly",
                "primary_action": "riff_current_comment",
            },
        },
    )

    assert updated is not None
    assert updated.controls_used["inner_os_discourse_shape"]["shape_id"] == "bright_bounce"
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]
    assert updated.controls_used["inner_os_allow_guarded_narrative_bridge"] is False
    assert updated.controls_used["inner_os_guarded_narrative_bridge_used"] is False
    assert "話せるところから" not in updated.text
    assert "いい感じ" in updated.text or "気持ちが軽くなる" in updated.text


def test_apply_inner_os_surface_profile_uses_current_surface_context_packet_for_listener_prefix() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    harness._last_surface_context_packet = coerce_surface_context_packet(  # type: ignore[attr-defined]
        {
            "source_state": {
                "listener_action_state": "soft_ack",
                "listener_token_profile": "soft_ack",
            }
        }
    )
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "mode": "watch",
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それ、ちょっと笑えるやつだね。"},
                {"act": "light_bounce", "text": "そういうのあると、ちょっと楽になるよね。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.56,
                "playfulness": 0.42,
                "tempo": 0.48,
            },
        },
    )
    updated = harness._apply_inner_os_surface_profile(
        response,
        {
            "surface_opening_delay": "measured",
            "surface_response_length": "short",
            "surface_sentence_temperature": "neutral",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "direct",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
            "surface_voice_texture": "light_playful",
            "surface_context_packet": coerce_surface_context_packet(
                {
                    "conversation_phase": "bright_continuity",
                    "surface_profile": {
                        "listener_action": "warm_laugh_ack",
                        "listener_token_profile": "soft_laugh",
                    },
                    "source_state": {
                        "listener_action_state": "warm_laugh_ack",
                        "listener_token_profile": "soft_laugh",
                    },
                }
            ),
        },
    )
    assert updated is not None
    assert updated.text.startswith("ふふっ、")
    assert updated.controls_used["inner_os_surface_context_packet"]["source_state"]["listener_token_profile"] == "soft_laugh"
    assert updated.controls_used["inner_os_discourse_shape"]["shape_id"] == "bright_bounce"
    assert updated.controls_used["inner_os_allow_guarded_narrative_bridge"] is False
    assert updated.controls_used["inner_os_guarded_narrative_bridge_used"] is False


def test_derive_runtime_discourse_shape_prefers_derived_bright_bounce_over_stale_reflect_shape() -> None:
    derived = runtime_module._derive_runtime_discourse_shape(
        content_sequence=[
            {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
        ],
        discourse_shape={
            "shape_id": "reflect_step",
            "primary_move": "reflect",
            "secondary_move": "gentle_close",
            "sentence_budget": 2,
            "question_budget": 0,
            "closing_mode": "soft_close",
            "energy": "neutral",
        },
        turn_delta={"kind": "bright_continuity", "preferred_act": "light_bounce"},
        surface_context_packet=coerce_surface_context_packet(
            {
                "conversation_phase": "bright_continuity",
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
            }
        ),
    )

    assert derived["shape_id"] == "bright_bounce"


def test_apply_interaction_policy_packet_to_current_state_uses_packet_as_source() -> None:
    current_state: Dict[str, Any] = {}
    packet = coerce_interaction_policy_packet(
        {
            "memory_write_class": "repair_trace",
            "memory_write_class_reason": "shared_repair",
            "agenda_state": {"state": "repair", "score": 0.62, "winner_margin": 0.2},
            "body_homeostasis_state": {"state": "recovering", "score": 0.48, "winner_margin": 0.14},
            "expressive_style_state": {"state": "grounded_gentle", "score": 0.55, "winner_margin": 0.18},
            "relation_competition_state": {"state": "single_bond", "winner_margin": 0.4},
            "social_topology_state": {"state": "one_to_one", "winner_margin": 0.5},
            "active_relation_table": {"entries": [{"person_id": "user"}], "total_people": 1},
        }
    )

    runtime_module._apply_interaction_policy_packet_to_current_state(
        current_state,
        interaction_policy_packet=packet,
    )

    assert current_state["interaction_policy_packet"]["memory_write_class"] == "repair_trace"
    assert current_state["memory_write_class"] == "repair_trace"
    assert current_state["agenda_state"]["state"] == "repair"
    assert current_state["body_homeostasis_state"]["state"] == "recovering"
    assert current_state["expressive_style_state"]["state"] == "grounded_gentle"
    assert current_state["social_topology_state"]["state"] == "one_to_one"
    assert current_state["active_relation_table"]["total_people"] == 1


def test_apply_interaction_policy_packet_to_gate_context_derives_carry_fields() -> None:
    gate_context: Dict[str, Any] = {}
    packet = coerce_interaction_policy_packet(
        {
            "response_strategy": "shared_world_next_step",
            "live_engagement_state": {"state": "pickup_comment"},
            "lightness_budget_state": {"state": "open_play"},
            "body_homeostasis_state": {"state": "recovering", "score": 0.6, "winner_margin": 0.25},
            "homeostasis_budget_state": {"state": "careful", "score": 0.5, "winner_margin": 0.2},
            "relational_continuity_state": {"state": "reopening", "score": 0.7, "winner_margin": 0.3},
            "social_topology_state": {"state": "one_to_one", "score": 0.8, "winner_margin": 0.2},
            "expressive_style_state": {"state": "warm_gentle", "score": 0.55, "winner_margin": 0.15},
            "relational_style_memory_state": {"banter_style": "gentle_play", "lexical_variation_bias": 0.4},
            "agenda_state": {"state": "repair", "score": 0.6, "winner_margin": 0.2, "reason": "repair thread"},
            "agenda_window_state": {"state": "near", "score": 0.45, "winner_margin": 0.1, "reason": "window open"},
            "learning_mode_state": {"state": "observe_and_adjust", "score": 0.4, "winner_margin": 0.1},
            "social_experiment_loop_state": {"state": "watch_and_read", "score": 0.35, "winner_margin": 0.1, "probe_intensity": 0.2},
            "commitment_carry": {
                "target_focus": "repair",
                "state_focus": "settle",
                "carry_bias": 0.42,
                "followup_focus": "stay_with_thread",
                "mode_focus": "gentle",
                "carry_reason": "shared_repair",
            },
            "relation_competition_state": {"state": "single_bond", "winner_margin": 0.3},
            "active_relation_table": {"entries": [{"person_id": "user"}], "total_people": 1},
            "overnight_bias_roles": {"agenda_state": "repair"},
            "reaction_vs_overnight_bias": {"same_turn": {"agenda_state": "repair"}},
        }
    )

    runtime_module._apply_interaction_policy_packet_to_gate_context(
        gate_context,
        interaction_policy_packet=packet,
        current_state={},
        initiative_followup_bias={"score": 0.33, "state": "offer"},
    )

    assert gate_context["inner_os_interaction_policy_packet"]["response_strategy"] == "shared_world_next_step"
    assert gate_context["inner_os_response_strategy"] == "shared_world_next_step"
    assert gate_context["inner_os_body_homeostasis_focus"] == "recovering"
    assert gate_context["inner_os_body_homeostasis_carry_bias"] > 0.0
    assert gate_context["inner_os_homeostasis_budget_focus"] == "careful"
    assert gate_context["inner_os_relational_continuity_focus"] == "reopening"
    assert gate_context["inner_os_group_thread_focus"] == "one_to_one"
    assert gate_context["inner_os_expressive_style_focus"] == "warm_gentle"
    assert gate_context["inner_os_banter_style_focus"] == "gentle_play"
    assert gate_context["inner_os_agenda_focus"] == "repair"
    assert gate_context["inner_os_agenda_window_focus"] == "near"
    assert gate_context["inner_os_commitment_target_focus"] == "repair"
    assert gate_context["inner_os_initiative_followup_state"] == "offer"


def test_live_response_loop_refreshes_persistent_plan_when_bright_sequence_is_shaped(monkeypatch) -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    original_gate_builder = runtime_module.build_expression_hints_from_gate_result
    original_surface_policy = harness._apply_inner_os_surface_policy
    original_surface_profile = harness._apply_inner_os_surface_profile
    original_style_feedback = harness._inner_os_live_style_feedback
    original_followup_signals = harness._inner_os_live_followup_signals
    original_stream_state = harness._inner_os_stream_state_from_hints
    original_stream_converged = harness._inner_os_stream_converged

    monkeypatch.setattr(
        runtime_module,
        "build_expression_hints_from_gate_result",
        lambda hook, existing_hints, expected_source: dict(existing_hints),
    )
    harness._apply_inner_os_surface_policy = lambda response, expression_hints, conscious_access: response  # type: ignore[assignment]

    def _fake_surface_profile(response, expression_hints):
        controls_used = dict(getattr(response, "controls_used", {}) or {})
        controls_used["inner_os_planned_content_sequence"] = [
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
        ]
        response.controls_used = controls_used
        response.text = "それは、ちょっといい感じだね。 その感じ、ちょっと嬉しいね。"
        return response

    harness._apply_inner_os_surface_profile = _fake_surface_profile  # type: ignore[assignment]
    harness._inner_os_live_style_feedback = lambda current_hints, next_hints: {}  # type: ignore[assignment]
    harness._inner_os_live_followup_signals = lambda expression_hints, safety_bias: {}  # type: ignore[assignment]
    harness._inner_os_stream_state_from_hints = lambda expression_hints: {}  # type: ignore[assignment]
    harness._inner_os_stream_converged = lambda previous, current: True  # type: ignore[assignment]

    def _response_gate(*, draft, current_state, safety_signals):
        return SimpleNamespace(
            expression_hints={
                "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
                "surface_context_packet": {
                    "conversation_phase": "bright_continuity",
                    "surface_profile": {"voice_texture": "light_playful"},
                },
                "surface_voice_texture": "light_playful",
                "lightness_budget_state_name": "open_play",
            },
            conscious_access={},
        )

    harness._integration_hooks = SimpleNamespace(response_gate=_response_gate)
    initial_hook = SimpleNamespace(
        expression_hints={
            "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
            "surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "surface_profile": {"voice_texture": "light_playful"},
            },
            "surface_voice_texture": "light_playful",
            "lightness_budget_state_name": "open_play",
        },
        conscious_access={},
    )
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ]
        },
    )

    try:
        updated, _hook, _steps = harness._run_inner_os_live_response_loop(
            response=response,
            initial_hook=initial_hook,
            current_state={},
            safety_bias=0.2,
        )
    finally:
        runtime_module.build_expression_hints_from_gate_result = original_gate_builder
        harness._apply_inner_os_surface_policy = original_surface_policy  # type: ignore[assignment]
        harness._apply_inner_os_surface_profile = original_surface_profile  # type: ignore[assignment]
        harness._inner_os_live_style_feedback = original_style_feedback  # type: ignore[assignment]
        harness._inner_os_live_followup_signals = original_followup_signals  # type: ignore[assignment]
        harness._inner_os_stream_state_from_hints = original_stream_state  # type: ignore[assignment]
        harness._inner_os_stream_converged = original_stream_converged  # type: ignore[assignment]

    assert updated is not None
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]


def test_live_response_loop_replaces_stale_reflect_shape_with_bright_guidance(monkeypatch) -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    original_gate_builder = runtime_module.build_expression_hints_from_gate_result
    original_surface_policy = harness._apply_inner_os_surface_policy
    original_surface_profile = harness._apply_inner_os_surface_profile
    original_style_feedback = harness._inner_os_live_style_feedback
    original_followup_signals = harness._inner_os_live_followup_signals
    original_stream_state = harness._inner_os_stream_state_from_hints
    original_stream_converged = harness._inner_os_stream_converged

    monkeypatch.setattr(
        runtime_module,
        "build_expression_hints_from_gate_result",
        lambda hook, existing_hints, expected_source: dict(existing_hints),
    )
    harness._apply_inner_os_surface_policy = lambda response, expression_hints, conscious_access: response  # type: ignore[assignment]
    harness._apply_inner_os_surface_profile = lambda response, expression_hints: response  # type: ignore[assignment]
    harness._inner_os_live_style_feedback = lambda current_hints, next_hints: {}  # type: ignore[assignment]
    harness._inner_os_live_followup_signals = lambda expression_hints, safety_bias: {}  # type: ignore[assignment]
    harness._inner_os_stream_state_from_hints = lambda expression_hints: {}  # type: ignore[assignment]
    harness._inner_os_stream_converged = lambda previous, current: True  # type: ignore[assignment]
    harness._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"  # type: ignore[attr-defined]
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    def _response_gate(*, draft, current_state, safety_signals):
        return SimpleNamespace(
            expression_hints={
                "interaction_policy_packet": {
                    "dialogue_act": "check_in",
                    "recent_dialogue_state": {"state": "continuing_thread"},
                    "discussion_thread_state": {"state": "open_thread"},
                    "issue_state": {"state": "light_tension"},
                    "live_engagement_state": {"state": "pickup_comment"},
                    "lightness_budget_state": {"state": "open_play"},
                },
                "interaction_policy_dialogue_act": "check_in",
                "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
                "interaction_constraints": {
                    "allow_small_next_step": True,
                    "keep_thread_visible": True,
                    "max_questions": 1,
                },
                "surface_context_packet": {
                    "conversation_phase": "bright_continuity",
                    "response_role": {"primary": "light_bounce"},
                    "constraints": {
                        "allow_small_next_step": True,
                        "keep_thread_visible": True,
                        "max_questions": 1,
                    },
                    "surface_profile": {
                        "voice_texture": "light_playful",
                        "response_length": "short",
                    },
                    "source_state": {
                        "lightness_budget_state": "open_play",
                        "live_engagement_state": "pickup_comment",
                    },
                },
                "surface_voice_texture": "light_playful",
                "surface_response_length": "short",
                "lightness_budget_state_name": "open_play",
                "live_engagement_state_name": "pickup_comment",
            },
            conscious_access={},
        )

    harness._integration_hooks = SimpleNamespace(response_gate=_response_gate)
    initial_hook = _response_gate(draft={}, current_state={}, safety_signals={})
    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "inner_os_planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
                {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "reflect_step",
                "sentence_budget": 2,
                "question_budget": 1,
                "keep_thread_visible": True,
            },
        },
    )

    try:
        updated, _hook, _steps = harness._run_inner_os_live_response_loop(
            response=response,
            initial_hook=initial_hook,
            current_state={},
            safety_bias=0.2,
        )
    finally:
        runtime_module.build_expression_hints_from_gate_result = original_gate_builder
        harness._apply_inner_os_surface_policy = original_surface_policy  # type: ignore[assignment]
        harness._apply_inner_os_surface_profile = original_surface_profile  # type: ignore[assignment]
        harness._inner_os_live_style_feedback = original_style_feedback  # type: ignore[assignment]
        harness._inner_os_live_followup_signals = original_followup_signals  # type: ignore[assignment]
        harness._inner_os_stream_state_from_hints = original_stream_state  # type: ignore[assignment]
        harness._inner_os_stream_converged = original_stream_converged  # type: ignore[assignment]

    assert updated is not None
    assert updated.controls_used["inner_os_discourse_shape"]["shape_id"] == "bright_bounce"
    acts = [
        str(item.get("act") or "").strip()
        for item in updated.controls_used["inner_os_planned_content_sequence"]
    ]
    assert acts == ["shared_delight", "light_bounce"]


def test_shape_inner_os_surface_profile_text_applies_surface_language_profile() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    compact = harness._shape_inner_os_surface_profile_text(
        "I am trying to stay with what is grounded.",
        surface_profile={
            "banter_move": "compact_wit",
            "lexical_variation_mode": "compact_varied",
            "group_register": "one_to_one",
        },
    )
    group_text = harness._shape_inner_os_surface_profile_text(
        "I can stay with what is visible first.",
        surface_profile={
            "banter_move": "thread_soften",
            "lexical_variation_mode": "group_attuned",
            "group_register": "threaded_group",
        },
    )

    assert compact.startswith("Short version:")
    assert "I'm trying" in compact
    assert group_text.startswith("For this thread,")


def test_apply_inner_os_surface_policy_adds_relational_probe_for_check_in_afterglow() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.36, "carry_gentleness": True, "interaction_afterglow_intent": "check_in"},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert "Would it help if I stay close to what is visible first?" in updated.text
    assert updated.text.startswith("Let me check in gently before I go further.")


def test_apply_inner_os_surface_policy_adds_reopening_line_when_recovery_returns() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.3, "allow_reopening": True, "recovery_reopening": 0.31},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert "I think we can open this a little more clearly now." in updated.text


def test_apply_inner_os_surface_policy_bypasses_clarify_prefix_for_bright_bounce() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(
        text="それは、ちょっといい感じだね。 その感じ、ちょっと嬉しいね。",
        controls_used={
            "mode": "watch",
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ],
        },
    )
    updated = harness._apply_inner_os_surface_policy(
        response,
        {
            "clarify_first": True,
            "turn_delta": {"kind": "bright_continuity", "preferred_act": "light_bounce"},
            "surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "surface_profile": {"voice_texture": "light_playful"},
            },
            "discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.52,
                "playfulness": 0.34,
                "tempo": 0.44,
            },
        },
        {"intent": "clarify"},
    )

    assert updated is not None
    assert updated.text == "それは、ちょっといい感じだね。 その感じ、ちょっと嬉しいね。"
    assert updated.controls_used["inner_os_surface_policy"] == "bypass:bright_bounce"


def test_apply_qualia_gate_prefers_response_shape_over_stale_runtime_plan() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._qualia_vector_to_array = lambda qualia: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)  # type: ignore[attr-defined]
    harness._prev_qualia_vec = None  # type: ignore[attr-defined]
    harness._qualia_gate_enabled = True  # type: ignore[attr-defined]
    harness._qualia_meta = SimpleNamespace(compute=lambda pred, current: 0.12)  # type: ignore[attr-defined]
    harness._qualia_gate = SimpleNamespace(decide=lambda **kwargs: {"allow": False, "reason": "normal"})  # type: ignore[attr-defined]
    harness._talk_mode = runtime_module.TalkMode.WATCH  # type: ignore[attr-defined]
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    harness._wrap_ack_response = lambda text, talk_mode, controls_used: SimpleNamespace(  # type: ignore[attr-defined]
        text=text,
        controls_used=controls_used,
        latency_ms=0.0,
        safety=None,
        retrieval_summary=None,
        perception_summary=None,
    )
    harness._render_ack_for_mode = lambda mode, gate_ctx: "ack"  # type: ignore[attr-defined]
    harness._render_presence_ack = lambda gate_ctx: "presence"  # type: ignore[attr-defined]
    harness._last_gate_context = {}
    harness._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"  # type: ignore[attr-defined]
    harness._last_planned_content_sequence = [  # type: ignore[attr-defined]
        {"act": "quiet_presence", "text": "話せるところからでいいよ。"},
    ]

    response = SimpleNamespace(
        text="placeholder",
        controls_used={
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.52,
                "playfulness": 0.34,
                "tempo": 0.44,
            },
            "inner_os_surface_profile": {
                "response_length": "short",
                "voice_texture": "measured",
                "sentence_temperature": "neutral",
                "pause_insertion": "soft_pause",
                "certainty_style": "direct",
            },
            "inner_os_surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "shared_core": {
                    "already_shared": [
                        "あのあとちょっと笑えることもあって。",
                    ],
                },
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                    "listener_action_state": "warm_laugh_ack",
                },
            },
        },
    )

    updated, narrative = runtime_module.EmotionalHubRuntime._apply_qualia_gate(
        harness,
        response=response,
        narrative_text="raw narrative",
        qualia_vec=SimpleNamespace(valence=0.0, arousal=0.0, love=0.0, stress=0.0, mask=0.0),
        gate_ctx=SimpleNamespace(delta_m=0.0, jerk=0.0, text_input=True, force_listen=False),
        boundary_signal=SimpleNamespace(score=0.0),
        ack_text=None,
        ack_for_fast=None,
        metrics={},
        prediction_error=0.0,
    )

    assert updated is not None
    assert "笑えるやつ" in updated.text
    assert "話せるところからでいいよ。" not in updated.text
    assert updated.controls_used["inner_os_discourse_shape"]["shape_id"] == "bright_bounce"
    planned = updated.controls_used["inner_os_planned_content_sequence"]
    assert "笑えるやつ" in str(planned[0]["text"])
    assert "ちょっと楽になるよね" in str(planned[1]["text"])
    assert updated.controls_used["inner_os_qualia_gate_suppressed"] is True
    assert narrative is None


def test_effective_interaction_policy_packet_promotes_shared_world_next_step_for_small_bright_progression() -> None:
    effective = runtime_module._derive_effective_interaction_policy_packet(
        {
            "response_strategy": "attune_then_extend",
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "reopen_pressure": 0.24,
                "recent_anchor": "前の流れはまだしんどさが残っていた",
            },
        },
        response_controls_used={
            "inner_os_discourse_shape": {"shape_id": "bright_bounce"},
            "inner_os_surface_profile": {
                "response_length": "balanced",
            },
            "inner_os_surface_context_packet": {
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
            },
        },
        actuation_plan={
            "execution_mode": "shared_progression",
            "primary_action": "riff_current_comment",
        },
    )

    assert effective["response_strategy"] == "shared_world_next_step"
    assert effective["opening_move"] == "synchronize_then_propose"
    assert effective["recent_dialogue_state"]["state"] == "continuing_thread"
    assert effective["recent_dialogue_state"]["reopen_pressure"] == 0.18


def test_effective_talk_mode_promotes_shared_progression_to_talk() -> None:
    talk_mode = runtime_module._derive_effective_talk_mode_name(
        raw_talk_mode="watch",
        interaction_policy_packet={
            "response_strategy": "shared_world_next_step",
        },
        actuation_plan={
            "execution_mode": "shared_progression",
            "primary_action": "riff_current_comment",
        },
        response_text="ふふっ、それ、ちょっと笑えるやつだね。",
    )

    assert talk_mode == "talk"


def test_effective_response_controls_used_rewrites_small_bright_sequence_to_cue_aware() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    harness._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"  # type: ignore[attr-defined]

    response = SimpleNamespace(
        text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
        controls_used={
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
            ],
            "inner_os_discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.52,
                "playfulness": 0.34,
                "tempo": 0.44,
            },
            "inner_os_surface_profile": {
                "response_length": "balanced",
                "voice_texture": "light_playful",
                "sentence_temperature": "neutral",
                "pause_insertion": "none",
                "certainty_style": "",
            },
            "inner_os_surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "shared_core": {
                    "already_shared": [
                        "あのあとちょっと笑えることもあって。",
                    ],
                },
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                    "listener_action_state": "warm_laugh_ack",
                    "listener_token_profile": "soft_laugh",
                },
            },
        },
    )

    effective = runtime_module._derive_effective_response_controls_used(
        harness,
        response=response,
        expression_hints={
            "turn_delta": {
                "kind": "bright_continuity",
                "preferred_act": "light_bounce",
            },
            "surface_response_length": "balanced",
            "surface_voice_texture": "light_playful",
        },
    )

    planned = effective["inner_os_planned_content_sequence"]
    assert str(planned[0]["text"]).startswith("ふふっ、")
    assert "笑えるやつ" in str(planned[0]["text"])
    assert "ちょっと楽になるよね" in str(planned[1]["text"])
    assert effective["inner_os_surface_profile"]["content_sequence_length"] == 2
    assert effective["inner_os_discourse_shape"]["shape_id"] == "bright_bounce"


def test_process_turn_injects_inner_os_metadata(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_social_update_strength"] = 0.64
    harness._last_gate_context["inner_os_identity_update_strength"] = 0.52
    harness._last_gate_context["inner_os_interaction_afterglow"] = 0.31
    harness._last_gate_context["inner_os_interaction_afterglow_intent"] = "clarify"
    harness._last_gate_context["inner_os_association_reweighting_focus"] = "repeated_links"
    harness._last_gate_context["inner_os_association_reweighting_reason"] = "repeated_insight_trace"
    harness._last_gate_context["inner_os_insight_terrain_shape_target"] = "soft_relation"
    result = harness.process_turn(
        user_text="do you remember this new place",
        context="previous framing",
        transferred_lessons=[
            {
                "kind": "transferred_learning",
                "policy_hint": "pause_and_observe_under_ambiguity",
                "summary": "observe before approaching when ambiguity is high",
            }
        ],
    )
    assert result.metrics["inner_os/stress"] >= 0.0
    assert result.metrics["inner_os/temporal_pressure_after"] >= 0.0
    assert result.metrics["inner_os/transferred_lessons_used"] == 1.0
    assert result.metrics["inner_os/interaction_afterglow"] > 0.0
    assert result.metrics["inner_os/replay_intensity"] >= 0.0
    assert result.metrics["inner_os/anticipation_tension"] >= 0.0
    assert result.metrics["inner_os/recovery_reopening"] >= 0.0
    assert result.metrics["inner_os/consolidation_priority"] >= 0.0
    assert result.metrics["inner_os/interference_pressure"] >= 0.0
    assert result.metrics["inner_os/prospective_memory_pull"] >= 0.0
    assert result.metrics["inner_os/object_affordance_bias"] >= 0.0
    assert result.metrics["inner_os/defensive_salience"] >= 0.0
    assert result.metrics["inner_os/surface_policy_active"] == 1.0
    assert result.metrics["inner_os/surface_policy_layered"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_clarify"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_check_in"] == 0.0
    assert result.response is not None
    assert result.response.text.startswith("Let me check one small thing before I go further.")
    assert (
        "Then I can answer a little more cleanly from there." in result.response.text
        or "I can stay with what is visible first." in result.response.text
    )
    assert result.response.retrieval_summary is not None
    assert "inner_os" in result.response.retrieval_summary
    assert result.response.controls is not None
    assert "inner_os" in result.response.controls
    assert 2 <= result.response.controls["inner_os"]["live_response_steps"] <= 4
    assert result.response.controls["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.response.controls["inner_os"]["headless_actuation"]["execution_mode"] in {"attuned_contact", "defer_with_presence", "repair_contact", "shared_progression", "stabilize_boundary", "stabilize_before_contact", "open_reflection"}
    assert result.response.controls["inner_os"]["headless_actuation"]["primary_action"]
    assert result.response.controls["inner_os"]["headless_actuation"]["response_channel"] in {"speak", "backchannel", "hold", "defer"}
    assert "turn_timing_hint" in result.response.controls["inner_os"]["headless_actuation"]
    assert result.response.controls["inner_os"]["headless_actuation"]["turn_timing_hint"]["minimum_wait_ms"] >= 0
    assert result.response.controls_used["inner_os_emit_timing"]["minimum_wait_ms"] >= 0
    assert result.response.controls_used["inner_os_emit_timing"]["effective_latency_ms"] >= 0
    assert result.response.controls_used["inner_os_emit_timing"]["interrupt_guard_ms"] >= 0
    assert result.response.controls_used["inner_os_emit_timing"]["emit_not_before_ms"] >= 0
    assert result.response.controls_used["inner_os_emit_timing"]["interrupt_guard_until_ms"] >= result.response.controls_used["inner_os_emit_timing"]["emit_not_before_ms"]
    assert isinstance(result.response.controls_used["inner_os_timing_guard"], dict)
    assert result.response.controls_used["inner_os_gate_force_listen"] in {True, False}
    if result.response.controls_used["inner_os_emit_timing"]["response_channel"] in {"backchannel", "hold", "defer"}:
        assert result.response.latency_ms >= result.response.controls_used["inner_os_emit_timing"]["minimum_wait_ms"]
    assert result.response.controls_used["inner_os_surface_profile"]["certainty_style"] in {"direct", "tentative", "careful"}
    assert result.response.controls_used["inner_os_surface_profile"]["content_sequence_length"] >= 1
    assert result.response.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"
    assert "inner_os" in result.qualia_gate
    assert result.persona_meta["inner_os"]["route"] in {"reflex", "conscious", "watch"}
    assert result.persona_meta["inner_os"]["culture_id"] == "coastal"
    assert result.persona_meta["inner_os"]["community_id"] == "harbor_collective"
    assert result.response.controls["inner_os"]["hesitation_bias"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["tentative_bias"] >= 0.0
    assert round(result.response.controls["inner_os"]["expression_hints"]["express_now"] + result.response.controls["inner_os"]["expression_hints"]["hold_back"], 4) == 1.0
    assert result.persona_meta["inner_os"]["continuity_score"] >= 0.0
    assert result.persona_meta["inner_os"]["culture_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["community_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["terrain_transition_roughness"] >= 0.0
    assert result.persona_meta["inner_os"]["meaning_pacing"] in {"steady", "slow"}
    assert result.persona_meta["inner_os"]["interaction_pacing"] in {"flow", "check"}
    assert result.persona_meta["inner_os"]["actuation_execution_mode"] in {"attuned_contact", "defer_with_presence", "repair_contact", "shared_progression", "stabilize_boundary", "stabilize_before_contact", "open_reflection"}
    assert result.persona_meta["inner_os"]["actuation_primary_action"]
    assert result.persona_meta["inner_os"]["actuation_response_channel"] in {"speak", "backchannel", "hold", "defer"}
    assert result.persona_meta["inner_os"]["actuation_reply_permission"] in {"speak", "speak_briefly", "hold_or_brief", "speak_forward", "speak_minimal", "speak_reflective", "speak_expand"}
    assert result.persona_meta["inner_os"]["actuation_turn_timing_hint"]["minimum_wait_ms"] >= 0
    assert result.persona_meta["inner_os"]["actuation_emit_timing"]["effective_latency_ms"] >= 0
    assert result.persona_meta["inner_os"]["actuation_emit_timing"]["interrupt_guard_ms"] >= 0
    assert result.persona_meta["inner_os"]["actuation_emit_timing"]["emit_not_before_ms"] >= 0
    assert isinstance(result.persona_meta["inner_os"]["timing_guard"], dict)
    assert result.persona_meta["inner_os"]["gate_force_listen"] in {True, False}
    assert result.persona_meta["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.persona_meta["inner_os"]["surface_policy_intent"] == "clarify"
    assert isinstance(result.persona_meta["inner_os"]["scene_family"], str)
    assert isinstance(result.persona_meta["inner_os"]["top_interaction_option_family"], str)
    assert result.persona_meta["inner_os"]["interaction_option_candidate_count"] >= 0
    assert result.persona_meta["inner_os"]["conscious_workspace_mode"] in {"preconscious", "latent_foreground", "foreground", "guarded_foreground", ""}
    assert isinstance(result.persona_meta["inner_os"]["audit_reference_case_ids"], list)
    assert isinstance(result.persona_meta["inner_os"]["audit_reference_case_meta"], dict)
    assert result.response.controls["inner_os"]["expression_hints"]["surface_opening_delay"] in {"brief", "measured", "long"}
    assert result.response.controls["inner_os"]["expression_hints"]["surface_response_length"] in {"short", "balanced", "reflective", "forward_leaning"}
    assert result.response.controls["inner_os"]["expression_hints"]["surface_certainty_style"] in {"direct", "tentative", "careful"}
    assert result.response.controls["inner_os"]["expression_hints"]["stream_update_count"] >= 2
    assert result.response.controls["inner_os"]["expression_hints"]["stream_shared_attention_window_mean"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["stream_strained_pause_window_mean"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["stream_repair_window_hold"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["opening_pace_windowed"] in {"ready", "measured", "held"}
    assert result.response.controls["inner_os"]["expression_hints"]["return_gaze_expectation"] in {"soft_return", "steady_return", "careful_return", "defer_return"}
    assert result.response.controls["inner_os"]["expression_hints"]["live_opening_pace_mismatch"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["live_return_gaze_mismatch"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["live_style_alignment"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["contact_field"]["points"]
    assert result.response.controls["inner_os"]["expression_hints"]["contact_dynamics"]["stabilized_points"]
    assert result.response.controls["inner_os"]["expression_hints"]["contact_reflection_state"]["reflection_style"] in {"reflect_then_question", "reflect_only", "boundary_only"}
    assert "green_kernel_composition" in result.response.controls["inner_os"]["expression_hints"]
    assert "field" in result.response.controls["inner_os"]["expression_hints"]["green_kernel_composition"]
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["qualia"]
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["gate"]
    assert len(result.response.controls["inner_os"]["expression_hints"]["qualia_state"]["qualia"]) == len(
        result.response.controls["inner_os"]["expression_hints"]["qualia_axis_labels"]
    )
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_estimator_health"]["trust"] >= 0.0
    assert isinstance(result.response.controls["inner_os"]["expression_hints"]["qualia_protection_grad_x"], list)
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_source"] == "shared"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_version"] >= 1
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_expected_source"] == "shared"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_expected_mismatch"] is False
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"]["trust"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"]["felt_energy"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_bundle"]["qualia_hint_source"] == "shared"
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_hint_bundle"]["qualia_planner_view"] == result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"]
    assert result.response.controls["inner_os"]["expression_hints"]["qualia_planner_view"] == build_expression_hints_from_gate_result(
        result.qualia_gate["inner_os"],
        existing_hints=result.qualia_gate["inner_os"]["expression_hints"],
    )["qualia_planner_view"]
    assert result.response.controls["inner_os"]["expression_hints"]["access_projection"]["regions"]
    assert "access_qualia_input" in result.response.controls["inner_os"]["expression_hints"]["access_projection"]["cues"]
    assert result.response.controls["inner_os"]["expression_hints"]["access_dynamics"]["stabilized_regions"]
    assert result.response.controls["inner_os"]["expression_hints"]["conscious_workspace"]["workspace_mode"] in {"preconscious", "latent_foreground", "foreground", "guarded_foreground"}
    assert isinstance(result.response.controls["inner_os"]["expression_hints"]["conscious_workspace_actionable_slice"], list)
    assert result.response.controls["inner_os"]["expression_hints"]["conscious_workspace_ignition_phase"] in {"dormant", "priming", "ignited", "guarded"}
    assert result.response.controls["inner_os"]["expression_hints"]["scene_hint_bundle"]["scene_state"] == result.response.controls["inner_os"]["expression_hints"]["scene_state"]
    assert result.response.controls["inner_os"]["expression_hints"]["scene_hint_bundle"]["scene_family"] == result.response.controls["inner_os"]["expression_hints"]["scene_family"]
    assert result.response.controls["inner_os"]["expression_hints"]["workspace_hint_bundle"]["conscious_workspace"] == result.response.controls["inner_os"]["expression_hints"]["conscious_workspace"]
    assert result.response.controls["inner_os"]["expression_hints"]["workspace_hint_bundle"]["conscious_workspace_mode"] == result.response.controls["inner_os"]["expression_hints"]["conscious_workspace_mode"]
    assert result.response.controls["inner_os"]["expression_hints"]["conversational_objects"]["objects"]
    assert result.response.controls["inner_os"]["expression_hints"]["object_operations"]["operations"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_effects"]["effects"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_reasoning_hint_bundle"]["conversational_objects"] == result.response.controls["inner_os"]["expression_hints"]["conversational_objects"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_reasoning_hint_bundle"]["object_operations"] == result.response.controls["inner_os"]["expression_hints"]["object_operations"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_view"]["observed_signals"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_view"]["inferred_signals"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_summary"]["observed_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_summary"]["operation_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_condition_report"]["scene_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_condition_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_inspection_report"]["case_reports"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_inspection_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_reasoning_hint_bundle"]["interaction_judgement_view"] == result.response.controls["inner_os"]["expression_hints"]["interaction_judgement_view"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_reasoning_hint_bundle"]["interaction_condition_report"] == result.response.controls["inner_os"]["expression_hints"]["interaction_condition_report"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_bundle"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_bundle"]["key_metrics"]["question_budget"] >= 0
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_casebook"]["cases"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_report"]["report_lines"]
    assert result.response.controls["inner_os"]["expression_hints"]["interaction_audit_hint_bundle"]["interaction_audit_bundle"] == result.response.controls["inner_os"]["expression_hints"]["interaction_audit_bundle"]
    assert result.response.controls["inner_os"]["expression_hints"]["field_regulation_hint_bundle"]["contact_dynamics"] == result.response.controls["inner_os"]["expression_hints"]["contact_dynamics"]
    assert result.response.controls["inner_os"]["expression_hints"]["field_regulation_hint_bundle"]["access_dynamics"] == result.response.controls["inner_os"]["expression_hints"]["access_dynamics"]
    assert result.response.controls["inner_os"]["expression_hints"]["terrain_insight_hint_bundle"]["terrain_readout"] == result.response.controls["inner_os"]["expression_hints"]["terrain_readout"]
    assert result.response.controls["inner_os"]["expression_hints"]["terrain_insight_hint_bundle"]["resonance_evaluation"] == result.response.controls["inner_os"]["expression_hints"]["resonance_evaluation"]
    assert isinstance(harness._last_gate_context["inner_os_scene_hint_bundle"], SceneHintBundleContract)
    assert isinstance(harness._last_gate_context["inner_os_workspace_hint_bundle"], WorkspaceHintBundleContract)
    assert isinstance(harness._last_gate_context["inner_os_field_regulation_hint_bundle"], FieldRegulationHintBundleContract)
    assert isinstance(harness._last_gate_context["inner_os_terrain_insight_hint_bundle"], TerrainInsightHintBundleContract)
    assert isinstance(
        harness._last_gate_context["inner_os_qualia_hint_bundle"],
        QualiaHintBundleContract,
    )
    assert isinstance(
        harness._last_gate_context["inner_os_interaction_reasoning_hint_bundle"],
        InteractionReasoningHintBundleContract,
    )
    assert isinstance(
        harness._last_gate_context["inner_os_interaction_audit_hint_bundle"],
        InteractionAuditHintBundleContract,
    )
    assert harness._last_gate_context["inner_os_scene_family"] == harness._last_gate_context["inner_os_scene_hint_bundle"]["scene_family"]
    assert harness._last_gate_context["inner_os_conscious_workspace_mode"] == harness._last_gate_context["inner_os_workspace_hint_bundle"]["conscious_workspace_mode"]
    assert harness._last_gate_context["inner_os_conversational_objects"] == harness._last_gate_context["inner_os_interaction_reasoning_hint_bundle"]["conversational_objects"]
    assert harness._last_gate_context["inner_os_interaction_audit_reference_case_ids"] == harness._last_gate_context["inner_os_interaction_audit_hint_bundle"]["interaction_audit_reference_case_ids"]
    assert harness._last_gate_context["inner_os_qualia_hint_source"] == harness._last_gate_context["inner_os_qualia_hint_bundle"]["qualia_hint_source"]
    assert harness._last_gate_context["inner_os_contact_dynamics"] == harness._last_gate_context["inner_os_field_regulation_hint_bundle"]["contact_dynamics"]
    assert harness._last_gate_context["inner_os_resonance_evaluation"] == harness._last_gate_context["inner_os_terrain_insight_hint_bundle"]["resonance_evaluation"]
    assert result.response.controls["inner_os"]["expression_hints"]["affective_position"]["z_aff"]
    assert result.response.controls["inner_os"]["expression_hints"]["terrain_readout"]["active_patch_label"]
    assert result.response.controls["inner_os"]["expression_hints"]["protection_mode"]["mode"] in {"monitor", "contain", "stabilize", "repair", "shield"}
    assert result.response.controls["inner_os"]["expression_hints"]["association_graph"]["state_hint"]["link_weights"] is not None
    assert result.response.controls["inner_os"]["expression_hints"]["insight_event"]["score"]["total"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["resonance_evaluation"]["recommended_family_id"]
    assert result.response.controls["inner_os"]["expression_hints"]["resonance_evaluation"]["estimated_other_person_state"]["detail_room_level"] in {"low", "medium", "high"}
    assert result.response.controls_used["inner_os_surface_profile"]["live_opening_pace_mismatch"] >= 0.0
    assert result.response.controls_used["inner_os_surface_profile"]["live_return_gaze_mismatch"] >= 0.0
    assert result.metrics["inner_os/stream_update_count"] >= 2.0
    assert result.metrics["inner_os/headless_wait_required"] in {0.0, 1.0}
    assert result.metrics["inner_os/headless_wait_ms"] >= 0.0
    assert result.metrics["inner_os/headless_interrupt_guard_ms"] >= 0.0
    assert result.metrics["inner_os/effective_emit_delay_ms"] >= 0.0
    assert result.metrics["inner_os/effective_latency_ms"] >= 0.0
    assert result.metrics["inner_os/emit_wait_applied"] in {0.0, 1.0}
    assert result.metrics["inner_os/emit_wait_applied_ms"] >= 0.0
    assert result.metrics["inner_os/timing_guard_active"] in {0.0, 1.0}
    assert result.metrics["inner_os/timing_guard_emit_delay"] in {0.0, 1.0}
    assert result.metrics["inner_os/timing_guard_interrupt_guard"] in {0.0, 1.0}
    assert result.metrics["inner_os/resonance_score"] >= 0.0
    assert result.metrics["inner_os/qualia_hint_shared"] == 1.0
    assert result.metrics["inner_os/qualia_hint_fallback"] == 0.0
    assert result.metrics["inner_os/qualia_hint_none"] == 0.0
    assert result.metrics["inner_os/overnight_association_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_terrain_shape_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_bias_alignment_visible"] == 1.0
    assert result.metrics["inner_os/memory_write_winner_margin"] >= 0.0
    assert result.metrics["inner_os/protection_mode_winner_margin"] >= 0.0
    assert result.metrics["inner_os/memory_write_mode_prior_active"] == 1.0
    assert result.metrics["inner_os/memory_write_insight_prior_active"] in {0.0, 1.0}
    assert result.metrics["inner_os/workspace_winner_margin"] >= 0.0
    assert result.metrics["inner_os/association_graph_winner_margin"] >= 0.0
    assert result.metrics["inner_os/body_recovery_guard_winner_margin"] >= 0.0
    assert result.metrics["inner_os/body_homeostasis_score"] >= 0.0
    assert result.metrics["inner_os/body_homeostasis_winner_margin"] >= 0.0
    assert result.metrics["inner_os/initiative_readiness"] >= 0.0
    assert result.metrics["inner_os/commitment_winner_margin"] >= 0.0
    assert result.metrics["inner_os/relational_style_lexical_variation_bias"] >= 0.0
    assert result.metrics["inner_os/relational_style_banter_room"] >= 0.0
    assert result.metrics["inner_os/relational_continuity_score"] >= 0.0
    assert result.metrics["inner_os/relational_continuity_winner_margin"] >= 0.0
    assert result.metrics["inner_os/relation_competition_level"] >= 0.0
    assert result.metrics["inner_os/relation_competition_winner_margin"] >= 0.0
    assert result.metrics["inner_os/social_topology_score"] >= 0.0
    assert result.metrics["inner_os/social_topology_winner_margin"] >= 0.0
    assert result.metrics["inner_os/growth_relational_trust"] >= 0.0
    assert result.metrics["inner_os/growth_self_coherence"] >= 0.0
    serialized_meta = harness._serialize_response_meta(result.response)
    assert serialized_meta is not None
    assert isinstance(serialized_meta["actuation_emit_timing"], dict)
    assert isinstance(serialized_meta["timing_guard"], dict)
    assert serialized_meta["gate_force_listen"] in {True, False}
    assert result.metrics["inner_os/active_relation_total_people"] >= 0.0
    assert result.metrics["inner_os/commitment_accepted_cost"] >= 0.0
    assert result.metrics["inner_os/initiative_followup_bias"] >= 0.0


def test_process_turn_applies_streaming_emit_wait_for_guarded_channel(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness.config = SimpleNamespace(latency=SimpleNamespace(enable_loose=True))
    harness._surface_mode = lambda: "streaming"  # type: ignore[assignment]
    harness._inner_os_emit_clock_ms = lambda: 5000.0  # type: ignore[assignment]
    harness._inner_os_headless_runtime.step = lambda actuation_plan=None: HeadlessTurnResult(  # type: ignore[assignment]
        execution_mode="stabilize_boundary",
        primary_action="hold_presence",
        response_channel="hold",
        response_channel_score=0.91,
        reply_permission="hold_or_brief",
        wait_before_action="held",
        repair_window_commitment="soft",
        outcome_goal="leave_room",
        boundary_mode="soft_guard",
        attention_target="user",
        memory_write_priority="carry",
        nonverbal_response_state={"timing_bias": "wait"},
        presence_hold_state={"hold_state": "holding_space"},
        turn_timing_hint={
            "response_channel": "hold",
            "wait_before_action": "held",
            "timing_bias": "wait",
            "entry_window": "held",
            "pause_profile": "soft_pause",
            "overlap_policy": "wait_for_release",
            "interruptibility": "low",
            "minimum_wait_ms": 40,
            "interrupt_guard_ms": 90,
        },
        do_not_cross=["force_reopen"],
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(runtime_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = harness.process_turn(
        user_text="do you remember this new place",
        context="previous framing",
    )

    assert sleep_calls == [0.028]
    assert result.response is not None
    assert result.response.controls_used["inner_os_emit_timing"]["response_channel"] == "hold"
    assert result.response.controls_used["inner_os_emit_timing"]["wait_applied"] is True
    assert result.response.controls_used["inner_os_emit_timing"]["wait_applied_ms"] == 28.0
    assert result.metrics["inner_os/emit_wait_applied"] == 1.0
    assert result.metrics["inner_os/emit_wait_applied_ms"] == 28.0
    assert result.response.latency_ms == 40.0
    assert result.response.controls_used["inner_os_emit_timing"]["emit_not_before_ms"] == 5028.0
    assert result.persona_meta["inner_os"]["actuation_emit_timing"]["wait_applied"] is True


def test_process_turn_restores_initiative_followup_bias_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_followup.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_initiative_followup_bias"] = 0.41
    harness._last_gate_context["inner_os_initiative_followup_state"] = "offer_next_step"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )
    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["initiative_followup_state"] == "offer_next_step"
    assert overnight["initiative_followup_bias"] == 0.41
    assert harness._last_gate_context["inner_os_initiative_followup_state"] in {"hold", "reopen_softly", "offer_next_step"}
    assert harness._last_gate_context["inner_os_initiative_followup_bias"] >= 0.0


def test_process_turn_restores_commitment_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_commitment.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_commitment_target_focus"] = "repair"
    harness._last_gate_context["inner_os_commitment_state_focus"] = "commit"
    harness._last_gate_context["inner_os_commitment_carry_bias"] = 0.37
    harness._last_gate_context["inner_os_commitment_followup_focus"] = "reopen_softly"
    harness._last_gate_context["inner_os_commitment_mode_focus"] = "repair"
    harness._last_gate_context["inner_os_commitment_carry_reason"] = "commit:repair"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["commitment_target_focus"] == "repair"
    assert overnight["commitment_mode_focus"] == "repair"
    assert overnight["commitment_carry_bias"] == 0.37
    assert result.metrics["inner_os/overnight_commitment_bias_active"] == 1.0
    assert harness._last_gate_context["inner_os_commitment_target_focus"] in {"hold", "stabilize", "repair", "bond_protect", "step_forward"}


def test_process_turn_restores_agenda_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_agenda.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_agenda_focus"] = "repair"
    harness._last_gate_context["inner_os_agenda_bias"] = 0.22
    harness._last_gate_context["inner_os_agenda_reason"] = "repair_window"
    harness._last_gate_context["inner_os_agenda_window_focus"] = "next_private_window"
    harness._last_gate_context["inner_os_agenda_window_bias"] = 0.18
    harness._last_gate_context["inner_os_agenda_window_reason"] = "wait_for_private_window"
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["agenda_focus"] == "repair"
    assert overnight["agenda_bias"] == 0.22
    assert overnight["agenda_reason"] == "repair_window"
    assert overnight["agenda_window_focus"] == "next_private_window"
    assert overnight["agenda_window_bias"] == 0.18
    assert overnight["agenda_window_reason"] == "wait_for_private_window"
    assert result.metrics["inner_os/overnight_agenda_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_agenda_window_bias_active"] == 1.0
    assert result.metrics["inner_os/agenda_winner_margin"] >= 0.0
    assert result.persona_meta["inner_os"]["agenda_state"]["winner_margin"] >= 0.0


def test_process_turn_restores_learning_and_social_experiment_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_learning_mode.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_learning_mode_focus"] = "repair_probe"
    harness._last_gate_context["inner_os_learning_mode_carry_bias"] = 0.16
    harness._last_gate_context["inner_os_social_experiment_focus"] = "repair_signal_probe"
    harness._last_gate_context["inner_os_social_experiment_carry_bias"] = 0.14
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["learning_mode_focus"] == "repair_probe"
    assert overnight["learning_mode_carry_bias"] == 0.16
    assert overnight["social_experiment_focus"] == "repair_signal_probe"
    assert overnight["social_experiment_carry_bias"] == 0.14
    assert result.metrics["inner_os/overnight_learning_mode_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_social_experiment_bias_active"] == 1.0
    assert result.metrics["inner_os/learning_mode_winner_margin"] >= 0.0
    assert result.metrics["inner_os/social_experiment_winner_margin"] >= 0.0


def test_process_turn_restores_body_and_relational_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_body_rel.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_body_homeostasis_focus"] = "recovering"
    harness._last_gate_context["inner_os_body_homeostasis_carry_bias"] = 0.16
    harness._last_gate_context["inner_os_homeostasis_budget_focus"] = "recovering"
    harness._last_gate_context["inner_os_homeostasis_budget_bias"] = 0.09
    harness._last_gate_context["inner_os_relational_continuity_focus"] = "reopening"
    harness._last_gate_context["inner_os_relational_continuity_carry_bias"] = 0.14
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["body_homeostasis_focus"] == "recovering"
    assert overnight["body_homeostasis_carry_bias"] == 0.16
    assert overnight["homeostasis_budget_focus"] == "recovering"
    assert overnight["homeostasis_budget_bias"] == 0.09
    assert overnight["relational_continuity_focus"] == "reopening"
    assert overnight["relational_continuity_carry_bias"] == 0.14
    assert result.metrics["inner_os/overnight_body_homeostasis_bias_active"] == 1.0
    assert result.metrics["inner_os/overnight_homeostasis_budget_active"] == 1.0
    assert result.metrics["inner_os/overnight_relational_continuity_bias_active"] == 1.0
    assert result.persona_meta["inner_os"]["homeostasis_budget_state"]["winner_margin"] >= 0.0
    assert result.metrics["inner_os/homeostasis_budget_winner_margin"] >= 0.0


def test_process_turn_restores_expressive_style_history_and_banter_carry_into_overnight_view(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_style_history.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_expressive_style_history_focus"] = "warm_companion"
    harness._last_gate_context["inner_os_expressive_style_history_bias"] = 0.08
    harness._last_gate_context["inner_os_banter_style_focus"] = "gentle_tease"
    harness._last_gate_context["inner_os_lexical_variation_carry_bias"] = 0.11
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    overnight = result.persona_meta["inner_os"]["reaction_vs_overnight_bias"]["overnight"]
    assert overnight["expressive_style_history_focus"] == "warm_companion"
    assert overnight["expressive_style_history_bias"] == 0.08
    assert overnight["banter_style_focus"] == "gentle_tease"
    assert overnight["lexical_variation_carry_bias"] == 0.11
    assert result.metrics["inner_os/overnight_expressive_style_history_active"] == 1.0
    assert result.metrics["inner_os/overnight_lexical_variation_carry_active"] == 1.0


def test_process_turn_restores_temperament_trace_into_persona_meta(tmp_path) -> None:
    memory_core = MemoryCore(tmp_path / "runtime_temperament.jsonl")
    hooks = IntegrationHooks(memory_core=memory_core)
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_temperament_trace"] = {
        "temperament_forward_trace": 0.44,
        "temperament_guard_trace": 0.18,
        "temperament_bond_trace": 0.22,
        "temperament_recovery_trace": 0.12,
    }
    result = harness.process_turn(
        user_text="stay with what is visible first",
        context="visual scene",
        intent="check_in",
    )

    assert result.persona_meta["inner_os"]["temperament_trace"]["temperament_forward_trace"] >= 0.0
    assert result.metrics["inner_os/temperament_forward_trace"] >= 0.0
    assert harness._last_gate_context["inner_os_temperament_trace"]["temperament_guard_trace"] >= 0.0


def test_process_turn_emits_distillation_record_with_model_source(tmp_path: Path) -> None:
    @dataclass
    class _DistillHarness(_ProcessTurnHarness):
        def step(
            self,
            *,
            user_text: Optional[str] = None,
            context: Optional[str] = None,
            intent: Optional[str] = None,
            fast_only: bool = False,
            image_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            self._last_step_context = context
            response = SimpleNamespace(
                text="I will stay with what is visible first.",
                model="qwen-3.5-instruct",
                model_source="live_list",
                trace_id="hub-123",
                latency_ms=12.0,
                safety={"rating": "G"},
                controls_used={"mode": "watch"},
                retrieval_summary={"hits": [{"id": "ctx-1"}]},
                perception_summary={"text": "soft evening slope and signboard"},
                confidence=0.74,
                uncertainty_reason=(),
            )
            return {
                "talk_mode": "watch",
                "response_route": "watch",
                "metrics": {},
                "persona_meta": {},
                "heart": {},
                "shadow": None,
                "qualia_gate": {},
                "affect": None,
                "response": response,
                "memory_reference": None,
            }

    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _DistillHarness(_integration_hooks=hooks)
    harness._distillation_log_path = tmp_path / "inner_os_distillation.jsonl"
    harness._distillation_log_include_text = False

    _ = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    rows = [
        json.loads(line)
        for line in harness._distillation_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    record = rows[0]
    assert record["model"]["name"] == "qwen-3.5-instruct"
    assert record["model"]["source"] == "live_list"
    assert record["text_payload"] == {}
    assert record["input_fingerprint"]["user_text_sha256"]
    assert record["decision_snapshot"]["commitment_state"]["state"] in {"waver", "settle", "commit"}


def test_process_turn_exposes_transfer_package_and_runtime_seed(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    package = result.persona_meta["inner_os"]["transfer_package"]
    assert package["schema"] == "inner_os_transfer_package/v1"
    assert package["portable_state"]["same_turn"]["commitment_state"]["state"] in {"waver", "settle", "commit"}
    assert "runtime_seed" in package

    restored_seed = harness.build_inner_os_state_seed_from_transfer_package(package)
    assert "commitment_state_focus" in restored_seed
    assert "body_homeostasis_focus" in restored_seed
    assert "homeostasis_budget_focus" in restored_seed
    assert "relational_continuity_focus" in restored_seed
    assert "person_registry_snapshot" in restored_seed
    assert "temperament_forward_trace" in restored_seed
    assert result.metrics["inner_os/transfer_package_ready"] == 1.0


def test_process_turn_writes_transfer_package_snapshot_when_path_is_set(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._transfer_package_path = tmp_path / "inner_os_transfer_package.json"

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    payload = json.loads(harness._transfer_package_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inner_os_transfer_package/v1"
    assert payload["portable_state"]["same_turn"]["initiative_readiness"]["state"] in {"hold", "tentative", "ready"}
    assert result.metrics["inner_os/transfer_package_written"] == 1.0


def test_process_turn_writes_dashboard_snapshot_when_path_is_set(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._dashboard_snapshot_path = tmp_path / "inner_os_dashboard_snapshot.json"

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    payload = json.loads(harness._dashboard_snapshot_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inner_os_dashboard_snapshot/v1"
    assert payload["same_turn"]["social_topology_state"] in {
        "ambient",
        "one_to_one",
        "threaded_group",
        "public_visible",
        "hierarchical",
        "",
    }
    assert "temporal_membrane_mode" in payload["same_turn"]
    assert "temporal_membrane_focus" in payload["overnight"]
    assert "temporal_membrane_focus" in payload["transfer"]
    assert "temporal_alignment" in payload
    assert "focus_alignment" in payload["temporal_alignment"]
    assert "same_to_overnight_reentry_delta" in payload["temporal_alignment"]
    assert "contact_alignment" in payload
    assert "style" in payload["contact_alignment"]
    assert "dominant_carry_channel" in payload
    assert result.metrics["inner_os/dashboard_snapshot_ready"] == 1.0
    assert result.metrics["inner_os/dashboard_snapshot_written"] == 1.0


def test_apply_transfer_package_restores_working_memory_seed_and_gate_context() -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    package = {
        "runtime_seed": {
            "prev_qualia": [0.3],
            "commitment_state_focus": "commit",
            "temporal_membrane_focus": "coherent_reentry",
            "temporal_timeline_bias": 0.24,
            "temporal_reentry_bias": 0.19,
            "temporal_supersession_bias": 0.05,
            "temporal_continuity_bias": 0.17,
            "temporal_relation_reentry_bias": 0.14,
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.38,
            "body_homeostasis_focus": "recovering",
            "body_homeostasis_carry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.09,
            "relational_continuity_focus": "reopening",
            "relational_continuity_carry_bias": 0.14,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
            "temperament_forward_trace": 0.29,
        },
        "portable_state": {
            "carry": {
                "monument_carry": {
                    "memory_anchor": "harbor slope",
                    "semantic_seed_focus": "harbor slope",
                    "semantic_seed_anchor": "harbor slope",
                    "semantic_seed_strength": 0.73,
                    "semantic_seed_recurrence": 1.18,
                    "long_term_theme_focus": "quiet harbor routine",
                    "long_term_theme_anchor": "harbor slope",
                    "long_term_theme_strength": 0.61,
                    "long_term_theme_kind": "place",
                    "long_term_theme_summary": "quiet harbor routine",
                    "relation_seed_summary": "shared harbor thread",
                    "relation_seed_strength": 0.55,
                },
                "relationship_summary": {
                    "related_person_id": "person:harbor",
                    "attachment": 0.61,
                    "familiarity": 0.57,
                    "trust_memory": 0.49,
                    "partner_address_hint": "gentle",
                    "partner_timing_hint": "slow",
                    "partner_stance_hint": "careful",
                    "partner_social_interpretation": "future_open",
                },
            }
        },
    }

    restored = harness.apply_inner_os_transfer_package(package)

    assert restored["commitment_state_focus"] == "commit"
    assert harness._last_gate_context["inner_os_commitment_state_focus"] == "commit"
    assert harness._last_gate_context["inner_os_temporal_membrane_focus"] == "coherent_reentry"
    assert harness._last_gate_context["inner_os_autobiographical_thread_mode"] == "unfinished_thread"
    assert harness._last_gate_context["inner_os_autobiographical_thread_anchor"] == "harbor promise"
    assert harness._last_gate_context["inner_os_autobiographical_thread_strength"] == 0.38
    assert harness._last_gate_context["inner_os_temporal_timeline_bias"] == 0.24
    assert harness._last_gate_context["inner_os_temporal_reentry_bias"] == 0.19
    assert harness._last_gate_context["inner_os_body_homeostasis_focus"] == "recovering"
    assert harness._last_gate_context["inner_os_homeostasis_budget_focus"] == "recovering"
    assert harness._last_gate_context["inner_os_relational_continuity_focus"] == "reopening"
    assert harness._last_gate_context["inner_os_person_registry_snapshot"]["total_people"] == 2
    assert harness._last_gate_context["inner_os_group_thread_registry_snapshot"]["total_threads"] == 1
    assert harness._last_gate_context["inner_os_discussion_thread_registry_snapshot"]["dominant_issue_state"] == "pausing_issue"
    assert harness._last_gate_context["inner_os_group_thread_focus"] == "threaded_group"
    seed = harness._surface_world_state["working_memory_seed"]
    assert seed["semantic_seed_focus"] == "harbor slope"
    assert seed["long_term_theme_summary"] == "quiet harbor routine"
    assert seed["relation_seed_summary"] == "shared harbor thread"
    assert seed["group_thread_id"] == "threaded_group:person:friend|person:harbor"


def test_load_inner_os_transfer_package_returns_warm_start_summary() -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)

    summary = harness.load_inner_os_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
                "semantic_seed_strength": 0.64,
            },
            "initiative_followup_state": "reopen_softly",
            "commitment_target_focus": "repair",
            "temporal_membrane_focus": "same_group_reentry",
            "temporal_reentry_bias": 0.18,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.07,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
        }
    )

    assert summary["schema"] == "inner_os_transfer_package/v1"
    assert summary["migration"]["applied"] is True
    assert summary["source_model"]["name"] == "qwen-legacy"
    assert summary["initiative_followup_state"] == "reopen_softly"
    assert summary["commitment_target_focus"] == "repair"
    assert summary["temporal_membrane_focus"] == "same_group_reentry"
    assert summary["temporal_reentry_bias"] == 0.18
    assert summary["homeostasis_budget_focus"] == "recovering"
    assert summary["person_registry_total_people"] == 2
    assert summary["group_thread_total_threads"] == 1
    assert summary["discussion_thread_total_threads"] == 1
    assert summary["discussion_thread_dominant_anchor"] == "repair anchor"
    assert summary["discussion_thread_dominant_issue_state"] == "pausing_issue"
    assert summary["group_thread_focus"] == "threaded_group"
    assert summary["semantic_seed_anchor"] == "harbor slope"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_focus"] == "harbor slope"


def test_load_transfer_package_from_disk_restores_warm_start_seed(tmp_path: Path) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._transfer_package_path = tmp_path / "inner_os_transfer_package.json"
    harness._transfer_package_path.write_text(
        json.dumps(
            {
                "source_model": "qwen-legacy",
                "working_memory_seed": {
                    "memory_anchor": "harbor slope",
                    "semantic_seed_focus": "harbor slope",
                    "semantic_seed_anchor": "harbor slope",
                    "semantic_seed_strength": 0.68,
                    "related_person_id": "person:harbor",
                },
                "initiative_followup_state": "offer_next_step",
                "temperament_forward_trace": 0.32,
                "daily_carry_summary": {
                    "overnight_focus": {
                        "commitment_target_focus": "repair",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = harness._load_inner_os_transfer_package_from_disk()

    assert loaded is True
    assert harness._last_gate_context["inner_os_initiative_followup_state"] == "offer_next_step"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_anchor"] == "harbor slope"
    assert harness._last_gate_context["inner_os_transfer_package_migration"]["applied"] is True
    persisted = json.loads(harness._transfer_package_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "inner_os_transfer_package/v1"
    assert persisted["portable_state"]["carry"]["relationship_summary"]["related_person_id"] == "person:harbor"


def test_load_inner_os_transfer_package_from_path_returns_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    package_path = tmp_path / "transfer_package.json"
    package_path.write_text(
        json.dumps(
            {
                "schema": "inner_os_transfer_package/v1",
                "package_version": "v1",
                "source_model": {"name": "qwen-3.5-instruct", "source": "live_list"},
                "portable_state": {
                    "carry": {
                        "monument_carry": {
                            "semantic_seed_focus": "harbor slope",
                            "semantic_seed_anchor": "harbor slope",
                        }
                    }
                },
                "runtime_seed": {
                    "initiative_followup_state": "offer_next_step",
                    "commitment_target_focus": "step_forward",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = harness.load_inner_os_transfer_package(package_path)

    assert summary["migration"]["applied"] is False
    assert summary["source_model"]["name"] == "qwen-3.5-instruct"
    assert summary["initiative_followup_state"] == "offer_next_step"
    assert summary["commitment_target_focus"] == "step_forward"
    assert summary["semantic_seed_focus"] == "harbor slope"


def test_build_inner_os_model_swap_bundle_prefers_target_model_in_cache(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )
    bundle = harness.build_inner_os_model_swap_bundle(
        target_model="qwen-3.5-coder",
        result=result,
    )

    assert bundle["schema"] == "inner_os_model_swap_bundle/v1"
    assert bundle["target_model"] == "qwen-3.5-coder"
    assert bundle["selection_mode"] == "cache_preferred"
    assert bundle["transfer_package"]["schema"] == "inner_os_transfer_package/v1"
    assert terrain_llm.get_cached_selected_model("http://localhost:1234/v1") == "qwen-3.5-coder"


def test_warm_start_from_transfer_package_applies_target_model_preference(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks()
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    summary = harness.warm_start_from_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
            },
            "initiative_followup_state": "offer_next_step",
            "agenda_focus": "step_forward",
            "agenda_bias": 0.14,
        },
        target_model="qwen-3.5-coder",
    )

    assert summary["target_model"] == "qwen-3.5-coder"
    assert summary["selected_model_after_warm_start"] == "qwen-3.5-coder"
    assert harness._last_gate_context["inner_os_transfer_target_model"] == "qwen-3.5-coder"
    assert harness._surface_world_state["working_memory_seed"]["semantic_seed_anchor"] == "harbor slope"


def test_process_turn_exposes_transfer_summary_after_warm_start(tmp_path: Path, monkeypatch) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(terrain_llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(terrain_llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(terrain_llm, "CUSTOM_BASE", None)

    harness.warm_start_from_transfer_package(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
            },
            "initiative_followup_state": "offer_next_step",
            "commitment_target_focus": "repair",
            "agenda_focus": "repair",
            "agenda_bias": 0.18,
            "temporal_membrane_focus": "same_group_reentry",
            "temporal_timeline_bias": 0.22,
            "temporal_reentry_bias": 0.2,
            "temporal_supersession_bias": 0.04,
            "temporal_continuity_bias": 0.18,
            "temporal_relation_reentry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.08,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
        },
        target_model="qwen-3.5-coder",
    )

    result = harness.process_turn(
        user_text="Can you stay with what is visible first?",
        context="known context",
        intent="check_in",
    )

    summary = result.persona_meta["inner_os"]["transfer_summary"]
    assert summary["migration"]["applied"] is True
    assert summary["semantic_seed_anchor"] == "harbor slope"
    assert summary["commitment_target_focus"] == "repair"
    assert summary["agenda_focus"] == "repair"
    assert summary["agenda_bias"] == 0.18
    assert summary["temporal_membrane_focus"] == "same_group_reentry"
    assert summary["temporal_reentry_bias"] == 0.2
    assert summary["homeostasis_budget_focus"] == "recovering"
    assert summary["person_registry_total_people"] == 2
    assert summary["group_thread_total_threads"] == 1
    assert summary["discussion_thread_total_threads"] == 1
    assert summary["discussion_thread_dominant_anchor"] == "repair anchor"
    assert summary["discussion_thread_dominant_issue_state"] == "pausing_issue"
    assert summary["group_thread_focus"] == "threaded_group"
    assert summary["target_model"] == "qwen-3.5-coder"
    assert result.metrics["inner_os/transfer_migration_active"] == 1.0
    assert result.metrics["inner_os/transfer_migration_from_legacy"] == 1.0
    assert result.metrics["inner_os/transfer_semantic_seed_visible"] == 1.0
    assert result.metrics["inner_os/transfer_target_model_requested"] == 1.0
    assert result.metrics["inner_os/person_registry_total_people"] == 2.0
    assert result.metrics["inner_os/group_thread_total_threads"] == 1.0
    assert result.metrics["inner_os/discussion_thread_total_threads"] == 1.0
    assert result.persona_meta["inner_os"]["group_thread_registry_summary"]["total_threads"] == 1
    assert result.persona_meta["inner_os"]["discussion_thread_registry_summary"]["total_threads"] >= 1
    assert result.persona_meta["inner_os"]["group_thread_focus"] == "threaded_group"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["group_thread_focus"] == "threaded_group"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["temporal_membrane_focus"] == "same_group_reentry"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["temporal_reentry_bias"] == 0.2


def test_process_turn_passes_working_memory_seed_into_inner_os_pre_turn(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._surface_world_state["working_memory_seed"] = {
        "semantic_seed_focus": "harbor slope",
        "semantic_seed_anchor": "harbor slope",
        "semantic_seed_strength": 0.78,
        "semantic_seed_recurrence": 1.22,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0


def test_process_turn_derives_working_memory_seed_from_eqnet_system(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness.eqnet_system = SimpleNamespace(
        _working_memory_signature_summary=lambda: {
            "dominant_focus": "harbor slope",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.56,
            "recurrence_weight": 1.24,
            "long_term_theme": {
                "focus": "harbor slope",
                "anchor": "harbor slope",
                "kind": "place",
                "strength": 0.61,
                "summary": "quiet harbor slope memory",
            },
        },
        _latest_nightly_working_memory_replay_summary=lambda: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.62,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/semantic_seed_recurrence"] > 0.0
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0
    assert result.persona_meta["inner_os"]["semantic_seed_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.61
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_kind"] == "place"


def test_build_context_payload_includes_working_memory_seed_tags() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    harness._talk_mode = SimpleNamespace(value="watch")
    harness._last_gate_context["inner_os_working_memory_seed"] = {
        "semantic_seed_focus": "harbor slope",
        "semantic_seed_anchor": "harbor slope",
        "long_term_theme_focus": "harbor slope",
        "long_term_theme_anchor": "harbor slope",
        "long_term_theme_kind": "place",
        "long_term_theme_summary": "quiet harbor slope memory",
    }
    payload = harness._build_context_payload(
        user_text="remember this place",
        context_text="previous framing",
        metrics={},
        gate_ctx=SimpleNamespace(force_listen=False, text_input=True),
        route=SimpleNamespace(value="conscious"),
    )
    assert "wm_seed_focus:harbor_slope" in payload["context_tags"]
    assert "wm_seed_anchor:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_focus:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_anchor:harbor_slope" in payload["context_tags"]
    assert "ltm_theme_kind:place" in payload["context_tags"]
    assert "ltm_theme_summary:quiet_harbor_slope_memory" in payload["context_tags"]


def test_process_turn_uses_conscious_mosaic_working_memory_seed_for_recall(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.42,
        },
    )
    result = harness.process_turn(user_text="harbor", context="previous framing")
    assert result.metrics["inner_os/conscious_seed_strength"] == 0.42


def test_process_turn_uses_conscious_mosaic_long_term_theme_as_residue_only(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {},
        latest_long_term_theme=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "kind": "place",
            "summary": "quiet harbor slope memory",
            "strength": 0.46,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.0
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] in {"", None}
    assert result.persona_meta["inner_os"]["current_focus"] == "place"
    assert result.persona_meta["inner_os"]["working_memory_pressure"] > 0.0


def test_process_turn_corroborates_conscious_long_term_theme_with_nightly_theme(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {},
        latest_long_term_theme=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "kind": "place",
            "summary": "quiet harbor slope memory",
            "strength": 0.46,
        },
    )
    harness._latest_nightly_long_term_theme_summary = lambda: {
        "focus": "harbor slope",
        "anchor": "harbor slope",
        "kind": "place",
        "summary": "quiet harbor slope memory",
        "strength": 0.6,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.6
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_summary"] == "quiet harbor slope memory"


def test_process_turn_derives_pre_turn_seed_from_conscious_mosaic(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._conscious_memory = SimpleNamespace(
        tail=lambda n: [{"id": "ep-1"}],
        latest_working_memory_seed=lambda n: {
            "focus": "harbor slope",
            "anchor": "harbor slope",
            "strength": 0.44,
        },
    )
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] == 0.44
    assert result.persona_meta["inner_os"]["semantic_seed_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["semantic_seed_anchor"] == "harbor slope"
    assert result.persona_meta["inner_os"]["current_focus"] in {"meaning", "place"}


def test_process_turn_derives_seed_from_nightly_long_term_theme(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_long_term_theme_summary = lambda: {
        "focus": "harbor slope",
        "anchor": "harbor slope",
        "kind": "place",
        "summary": "quiet harbor slope memory",
        "strength": 0.6,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/semantic_seed_strength"] > 0.0
    assert result.metrics["inner_os/long_term_theme_strength"] == 0.6
    assert result.persona_meta["inner_os"]["long_term_theme_focus"] == "harbor slope"
    assert result.persona_meta["inner_os"]["long_term_theme_summary"] == "quiet harbor slope memory"


def test_process_turn_derives_identity_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_identity_arc_summary = lambda: {
        "arc_kind": "repairing_bond",
        "phase": "shifting",
        "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
        "open_tension": "timing_sensitive_reentry",
        "stability": 0.58,
        "memory_anchor": "harbor slope",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.metrics["inner_os/identity_arc_visible"] == 1.0
    assert result.metrics["inner_os/identity_arc_stability"] == 0.58
    assert result.persona_meta["inner_os"]["identity_arc_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["identity_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["identity_arc_summary"].startswith("repair is gathering")
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["identity_arc_kind"] == "repairing_bond"


def test_process_turn_exposes_identity_arc_registry_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_identity_arc_registry_summary = lambda: {
        "dominant_arc_id": "repairing_bond::person:user::meaning::harbor",
        "dominant_arc_kind": "repairing_bond",
        "dominant_arc_phase": "shifting",
        "dominant_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
        "active_arc_count": 1,
        "total_arcs": 1,
        "top_arc_ids": ["repairing_bond::person:user::meaning::harbor"],
        "status_counts": {"active": 1},
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["identity_arc_registry_summary"]["dominant_arc_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["identity_arc_registry_dominant_kind"] == "repairing_bond"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["identity_arc_registry_active_count"] == 1


def test_process_turn_derives_relation_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_relation_arc_summary = lambda: {
        "arc_kind": "repairing_relation",
        "phase": "shifting",
        "summary": "repair is gathering around a companion thread",
        "open_tension": "timing_sensitive_reentry",
        "stability": 0.56,
        "related_person_id": "person:harbor",
        "group_thread_id": "threaded_group:person:friend|person:harbor",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["relation_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["relation_arc_summary"] == "repair is gathering around a companion thread"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["relation_arc_summary"] == "repair is gathering around a companion thread"


def test_process_turn_exposes_relation_arc_registry_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_relation_arc_registry_summary = lambda: {
        "dominant_arc_id": "repairing_relation::person:harbor::companion",
        "dominant_arc_kind": "repairing_relation",
        "dominant_arc_phase": "shifting",
        "dominant_arc_summary": "repair is gathering around a companion thread",
        "dominant_person_id": "person:harbor",
        "dominant_group_thread_id": "threaded_group:person:friend|person:harbor",
        "active_arc_count": 1,
        "total_arcs": 1,
        "top_arc_ids": ["repairing_relation::person:harbor::companion"],
        "status_counts": {"active": 1},
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["relation_arc_registry_summary"]["dominant_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["relation_arc_registry_dominant_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["relation_arc_registry_active_count"] == 1


def test_process_turn_derives_group_relation_arc_from_nightly_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_group_relation_arc_summary = lambda: {
        "arc_kind": "repairing_relation",
        "phase": "shifting",
        "summary": "repair is moving through a shared group thread in small steps",
        "group_thread_id": "threaded_group:person:friend|person:harbor",
        "topology_focus": "threaded_group",
        "boundary_mode": "same_group_reentry",
        "reentry_window_focus": "next_same_group_window",
        "dominant_person_id": "person:harbor",
        "stability": 0.61,
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["group_relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["group_relation_arc_phase"] == "shifting"
    assert result.persona_meta["inner_os"]["group_relation_arc_boundary_mode"] == "same_group_reentry"
    assert result.persona_meta["inner_os"]["group_relation_arc_group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert result.persona_meta["inner_os"]["group_relation_arc"]["summary"] == "repair is moving through a shared group thread in small steps"
    assert result.persona_meta["inner_os"]["continuity_summary"]["same_turn"]["group_relation_arc_kind"] == "repairing_relation"
    assert result.persona_meta["inner_os"]["continuity_summary"]["overnight"]["group_relation_boundary_mode"] == "same_group_reentry"


def test_process_turn_derives_relation_seed_from_nightly_partner_summary(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._latest_nightly_partner_relation_summary = lambda: {
        "person_id": "user",
        "summary": "gentle harbor companion thread",
        "memory_anchor": "harbor slope",
        "social_role": "companion",
        "attachment": 0.74,
        "familiarity": 0.69,
        "trust_memory": 0.71,
        "strength": 0.72,
        "address_hint": "companion",
        "timing_hint": "open",
        "stance_hint": "familiar",
        "social_interpretation": "familiar:companion:open",
    }
    result = harness.process_turn(user_text="", context="previous framing")
    assert result.persona_meta["inner_os"]["related_person_id"] == "user"
    assert result.persona_meta["inner_os"]["attachment"] >= 0.6
    assert result.persona_meta["inner_os"]["familiarity"] >= 0.58
    assert result.persona_meta["inner_os"]["trust_memory"] >= 0.6
    assert result.persona_meta["inner_os"]["partner_address_hint"] == "companion"
    assert result.persona_meta["inner_os"]["partner_timing_hint"] == "open"
    assert result.persona_meta["inner_os"]["partner_stance_hint"] == "familiar"


def test_visual_reflection_uses_degraded_surface_text(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _DegradedHarness(_integration_hooks=hooks)
    normalized = harness._normalize_perception_summary({"error": "vision_error", "detail": "500 Server Error"})
    response = SimpleNamespace(text="I am keeping the reading tentative.", perception_summary=None)
    response = harness._attach_visual_reflection(response, normalized)
    assert response is not None
    assert normalized is not None
    assert normalized["status"] == "degraded"
    assert "visual read did not settle" in response.text
