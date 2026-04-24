# -*- coding: utf-8 -*-
"""LM Studio と EQNet の接続状態を可視化する probe。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Optional, Sequence

from eqnet_core.models.runtime_turn import RuntimeTurnResult
from inner_os.expression.content_policy import (
    derive_content_sequence,
    localize_content_sequence,
)
from inner_os.expression.discourse_shape import derive_discourse_shape
from inner_os.expression.interaction_constraints import derive_interaction_constraints
from inner_os.expression.llm_bridge_contract import review_llm_bridge_text
from inner_os.expression.reaction_contract import derive_reaction_contract
from inner_os.expression.repetition_guard import derive_repetition_guard
from inner_os.expression.turn_delta import derive_turn_delta


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(source: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = source or {}
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _compact_mapping(
    source: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> dict[str, Any]:
    payload = _mapping(source)
    return {key: payload.get(key) for key in keys if key in payload}


def _align_sequence_to_response_text(
    content_sequence: Sequence[Mapping[str, Any]],
    *,
    discourse_shape: Mapping[str, Any] | None,
    response_text: str,
) -> list[dict[str, str]]:
    aligned = [
        {
            "act": _text(item.get("act")),
            "text": _text(item.get("text")),
        }
        for item in content_sequence
        if isinstance(item, Mapping) and _text(item.get("text"))
    ]
    if str((discourse_shape or {}).get("shape_id") or "").strip() != "bright_bounce":
        return aligned
    if not response_text:
        return aligned
    rendered = " ".join(_text(item.get("text")) for item in aligned).strip()
    if not rendered or rendered == response_text.strip():
        return aligned
    sentence_parts = [
        part.strip()
        for part in re.findall(r"[^。！？!?]+[。！？!?]?", response_text)
        if str(part or "").strip()
    ]
    if len(sentence_parts) < len(aligned):
        return aligned
    return [
        {
            **dict(item),
            "text": sentence_parts[index],
        }
        for index, item in enumerate(aligned)
    ]


@dataclass(frozen=True)
class LMStudioPipelineProbe:
    """LM Studio の生出力と EQNet 最終出力を並べて見るための要約。"""

    llm_model: str = ""
    llm_model_source: str = ""
    llm_used: bool = False
    llm_bridge_called: bool = False
    force_llm_bridge: bool = False
    llm_raw_text: str = ""
    llm_raw_original_text: str = ""
    llm_raw_model: str = ""
    llm_raw_model_source: str = ""
    llm_raw_differs_from_final: bool = False
    llm_raw_contract_ok: bool = True
    llm_raw_contract_violations: list[str] = field(default_factory=list)
    talk_mode: str = ""
    route: str = ""
    gate_force_listen: bool = False
    timing_guard: dict[str, Any] = field(default_factory=dict)
    response_text: str = ""
    response_latency_ms: float = 0.0
    response_strategy: str = ""
    action_mode: str = ""
    actuation_primary_action: str = ""
    actuation_execution_mode: str = ""
    actuation_response_channel: str = ""
    actuation_wait_before_action: str = ""
    actuation_turn_timing_hint: dict[str, Any] = field(default_factory=dict)
    actuation_emit_timing: dict[str, Any] = field(default_factory=dict)
    reaction_contract: dict[str, Any] = field(default_factory=dict)
    commitment_target: str = ""
    agenda_window_state: str = ""
    social_topology_state: str = ""
    temporal_membrane_mode: str = ""
    temporal_membrane_focus: str = ""
    qualia_gate_allow: bool = True
    qualia_gate_suppress_narrative: bool = False
    qualia_gate_reason: str = ""
    qualia_gate_details: dict[str, Any] = field(default_factory=dict)
    interaction_constraints: dict[str, Any] = field(default_factory=dict)
    repetition_guard: dict[str, Any] = field(default_factory=dict)
    turn_delta: dict[str, Any] = field(default_factory=dict)
    discourse_shape: dict[str, Any] = field(default_factory=dict)
    content_sequence: list[dict[str, str]] = field(default_factory=list)
    surface_profile: dict[str, Any] = field(default_factory=dict)
    planned_content_sequence_present: bool = False
    allow_guarded_narrative_bridge: bool = False
    guarded_narrative_bridge_used: bool = False
    continuity_same_turn: dict[str, Any] = field(default_factory=dict)
    continuity_overnight: dict[str, Any] = field(default_factory=dict)
    temporal_alignment: dict[str, Any] = field(default_factory=dict)
    recent_dialogue_state: dict[str, Any] = field(default_factory=dict)
    live_engagement_state: dict[str, Any] = field(default_factory=dict)
    lightness_budget_state: dict[str, Any] = field(default_factory=dict)
    expressive_style_state: dict[str, Any] = field(default_factory=dict)
    green_field: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_model": self.llm_model,
            "llm_model_source": self.llm_model_source,
            "llm_used": self.llm_used,
            "llm_bridge_called": self.llm_bridge_called,
            "force_llm_bridge": self.force_llm_bridge,
            "llm_raw_text": self.llm_raw_text,
            "llm_raw_original_text": self.llm_raw_original_text,
            "llm_raw_model": self.llm_raw_model,
            "llm_raw_model_source": self.llm_raw_model_source,
            "llm_raw_differs_from_final": self.llm_raw_differs_from_final,
            "llm_raw_contract_ok": self.llm_raw_contract_ok,
            "llm_raw_contract_violations": list(self.llm_raw_contract_violations),
            "talk_mode": self.talk_mode,
            "route": self.route,
            "gate_force_listen": self.gate_force_listen,
            "timing_guard": dict(self.timing_guard),
            "response_text": self.response_text,
            "response_latency_ms": round(float(self.response_latency_ms), 4),
            "response_strategy": self.response_strategy,
            "action_mode": self.action_mode,
            "actuation_primary_action": self.actuation_primary_action,
            "actuation_execution_mode": self.actuation_execution_mode,
            "actuation_response_channel": self.actuation_response_channel,
            "actuation_wait_before_action": self.actuation_wait_before_action,
            "actuation_turn_timing_hint": dict(self.actuation_turn_timing_hint),
            "actuation_emit_timing": dict(self.actuation_emit_timing),
            "reaction_contract": dict(self.reaction_contract),
            "commitment_target": self.commitment_target,
            "agenda_window_state": self.agenda_window_state,
            "social_topology_state": self.social_topology_state,
            "temporal_membrane_mode": self.temporal_membrane_mode,
            "temporal_membrane_focus": self.temporal_membrane_focus,
            "qualia_gate_allow": self.qualia_gate_allow,
            "qualia_gate_suppress_narrative": self.qualia_gate_suppress_narrative,
            "qualia_gate_reason": self.qualia_gate_reason,
            "qualia_gate_details": dict(self.qualia_gate_details),
            "interaction_constraints": dict(self.interaction_constraints),
            "repetition_guard": dict(self.repetition_guard),
            "turn_delta": dict(self.turn_delta),
            "discourse_shape": dict(self.discourse_shape),
            "content_sequence": [dict(item) for item in self.content_sequence],
            "surface_profile": dict(self.surface_profile),
            "planned_content_sequence_present": self.planned_content_sequence_present,
            "allow_guarded_narrative_bridge": self.allow_guarded_narrative_bridge,
            "guarded_narrative_bridge_used": self.guarded_narrative_bridge_used,
            "continuity_same_turn": dict(self.continuity_same_turn),
            "continuity_overnight": dict(self.continuity_overnight),
            "temporal_alignment": dict(self.temporal_alignment),
            "recent_dialogue_state": dict(self.recent_dialogue_state),
            "live_engagement_state": dict(self.live_engagement_state),
            "lightness_budget_state": dict(self.lightness_budget_state),
            "expressive_style_state": dict(self.expressive_style_state),
            "green_field": dict(self.green_field),
        }


def build_lmstudio_pipeline_probe(
    result: RuntimeTurnResult,
    *,
    current_text: str,
    history: Optional[Sequence[str]] = None,
) -> LMStudioPipelineProbe:
    """Runtime 結果から LM Studio 検証用の可視化データを組む。"""

    persona_meta = _mapping(getattr(result, "persona_meta", None))
    inner_os = _mapping(persona_meta.get("inner_os"))
    response = getattr(result, "response", None)
    response_text = _text(getattr(response, "text", ""))
    response_controls = _mapping(getattr(response, "controls", None))
    response_controls_used = _mapping(getattr(response, "controls_used", None))
    merged_response_controls = dict(response_controls)
    merged_response_controls.update(response_controls_used)
    effective_llm_raw_text = _text(
        merged_response_controls.get("inner_os_llm_raw_text")
        or inner_os.get("llm_raw_text")
    )
    effective_llm_raw_original_text = _text(
        merged_response_controls.get("inner_os_llm_raw_original_text")
        or inner_os.get("llm_raw_original_text")
    )
    effective_llm_raw_contract_ok = bool(
        merged_response_controls.get("inner_os_llm_raw_contract_ok")
        if "inner_os_llm_raw_contract_ok" in merged_response_controls
        else inner_os.get("llm_raw_contract_ok", True)
    )
    effective_llm_raw_contract_violations = [
        _text(item)
        for item in (
            merged_response_controls.get("inner_os_llm_raw_contract_violations")
            or inner_os.get("llm_raw_contract_violations")
            or []
        )
        if _text(item)
    ]
    interaction_policy = _mapping(inner_os.get("interaction_policy_packet"))
    continuity_summary = _mapping(inner_os.get("continuity_summary"))
    same_turn = _mapping(continuity_summary.get("same_turn"))
    overnight = _mapping(continuity_summary.get("overnight"))
    qualia_gate = _mapping(getattr(result, "qualia_gate", None))
    qualia_gate_details = _compact_mapping(
        qualia_gate,
        (
            "reason",
            "u_t",
            "m_t",
            "load_t",
            "p_t",
            "p_ema",
            "theta",
            "boundary_score",
            "boundary_term",
            "presence_override",
        ),
    )
    temporal_alignment = _mapping(
        inner_os.get("dashboard_snapshot", {})
    ).get("temporal_alignment")
    if not isinstance(temporal_alignment, Mapping):
        temporal_alignment = _mapping(
            inner_os.get("dashboard_snapshot_temporal_alignment")
        )
    if not temporal_alignment:
        temporal_alignment = _mapping(
            _nested(inner_os, "dashboard_snapshot", "temporal_alignment")
        )

    interaction_constraints = derive_interaction_constraints(interaction_policy).to_dict()
    repetition_guard = derive_repetition_guard(history).to_dict()
    turn_delta = derive_turn_delta(
        interaction_policy,
        interaction_constraints=interaction_constraints,
    ).to_dict()
    planned_content_sequence = merged_response_controls.get(
        "inner_os_planned_content_sequence"
    )
    if isinstance(planned_content_sequence, list) and planned_content_sequence:
        content_sequence = localize_content_sequence(
            [
                dict(item)
                for item in planned_content_sequence
                if isinstance(item, Mapping) and _text(item.get("text"))
            ],
            locale="ja-JP",
        )
    else:
        content_sequence = derive_content_sequence(
            current_text=current_text,
            interaction_policy=interaction_policy,
            conscious_access={"intent": _text(interaction_policy.get("dialogue_act"))},
            history=history,
            interaction_constraints=interaction_constraints,
            repetition_guard=repetition_guard,
            turn_delta=turn_delta,
            locale="ja-JP",
        )
    discourse_shape = _compact_mapping(
        merged_response_controls.get("inner_os_discourse_shape"),
        (
            "shape_id",
            "primary_move",
            "secondary_move",
            "sentence_budget",
            "question_budget",
            "anchor_mode",
            "closing_mode",
            "energy",
            "brightness",
            "playfulness",
            "tempo",
        ),
    )
    if not discourse_shape:
        discourse_shape = derive_discourse_shape(
            content_sequence=content_sequence,
            turn_delta=turn_delta,
        ).to_dict()
    reaction_contract = _mapping(
        merged_response_controls.get("inner_os_reaction_contract")
        or inner_os.get("reaction_contract")
    )
    if not reaction_contract:
        reaction_contract = dict(
            derive_reaction_contract(
                interaction_policy=interaction_policy,
                action_posture=_mapping(_nested(interaction_policy, "action_posture")),
                actuation_plan={
                    "execution_mode": _text(inner_os.get("actuation_execution_mode")),
                    "primary_action": _text(inner_os.get("actuation_primary_action")),
                    "response_channel": _text(inner_os.get("actuation_response_channel")),
                    "wait_before_action": _text(inner_os.get("actuation_wait_before_action")),
                    "nonverbal_response_state": {
                        "timing_bias": _text(
                            _nested(inner_os, "nonverbal_response_state", "timing_bias")
                        )
                    },
                },
                discourse_shape=discourse_shape,
                turn_delta=turn_delta,
            )
        )
    content_sequence = _align_sequence_to_response_text(
        content_sequence,
        discourse_shape=discourse_shape,
        response_text=response_text,
    )
    if (
        effective_llm_raw_original_text
        and not effective_llm_raw_contract_ok
        and effective_llm_raw_text == effective_llm_raw_original_text
    ):
        question_budget = int(discourse_shape.get("question_budget") or 0)
        probe_surface_context_packet = {
            "conversation_phase": _text(turn_delta.get("kind"))
            or _text(_nested(interaction_policy, "recent_dialogue_state", "state")),
            "constraints": {"max_questions": question_budget},
            "source_state": {
                "turn_delta_kind": _text(turn_delta.get("kind")),
                "discourse_shape_id": _text(discourse_shape.get("shape_id")),
                "utterance_reason_question_policy": "none"
                if question_budget <= 0
                else "",
            },
            "surface_profile": {
                "discourse_shape_id": _text(discourse_shape.get("shape_id")),
            },
        }
        probe_review = review_llm_bridge_text(
            raw_text=effective_llm_raw_original_text,
            surface_context_packet=probe_surface_context_packet,
            reaction_contract=reaction_contract or None,
            fallback_text=response_text,
        )
        effective_llm_raw_text = probe_review.sanitized_text
        effective_llm_raw_contract_ok = bool(probe_review.ok)
        effective_llm_raw_contract_violations = probe_review.violation_codes()

    return LMStudioPipelineProbe(
        llm_model=_text(inner_os.get("llm_model")),
        llm_model_source=_text(inner_os.get("llm_model_source")),
        llm_used=bool(inner_os.get("llm_bridge_called", False))
        or bool(
            _text(inner_os.get("llm_model"))
            or _float(getattr(response, "latency_ms", 0.0)) > 0.0
        ),
        llm_bridge_called=bool(inner_os.get("llm_bridge_called", False)),
        force_llm_bridge=bool(inner_os.get("force_llm_bridge", False)),
        llm_raw_text=effective_llm_raw_text,
        llm_raw_original_text=effective_llm_raw_original_text,
        llm_raw_model=_text(inner_os.get("llm_raw_model")),
        llm_raw_model_source=_text(inner_os.get("llm_raw_model_source")),
        llm_raw_differs_from_final=bool(
            effective_llm_raw_text and effective_llm_raw_text != response_text
        ),
        llm_raw_contract_ok=effective_llm_raw_contract_ok,
        llm_raw_contract_violations=effective_llm_raw_contract_violations,
        talk_mode=_text(inner_os.get("talk_mode") or getattr(result, "talk_mode", "")),
        route=_text(inner_os.get("route") or getattr(result, "response_route", "")),
        gate_force_listen=bool(inner_os.get("gate_force_listen", False)),
        timing_guard=_compact_mapping(
            _mapping(inner_os.get("timing_guard")),
            (
                "active",
                "reason",
                "response_channel",
                "overlap_policy",
                "emit_not_before_ms",
                "interrupt_guard_until_ms",
                "voice_conflict",
            ),
        ),
        response_text=response_text,
        response_latency_ms=_float(getattr(response, "latency_ms", 0.0)),
        response_strategy=_text(interaction_policy.get("response_strategy")),
        action_mode=_text(
            _nested(interaction_policy, "action_posture", "mode")
            or inner_os.get("actuation_execution_mode")
        ),
        actuation_primary_action=_text(inner_os.get("actuation_primary_action")),
        actuation_execution_mode=_text(inner_os.get("actuation_execution_mode")),
        actuation_response_channel=_text(inner_os.get("actuation_response_channel")),
        actuation_wait_before_action=_text(inner_os.get("actuation_wait_before_action")),
        actuation_turn_timing_hint=_compact_mapping(
            _mapping(inner_os.get("actuation_turn_timing_hint")),
            (
                "entry_window",
                "pause_profile",
                "overlap_policy",
                "interruptibility",
                "minimum_wait_ms",
                "interrupt_guard_ms",
            ),
        ),
        actuation_emit_timing=_compact_mapping(
            _mapping(
                merged_response_controls.get("inner_os_emit_timing")
                or inner_os.get("actuation_emit_timing")
            ),
            (
                "response_channel",
                "entry_window",
                "pause_profile",
                "overlap_policy",
                "interruptibility",
                "minimum_wait_ms",
                "interrupt_guard_ms",
                "effective_emit_delay_ms",
                "effective_latency_ms",
                "emit_not_before_ms",
                "interrupt_guard_until_ms",
                "wait_applied",
                "wait_applied_ms",
            ),
        ),
        reaction_contract=_compact_mapping(
            reaction_contract,
            (
                "stance",
                "scale",
                "initiative",
                "question_budget",
                "interpretation_budget",
                "response_channel",
                "timing_mode",
                "continuity_mode",
                "distance_mode",
                "closure_mode",
                "reason_tags",
                "shape_id",
                "strategy",
                "execution_mode",
                "wait_before_action",
            ),
        ),
        commitment_target=_text(
            _nested(inner_os, "commitment_state", "target")
            or same_turn.get("commitment_target")
        ),
        agenda_window_state=_text(
            _nested(inner_os, "agenda_window_state", "state")
            or same_turn.get("agenda_window_state")
        ),
        social_topology_state=_text(same_turn.get("social_topology_state")),
        temporal_membrane_mode=_text(same_turn.get("temporal_membrane_mode")),
        temporal_membrane_focus=_text(overnight.get("temporal_membrane_focus")),
        qualia_gate_allow=bool(qualia_gate.get("allow", True)),
        qualia_gate_suppress_narrative=bool(
            qualia_gate.get("suppress_narrative", False)
        ),
        qualia_gate_reason=_text(
            inner_os.get("qualia_gate_reason") or qualia_gate.get("reason")
        ),
        qualia_gate_details=qualia_gate_details,
        interaction_constraints=interaction_constraints,
        repetition_guard=repetition_guard,
        turn_delta=turn_delta,
        discourse_shape=discourse_shape,
        content_sequence=[dict(item) for item in content_sequence],
        surface_profile=_compact_mapping(
            merged_response_controls.get("inner_os_surface_profile"),
            (
                "opening_delay",
                "response_length",
                "sentence_temperature",
                "pause_insertion",
                "certainty_style",
                "opening_pace_windowed",
                "return_gaze_expectation",
                "voice_texture",
                "content_sequence_length",
            ),
        ),
        planned_content_sequence_present=bool(
            isinstance(planned_content_sequence, list) and planned_content_sequence
        ),
        allow_guarded_narrative_bridge=bool(
            merged_response_controls.get("inner_os_allow_guarded_narrative_bridge", False)
        ),
        guarded_narrative_bridge_used=bool(
            merged_response_controls.get("inner_os_guarded_narrative_bridge_used", False)
        ),
        continuity_same_turn=_compact_mapping(
            same_turn,
            (
                "protection_mode",
                "agenda_state",
                "agenda_window_state",
                "commitment_target",
                "social_topology_state",
                "temporal_membrane_mode",
                "temporal_reentry_pull",
                "identity_arc_kind",
                "relation_arc_kind",
                "group_relation_arc_kind",
            ),
        ),
        continuity_overnight=_compact_mapping(
            overnight,
            (
                "dominant_carry_channel",
                "agenda_focus",
                "group_thread_focus",
                "temporal_membrane_focus",
                "temporal_reentry_bias",
                "temporal_continuity_bias",
                "identity_arc_registry_dominant_kind",
                "relation_arc_registry_dominant_kind",
            ),
        ),
        temporal_alignment=_mapping(temporal_alignment),
        recent_dialogue_state=_compact_mapping(
            _mapping(interaction_policy.get("recent_dialogue_state")),
            ("state", "thread_carry", "reopen_pressure", "recent_anchor"),
        ),
        live_engagement_state=_compact_mapping(
            _mapping(interaction_policy.get("live_engagement_state")),
            ("state", "score", "winner_margin", "primary_move"),
        ),
        lightness_budget_state=_compact_mapping(
            _mapping(interaction_policy.get("lightness_budget_state")),
            ("state", "score", "winner_margin", "banter_room", "playful_ceiling", "suppression"),
        ),
        expressive_style_state=_compact_mapping(
            _mapping(interaction_policy.get("expressive_style_state")),
            ("state", "score", "lightness_room"),
        ),
        green_field=_compact_mapping(
            _mapping(_mapping(interaction_policy.get("green_kernel_composition")).get("field")),
            ("affective_charge", "guardedness", "reopening_pull"),
        ),
    )


def render_lmstudio_pipeline_probe(probe: LMStudioPipelineProbe) -> str:
    """probe を人間が読みやすい確認テキストへ整形する。"""

    lines: list[str] = []
    lines.append("## LM Studio / EQNet パイプライン")
    lines.append(
        f"- model: {probe.llm_model or '(unknown)'}"
        f" / source: {probe.llm_model_source or '(unknown)'}"
    )
    lines.append(f"- llm_used: {probe.llm_used}")
    lines.append(f"- llm_bridge_called: {probe.llm_bridge_called}")
    lines.append(f"- force_llm_bridge: {probe.force_llm_bridge}")
    lines.append(f"- llm_raw_differs_from_final: {probe.llm_raw_differs_from_final}")
    lines.append(f"- llm_raw_contract_ok: {probe.llm_raw_contract_ok}")
    if probe.llm_raw_contract_violations:
        lines.append(
            "- llm_raw_contract_violations: "
            + ", ".join(probe.llm_raw_contract_violations)
        )
    lines.append(
        f"- route: {probe.route or '(none)'} / talk_mode: {probe.talk_mode or '(none)'}"
    )
    lines.append(
        f"- gate_force_listen: {'true' if probe.gate_force_listen else 'false'}"
    )
    if probe.timing_guard:
        lines.append(
            "- timing_guard: "
            + ", ".join(
                f"{key}={value}"
                for key, value in probe.timing_guard.items()
            )
        )
    lines.append(
        f"- strategy: {probe.response_strategy or '(none)'}"
        f" / actuation: {probe.actuation_primary_action or '(none)'}"
        f" / execution: {probe.actuation_execution_mode or '(none)'}"
        f" / channel: {probe.actuation_response_channel or '(none)'}"
    )
    lines.append(
        f"- wait_before_action: {probe.actuation_wait_before_action or '(none)'}"
    )
    if probe.actuation_turn_timing_hint:
        lines.append(
            "- turn_timing_hint: "
            + ", ".join(
                f"{key}={value}"
                for key, value in probe.actuation_turn_timing_hint.items()
            )
        )
    if probe.actuation_emit_timing:
        lines.append(
            "- emit_timing: "
            + ", ".join(
                f"{key}={value}"
                for key, value in probe.actuation_emit_timing.items()
            )
        )
    if probe.reaction_contract:
        lines.append(
            "- reaction_contract: "
            + ", ".join(
                f"{key}={value}"
                for key, value in probe.reaction_contract.items()
            )
        )
    lines.append(
        f"- commitment: {probe.commitment_target or '(none)'}"
        f" / agenda_window: {probe.agenda_window_state or '(none)'}"
        f" / topology: {probe.social_topology_state or '(none)'}"
    )
    lines.append(
        f"- temporal: same-turn={probe.temporal_membrane_mode or '(none)'}"
        f" / overnight={probe.temporal_membrane_focus or '(none)'}"
    )
    lines.append(
        f"- qualia_gate: allow={probe.qualia_gate_allow}"
        f" / suppress_narrative={probe.qualia_gate_suppress_narrative}"
    )
    lines.append(f"- qualia_gate_reason: {probe.qualia_gate_reason or '(none)'}")
    lines.append("")
    lines.append("## LM Raw Output")
    lines.append(
        f"- raw_model: {probe.llm_raw_model or '(unknown)'}"
        f" / raw_source: {probe.llm_raw_model_source or '(unknown)'}"
    )
    if probe.llm_raw_original_text and probe.llm_raw_original_text != probe.llm_raw_text:
        lines.append("### Raw Original")
        lines.append(probe.llm_raw_original_text)
        lines.append("")
        lines.append("### Raw Effective")
    lines.append(probe.llm_raw_text or "(no raw llm output)")
    lines.append("")
    lines.append("## 最終応答")
    lines.append(probe.response_text or "(no response)")
    lines.append(f"- latency_ms: {round(float(probe.response_latency_ms), 2)}")
    lines.append("")
    lines.append("## Qualia Gate")
    if probe.qualia_gate_details:
        for key, value in probe.qualia_gate_details.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (empty)")
    lines.append("")
    lines.append("## Surface")
    for key, value in probe.surface_profile.items():
        lines.append(f"- {key}: {value}")
    lines.append(f"- planned_content_sequence_present: {probe.planned_content_sequence_present}")
    lines.append(f"- allow_guarded_narrative_bridge: {probe.allow_guarded_narrative_bridge}")
    lines.append(f"- guarded_narrative_bridge_used: {probe.guarded_narrative_bridge_used}")
    lines.append("")
    lines.append("## Interaction Constraints")
    for key, value in probe.interaction_constraints.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Turn Delta")
    for key, value in probe.turn_delta.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Discourse Shape")
    if probe.discourse_shape:
        for key, value in probe.discourse_shape.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (empty)")
    lines.append("")
    lines.append("## Content Sequence")
    if probe.content_sequence:
        for idx, item in enumerate(probe.content_sequence, start=1):
            lines.append(f"{idx}. [{_text(item.get('act'))}] {_text(item.get('text'))}")
    else:
        lines.append("- (empty)")
    lines.append("")
    lines.append("## Temporal Alignment")
    if probe.temporal_alignment:
        for key, value in probe.temporal_alignment.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (empty)")
    lines.append("")
    lines.append("## Continuity")
    lines.append(f"- same_turn: {probe.continuity_same_turn}")
    lines.append(f"- overnight: {probe.continuity_overnight}")
    lines.append("")
    lines.append("## Repetition Guard")
    lines.append(f"- {probe.repetition_guard}")
    return "\n".join(lines)
