# -*- coding: utf-8 -*-
"""LM Studio と EQNet の接続状態を可視化する probe。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from eqnet_core.models.runtime_turn import RuntimeTurnResult
from inner_os.expression.content_policy import (
    derive_content_sequence,
    localize_content_sequence,
)
from inner_os.expression.interaction_constraints import derive_interaction_constraints
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


@dataclass(frozen=True)
class LMStudioPipelineProbe:
    """LM Studio の生出力と EQNet 最終出力を並べて見るための要約。"""

    llm_model: str = ""
    llm_model_source: str = ""
    llm_used: bool = False
    llm_bridge_called: bool = False
    force_llm_bridge: bool = False
    llm_raw_text: str = ""
    llm_raw_model: str = ""
    llm_raw_model_source: str = ""
    llm_raw_differs_from_final: bool = False
    talk_mode: str = ""
    route: str = ""
    response_text: str = ""
    response_latency_ms: float = 0.0
    response_strategy: str = ""
    action_mode: str = ""
    actuation_primary_action: str = ""
    actuation_execution_mode: str = ""
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
    content_sequence: list[dict[str, str]] = field(default_factory=list)
    surface_profile: dict[str, Any] = field(default_factory=dict)
    planned_content_sequence_present: bool = False
    allow_guarded_narrative_bridge: bool = False
    guarded_narrative_bridge_used: bool = False
    continuity_same_turn: dict[str, Any] = field(default_factory=dict)
    continuity_overnight: dict[str, Any] = field(default_factory=dict)
    temporal_alignment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_model": self.llm_model,
            "llm_model_source": self.llm_model_source,
            "llm_used": self.llm_used,
            "llm_bridge_called": self.llm_bridge_called,
            "force_llm_bridge": self.force_llm_bridge,
            "llm_raw_text": self.llm_raw_text,
            "llm_raw_model": self.llm_raw_model,
            "llm_raw_model_source": self.llm_raw_model_source,
            "llm_raw_differs_from_final": self.llm_raw_differs_from_final,
            "talk_mode": self.talk_mode,
            "route": self.route,
            "response_text": self.response_text,
            "response_latency_ms": round(float(self.response_latency_ms), 4),
            "response_strategy": self.response_strategy,
            "action_mode": self.action_mode,
            "actuation_primary_action": self.actuation_primary_action,
            "actuation_execution_mode": self.actuation_execution_mode,
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
            "content_sequence": [dict(item) for item in self.content_sequence],
            "surface_profile": dict(self.surface_profile),
            "planned_content_sequence_present": self.planned_content_sequence_present,
            "allow_guarded_narrative_bridge": self.allow_guarded_narrative_bridge,
            "guarded_narrative_bridge_used": self.guarded_narrative_bridge_used,
            "continuity_same_turn": dict(self.continuity_same_turn),
            "continuity_overnight": dict(self.continuity_overnight),
            "temporal_alignment": dict(self.temporal_alignment),
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
    response_controls = _mapping(getattr(response, "controls", None))
    response_controls_used = _mapping(getattr(response, "controls_used", None))
    merged_response_controls = dict(response_controls)
    merged_response_controls.update(response_controls_used)
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
        llm_raw_text=_text(inner_os.get("llm_raw_text")),
        llm_raw_model=_text(inner_os.get("llm_raw_model")),
        llm_raw_model_source=_text(inner_os.get("llm_raw_model_source")),
        llm_raw_differs_from_final=bool(
            inner_os.get("llm_raw_differs_from_final", False)
        ),
        talk_mode=_text(inner_os.get("talk_mode") or getattr(result, "talk_mode", "")),
        route=_text(inner_os.get("route") or getattr(result, "response_route", "")),
        response_text=_text(getattr(response, "text", "")),
        response_latency_ms=_float(getattr(response, "latency_ms", 0.0)),
        response_strategy=_text(interaction_policy.get("response_strategy")),
        action_mode=_text(
            _nested(interaction_policy, "action_posture", "mode")
            or inner_os.get("actuation_execution_mode")
        ),
        actuation_primary_action=_text(inner_os.get("actuation_primary_action")),
        actuation_execution_mode=_text(inner_os.get("actuation_execution_mode")),
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
    lines.append(
        f"- route: {probe.route or '(none)'} / talk_mode: {probe.talk_mode or '(none)'}"
    )
    lines.append(
        f"- strategy: {probe.response_strategy or '(none)'}"
        f" / actuation: {probe.actuation_primary_action or '(none)'}"
        f" / execution: {probe.actuation_execution_mode or '(none)'}"
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
