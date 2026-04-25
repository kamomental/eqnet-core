from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from inner_os.orchestration.stimulus_history_influence import StimulusHistoryInfluence


@dataclass(frozen=True)
class ProtectiveTracePalaceState:
    protective_trace_density: float = 0.0
    current_crisis_binding: float = 0.0
    reentry_sensitivity: float = 0.0
    trigger_pressure: float = 0.0
    hyperarousal_pressure: float = 0.0
    rem_replay_pressure: float = 0.0
    sensory_flash_risk: float = 0.0
    dream_intrusion_pressure: float = 0.0
    somatic_reactivation: float = 0.0
    stabilization_need: float = 0.0
    safe_reconsolidation_readiness: float = 0.0
    recovery_path_strength: float = 0.0
    trusted_reentry_window: float = 0.0
    dominant_mode: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protective_trace_density": self.protective_trace_density,
            "current_crisis_binding": self.current_crisis_binding,
            "reentry_sensitivity": self.reentry_sensitivity,
            "trigger_pressure": self.trigger_pressure,
            "hyperarousal_pressure": self.hyperarousal_pressure,
            "rem_replay_pressure": self.rem_replay_pressure,
            "sensory_flash_risk": self.sensory_flash_risk,
            "dream_intrusion_pressure": self.dream_intrusion_pressure,
            "somatic_reactivation": self.somatic_reactivation,
            "stabilization_need": self.stabilization_need,
            "safe_reconsolidation_readiness": self.safe_reconsolidation_readiness,
            "recovery_path_strength": self.recovery_path_strength,
            "trusted_reentry_window": self.trusted_reentry_window,
            "dominant_mode": self.dominant_mode,
            "reasons": list(self.reasons),
        }


def derive_protective_trace_palace_state(
    expression_context_state: Mapping[str, Any] | None,
    *,
    stimulus_history_influence: StimulusHistoryInfluence | Mapping[str, Any] | None = None,
) -> ProtectiveTracePalaceState:
    context = _mapping(expression_context_state)
    memory = _group(context, "memory")
    safety = _group(context, "safety")
    emergency = _group(context, "emergency")
    body = _group(context, "body")
    environment = _group(context, "environment")
    homeostasis = _group(context, "homeostasis")
    sleep = _group(context, "sleep")
    protective = _first_mapping(
        context.get("protective_trace_palace"),
        context.get("protective_trace"),
        context.get("primary_impact_trace"),
    )
    stimulus = _mapping(
        stimulus_history_influence.to_dict()
        if isinstance(stimulus_history_influence, StimulusHistoryInfluence)
        else stimulus_history_influence
    )
    field_clarity = _float01(stimulus.get("field_clarity"))
    stimulus_pressure = _float01(stimulus.get("stimulus_pressure"))
    novelty_pressure = _float01(stimulus.get("novelty_pressure"))

    write_class = _text(
        memory.get("write_class")
        or memory.get("memory_write_class")
        or context.get("memory_write_class")
    )
    protective_write_pressure = _write_class_pressure(write_class)
    explicit_density = _max_float(
        protective.get("protective_trace_density"),
        protective.get("trace_density"),
        protective.get("primary_impact_density"),
    )
    memory_reentry = _max_float(
        memory.get("ignition_readiness"),
        memory.get("replay_priority"),
        memory.get("reconsolidation_priority"),
        memory.get("memory_tension"),
        stimulus.get("memory_reentry_pressure"),
    )
    safety_pressure = _max_float(
        safety.get("risk_pressure"),
        safety.get("boundary_pressure"),
        emergency.get("urgency"),
        emergency.get("risk_pressure"),
        0.72 if _text(safety.get("dialogue_permission")) == "boundary_only" else 0.0,
    )
    body_pressure = _max_float(
        body.get("stress"),
        body.get("startle"),
        body.get("somatic_reactivation"),
        body.get("recovery_need"),
        homeostasis.get("load"),
        homeostasis.get("recovery_need"),
    )
    trigger_pressure = _max_float(
        protective.get("trigger_pressure"),
        protective.get("trigger_match"),
        memory.get("trigger_pressure"),
        memory.get("trigger_match"),
        environment.get("trigger_salience"),
        environment.get("threat_cue_pressure"),
    )
    hyperarousal_pressure = _clamp01(
        _max_float(
            protective.get("hyperarousal_pressure"),
            body.get("hyperarousal"),
            body.get("startle"),
            body.get("stress"),
            homeostasis.get("arousal_load"),
        )
    )
    current_crisis_binding = _clamp01(
        _max_float(
            protective.get("current_crisis_binding"),
            protective.get("present_threat_binding"),
            memory.get("present_threat_binding"),
            safety.get("current_threat"),
            emergency.get("current_crisis"),
        )
        * 0.46
        + trigger_pressure * 0.22
        + hyperarousal_pressure * 0.2
        + (1.0 - field_clarity) * 0.12
    )
    rem_replay_pressure = _clamp01(
        _max_float(
            protective.get("rem_replay_pressure"),
            sleep.get("rem_replay_pressure"),
            sleep.get("rem_activation"),
            sleep.get("nightmare_pressure"),
            sleep.get("replay_pressure"),
        )
    )
    sensory_flash_risk = _clamp01(
        _max_float(
            protective.get("sensory_flash_risk"),
            protective.get("flashback_risk"),
            sleep.get("nightmare_pressure"),
        )
        * 0.58
        + memory_reentry * 0.24
        + body_pressure * 0.12
        + current_crisis_binding * 0.06
    )
    dream_intrusion_pressure = _clamp01(
        _max_float(
            protective.get("dream_intrusion_pressure"),
            protective.get("dream_pressure"),
            sleep.get("dream_intrusion_pressure"),
            sleep.get("nightmare_pressure"),
            sleep.get("replay_pressure"),
        )
        * 0.74
        + rem_replay_pressure * 0.26
    )
    protective_trace_density = _clamp01(
        max(explicit_density, protective_write_pressure) * 0.44
        + memory_reentry * 0.24
        + sensory_flash_risk * 0.14
        + safety_pressure * 0.1
        + current_crisis_binding * 0.08
    )
    reentry_sensitivity = _clamp01(
        protective_trace_density * 0.32
        + memory_reentry * 0.2
        + stimulus_pressure * 0.14
        + novelty_pressure * 0.08
        + trigger_pressure * 0.1
        + current_crisis_binding * 0.1
        + (1.0 - field_clarity) * 0.06
    )
    somatic_reactivation = _clamp01(
        _max_float(protective.get("somatic_reactivation"), body_pressure)
    )
    stabilization_need = _clamp01(
        reentry_sensitivity * 0.3
        + sensory_flash_risk * 0.18
        + dream_intrusion_pressure * 0.12
        + somatic_reactivation * 0.16
        + current_crisis_binding * 0.14
        + hyperarousal_pressure * 0.12
        + (1.0 - field_clarity) * 0.08
    )
    recovery_path_strength = _clamp01(
        _max_float(
            protective.get("recovery_path_strength"),
            protective.get("recovery_support"),
            memory.get("repair_trace"),
            memory.get("safe_repeat"),
            homeostasis.get("recovery_capacity"),
        )
    )
    trusted_reentry_window = _clamp01(
        _max_float(
            protective.get("trusted_reentry_window"),
            safety.get("trusted_reentry_window"),
            1.0 - safety_pressure,
        )
        * 0.44
        + field_clarity * 0.24
        + recovery_path_strength * 0.2
        + (1.0 - somatic_reactivation) * 0.12
    )
    safe_reconsolidation_readiness = _clamp01(
        recovery_path_strength * 0.34
        + trusted_reentry_window * 0.3
        + field_clarity * 0.18
        + (1.0 - stabilization_need) * 0.18
    )

    reasons: list[str] = []
    if write_class:
        reasons.append(f"memory.write_class.{write_class}")
    if protective_trace_density >= 0.45:
        reasons.append("protective_trace.density")
    if current_crisis_binding >= 0.45:
        reasons.append("protective_trace.current_crisis_binding")
    if reentry_sensitivity >= 0.5:
        reasons.append("protective_trace.reentry_sensitive")
    if trigger_pressure >= 0.45:
        reasons.append("protective_trace.trigger_pressure")
    if hyperarousal_pressure >= 0.45:
        reasons.append("body.hyperarousal")
    if rem_replay_pressure >= 0.45:
        reasons.append("sleep.rem_replay")
    if sensory_flash_risk >= 0.45:
        reasons.append("protective_trace.sensory_flash")
    if dream_intrusion_pressure >= 0.45:
        reasons.append("protective_trace.dream_intrusion")
    if somatic_reactivation >= 0.45:
        reasons.append("body.somatic_reactivation")
    if recovery_path_strength >= 0.45:
        reasons.append("recovery.path_available")

    dominant_mode = "ambient"
    if current_crisis_binding >= 0.55 and hyperarousal_pressure >= 0.45:
        dominant_mode = "protective_hold"
    elif stabilization_need >= 0.58 and reentry_sensitivity >= 0.5:
        dominant_mode = "protective_hold"
    elif rem_replay_pressure >= 0.55 or dream_intrusion_pressure >= 0.55 or sensory_flash_risk >= 0.62:
        dominant_mode = "restabilize"
    elif safe_reconsolidation_readiness >= 0.58 and stabilization_need <= 0.48:
        dominant_mode = "safe_reconsolidation"
    elif recovery_path_strength >= 0.52 and trusted_reentry_window >= 0.48:
        dominant_mode = "recovery_opening"

    return ProtectiveTracePalaceState(
        protective_trace_density=round(protective_trace_density, 4),
        current_crisis_binding=round(current_crisis_binding, 4),
        reentry_sensitivity=round(reentry_sensitivity, 4),
        trigger_pressure=round(trigger_pressure, 4),
        hyperarousal_pressure=round(hyperarousal_pressure, 4),
        rem_replay_pressure=round(rem_replay_pressure, 4),
        sensory_flash_risk=round(sensory_flash_risk, 4),
        dream_intrusion_pressure=round(dream_intrusion_pressure, 4),
        somatic_reactivation=round(somatic_reactivation, 4),
        stabilization_need=round(stabilization_need, 4),
        safe_reconsolidation_readiness=round(safe_reconsolidation_readiness, 4),
        recovery_path_strength=round(recovery_path_strength, 4),
        trusted_reentry_window=round(trusted_reentry_window, 4),
        dominant_mode=dominant_mode,
        reasons=tuple(_dedupe(reasons)),
    )


def apply_protective_trace_palace_to_contract_inputs(
    *,
    contract_inputs: Mapping[str, Any],
    palace_state: ProtectiveTracePalaceState,
) -> dict[str, Any]:
    adjusted = _clone_mapping_tree(contract_inputs)
    packet = _ensure_dict(adjusted, "surface_context_packet")
    source_state = _ensure_dict(packet, "source_state")
    source_state.update(
        {
            "protective_trace_density": palace_state.protective_trace_density,
            "protective_trace_current_crisis_binding": palace_state.current_crisis_binding,
            "protective_trace_reentry_sensitivity": palace_state.reentry_sensitivity,
            "protective_trace_trigger_pressure": palace_state.trigger_pressure,
            "protective_trace_hyperarousal_pressure": palace_state.hyperarousal_pressure,
            "protective_trace_rem_replay_pressure": palace_state.rem_replay_pressure,
            "protective_trace_sensory_flash_risk": palace_state.sensory_flash_risk,
            "protective_trace_dream_intrusion_pressure": palace_state.dream_intrusion_pressure,
            "protective_trace_somatic_reactivation": palace_state.somatic_reactivation,
            "protective_trace_stabilization_need": palace_state.stabilization_need,
            "protective_trace_safe_reconsolidation_readiness": palace_state.safe_reconsolidation_readiness,
            "protective_trace_recovery_path_strength": palace_state.recovery_path_strength,
            "protective_trace_trusted_reentry_window": palace_state.trusted_reentry_window,
            "protective_trace_dominant_mode": palace_state.dominant_mode,
        }
    )

    if palace_state.dominant_mode in {"protective_hold", "restabilize"}:
        _project_protective_hold(adjusted, palace_state)
    elif palace_state.dominant_mode in {"safe_reconsolidation", "recovery_opening"}:
        _project_recovery_opening(adjusted, palace_state)
    return adjusted


def _project_protective_hold(
    contract_inputs: dict[str, Any],
    palace_state: ProtectiveTracePalaceState,
) -> None:
    interaction_policy = _ensure_dict(contract_inputs, "interaction_policy")
    interaction_policy["response_strategy"] = "respectful_wait"
    interaction_policy["recent_dialogue_state"] = {"state": "reopening_thread"}

    action_posture = _ensure_dict(contract_inputs, "action_posture")
    action_posture["boundary_mode"] = "contain"
    action_posture["question_budget"] = 0

    actuation_plan = _ensure_dict(contract_inputs, "actuation_plan")
    actuation_plan["execution_mode"] = "defer_with_presence"
    actuation_plan["response_channel"] = "hold"
    actuation_plan["wait_before_action"] = "brief"

    discourse_shape = _ensure_dict(contract_inputs, "discourse_shape")
    discourse_shape["shape_id"] = "reflect_hold"
    discourse_shape["question_budget"] = 0

    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    packet["conversation_phase"] = "reopening_thread"
    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0
    constraints["prefer_return_point"] = True

    source_state = _ensure_dict(packet, "source_state")
    source_state["utterance_reason_offer"] = ""
    source_state["utterance_reason_preserve"] = "stabilize_before_reentry"
    source_state["organism_protective_tension"] = max(
        _float01(source_state.get("organism_protective_tension")),
        palace_state.stabilization_need,
        palace_state.reentry_sensitivity,
    )

    turn_delta = _ensure_dict(contract_inputs, "turn_delta")
    turn_delta["kind"] = "reopening_thread"
    turn_delta["preferred_act"] = "stabilize_before_reentry"


def _project_recovery_opening(
    contract_inputs: dict[str, Any],
    palace_state: ProtectiveTracePalaceState,
) -> None:
    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0
    constraints["prefer_return_point"] = False

    source_state = _ensure_dict(packet, "source_state")
    source_state["utterance_reason_offer"] = ""
    source_state["utterance_reason_preserve"] = "safe_reconsolidation_window"
    source_state["protective_trace_recovery_opening"] = max(
        palace_state.safe_reconsolidation_readiness,
        palace_state.recovery_path_strength,
    )

    action_posture = _ensure_dict(contract_inputs, "action_posture")
    action_posture["boundary_mode"] = "soft_hold"
    action_posture["question_budget"] = 0

    actuation_plan = _ensure_dict(contract_inputs, "actuation_plan")
    if _text(actuation_plan.get("response_channel")) != "hold":
        actuation_plan["response_channel"] = "speak"
        actuation_plan["execution_mode"] = "user_led_support"

    discourse_shape = _ensure_dict(contract_inputs, "discourse_shape")
    if _text(discourse_shape.get("shape_id")) != "reflect_hold":
        discourse_shape["shape_id"] = "reflect_step"
    discourse_shape["question_budget"] = 0


def _write_class_pressure(write_class: str) -> float:
    if write_class in {"body_risk", "bond_protection"}:
        return 0.86
    if write_class in {"repair_trace", "unresolved_tension"}:
        return 0.52
    if write_class == "safe_repeat":
        return 0.18
    return 0.0


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _clone_mapping_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_mapping_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_mapping_tree(item) for item in value]
    return value


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _group(context: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _mapping(context.get(name))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _max_float(*values: Any) -> float:
    return max((_float01(value) for value in values), default=0.0)


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered
