from __future__ import annotations

from dataclasses import dataclass, field, fields
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


@dataclass(frozen=True)
class ProtectiveTracePalaceConfig:
    boundary_only_pressure: float = 0.72
    body_risk_write_pressure: float = 0.86
    repair_trace_write_pressure: float = 0.52
    safe_repeat_write_pressure: float = 0.18
    current_explicit_weight: float = 0.46
    current_trigger_weight: float = 0.22
    current_hyperarousal_weight: float = 0.2
    current_unclear_field_weight: float = 0.12
    flash_explicit_weight: float = 0.58
    flash_memory_weight: float = 0.24
    flash_body_weight: float = 0.12
    flash_current_weight: float = 0.06
    dream_explicit_weight: float = 0.74
    dream_rem_weight: float = 0.26
    density_write_weight: float = 0.44
    density_memory_weight: float = 0.24
    density_flash_weight: float = 0.14
    density_safety_weight: float = 0.1
    density_current_weight: float = 0.08
    reentry_density_weight: float = 0.32
    reentry_memory_weight: float = 0.2
    reentry_stimulus_weight: float = 0.14
    reentry_novelty_weight: float = 0.08
    reentry_trigger_weight: float = 0.1
    reentry_current_weight: float = 0.1
    reentry_unclear_field_weight: float = 0.06
    stabilization_reentry_weight: float = 0.3
    stabilization_flash_weight: float = 0.18
    stabilization_dream_weight: float = 0.12
    stabilization_somatic_weight: float = 0.16
    stabilization_current_weight: float = 0.14
    stabilization_hyperarousal_weight: float = 0.12
    stabilization_unclear_field_weight: float = 0.08
    trusted_base_weight: float = 0.44
    trusted_clarity_weight: float = 0.24
    trusted_recovery_weight: float = 0.2
    trusted_somatic_clear_weight: float = 0.12
    readiness_recovery_weight: float = 0.34
    readiness_trusted_weight: float = 0.3
    readiness_clarity_weight: float = 0.18
    readiness_stability_weight: float = 0.18
    reason_presence_threshold: float = 0.45
    reason_reentry_threshold: float = 0.5
    mode_current_crisis_threshold: float = 0.55
    mode_hyperarousal_threshold: float = 0.45
    mode_stabilization_threshold: float = 0.58
    mode_reentry_threshold: float = 0.5
    mode_rem_replay_threshold: float = 0.55
    mode_dream_intrusion_threshold: float = 0.55
    mode_sensory_flash_threshold: float = 0.62
    mode_safe_reconsolidation_threshold: float = 0.58
    mode_stabilization_ceiling_for_recovery: float = 0.48
    mode_recovery_path_threshold: float = 0.52
    mode_trusted_window_threshold: float = 0.48

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> ProtectiveTracePalaceConfig:
        payload = _mapping(values)
        allowed = {item.name for item in fields(cls)}
        return cls(
            **{
                key: _float01(value)
                for key, value in payload.items()
                if key in allowed
            }
        )


def derive_protective_trace_palace_state(
    expression_context_state: Mapping[str, Any] | None,
    *,
    stimulus_history_influence: StimulusHistoryInfluence | Mapping[str, Any] | None = None,
    config: ProtectiveTracePalaceConfig | None = None,
) -> ProtectiveTracePalaceState:
    active_config = config or ProtectiveTracePalaceConfig()
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
    protective_write_pressure = _write_class_pressure(write_class, active_config)
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
        active_config.boundary_only_pressure
        if _text(safety.get("dialogue_permission")) == "boundary_only"
        else 0.0,
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
        * active_config.current_explicit_weight
        + trigger_pressure * active_config.current_trigger_weight
        + hyperarousal_pressure * active_config.current_hyperarousal_weight
        + (1.0 - field_clarity) * active_config.current_unclear_field_weight
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
        * active_config.flash_explicit_weight
        + memory_reentry * active_config.flash_memory_weight
        + body_pressure * active_config.flash_body_weight
        + current_crisis_binding * active_config.flash_current_weight
    )
    dream_intrusion_pressure = _clamp01(
        _max_float(
            protective.get("dream_intrusion_pressure"),
            protective.get("dream_pressure"),
            sleep.get("dream_intrusion_pressure"),
            sleep.get("nightmare_pressure"),
            sleep.get("replay_pressure"),
        )
        * active_config.dream_explicit_weight
        + rem_replay_pressure * active_config.dream_rem_weight
    )
    protective_trace_density = _clamp01(
        max(explicit_density, protective_write_pressure)
        * active_config.density_write_weight
        + memory_reentry * active_config.density_memory_weight
        + sensory_flash_risk * active_config.density_flash_weight
        + safety_pressure * active_config.density_safety_weight
        + current_crisis_binding * active_config.density_current_weight
    )
    reentry_sensitivity = _clamp01(
        protective_trace_density * active_config.reentry_density_weight
        + memory_reentry * active_config.reentry_memory_weight
        + stimulus_pressure * active_config.reentry_stimulus_weight
        + novelty_pressure * active_config.reentry_novelty_weight
        + trigger_pressure * active_config.reentry_trigger_weight
        + current_crisis_binding * active_config.reentry_current_weight
        + (1.0 - field_clarity) * active_config.reentry_unclear_field_weight
    )
    somatic_reactivation = _clamp01(
        _max_float(protective.get("somatic_reactivation"), body_pressure)
    )
    stabilization_need = _clamp01(
        reentry_sensitivity * active_config.stabilization_reentry_weight
        + sensory_flash_risk * active_config.stabilization_flash_weight
        + dream_intrusion_pressure * active_config.stabilization_dream_weight
        + somatic_reactivation * active_config.stabilization_somatic_weight
        + current_crisis_binding * active_config.stabilization_current_weight
        + hyperarousal_pressure * active_config.stabilization_hyperarousal_weight
        + (1.0 - field_clarity) * active_config.stabilization_unclear_field_weight
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
        * active_config.trusted_base_weight
        + field_clarity * active_config.trusted_clarity_weight
        + recovery_path_strength * active_config.trusted_recovery_weight
        + (1.0 - somatic_reactivation)
        * active_config.trusted_somatic_clear_weight
    )
    safe_reconsolidation_readiness = _clamp01(
        recovery_path_strength * active_config.readiness_recovery_weight
        + trusted_reentry_window * active_config.readiness_trusted_weight
        + field_clarity * active_config.readiness_clarity_weight
        + (1.0 - stabilization_need) * active_config.readiness_stability_weight
    )

    reasons: list[str] = []
    if write_class:
        reasons.append(f"memory.write_class.{write_class}")
    if protective_trace_density >= active_config.reason_presence_threshold:
        reasons.append("protective_trace.density")
    if current_crisis_binding >= active_config.reason_presence_threshold:
        reasons.append("protective_trace.current_crisis_binding")
    if reentry_sensitivity >= active_config.reason_reentry_threshold:
        reasons.append("protective_trace.reentry_sensitive")
    if trigger_pressure >= active_config.reason_presence_threshold:
        reasons.append("protective_trace.trigger_pressure")
    if hyperarousal_pressure >= active_config.reason_presence_threshold:
        reasons.append("body.hyperarousal")
    if rem_replay_pressure >= active_config.reason_presence_threshold:
        reasons.append("sleep.rem_replay")
    if sensory_flash_risk >= active_config.reason_presence_threshold:
        reasons.append("protective_trace.sensory_flash")
    if dream_intrusion_pressure >= active_config.reason_presence_threshold:
        reasons.append("protective_trace.dream_intrusion")
    if somatic_reactivation >= active_config.reason_presence_threshold:
        reasons.append("body.somatic_reactivation")
    if recovery_path_strength >= active_config.reason_presence_threshold:
        reasons.append("recovery.path_available")

    dominant_mode = "ambient"
    if (
        current_crisis_binding >= active_config.mode_current_crisis_threshold
        and hyperarousal_pressure >= active_config.mode_hyperarousal_threshold
    ):
        dominant_mode = "protective_hold"
    elif (
        stabilization_need >= active_config.mode_stabilization_threshold
        and reentry_sensitivity >= active_config.mode_reentry_threshold
    ):
        dominant_mode = "protective_hold"
    elif (
        rem_replay_pressure >= active_config.mode_rem_replay_threshold
        or dream_intrusion_pressure >= active_config.mode_dream_intrusion_threshold
        or sensory_flash_risk >= active_config.mode_sensory_flash_threshold
    ):
        dominant_mode = "restabilize"
    elif (
        safe_reconsolidation_readiness
        >= active_config.mode_safe_reconsolidation_threshold
        and stabilization_need <= active_config.mode_stabilization_ceiling_for_recovery
    ):
        dominant_mode = "safe_reconsolidation"
    elif (
        recovery_path_strength >= active_config.mode_recovery_path_threshold
        and trusted_reentry_window >= active_config.mode_trusted_window_threshold
    ):
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


def _write_class_pressure(
    write_class: str,
    config: ProtectiveTracePalaceConfig,
) -> float:
    if write_class in {"body_risk", "bond_protection"}:
        return config.body_risk_write_pressure
    if write_class in {"repair_trace", "unresolved_tension"}:
        return config.repair_trace_write_pressure
    if write_class == "safe_repeat":
        return config.safe_repeat_write_pressure
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
