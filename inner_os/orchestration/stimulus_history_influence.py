from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class StimulusHistoryInfluence:
    stimulus_pressure: float = 0.0
    novelty_pressure: float = 0.0
    habituation_pressure: float = 0.0
    memory_reentry_pressure: float = 0.0
    field_clarity: float = 0.0
    gradient_confidence: float = 0.0
    response_bias: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus_pressure": self.stimulus_pressure,
            "novelty_pressure": self.novelty_pressure,
            "habituation_pressure": self.habituation_pressure,
            "memory_reentry_pressure": self.memory_reentry_pressure,
            "field_clarity": self.field_clarity,
            "gradient_confidence": self.gradient_confidence,
            "response_bias": self.response_bias,
            "reasons": list(self.reasons),
        }


def derive_stimulus_history_influence(
    expression_context_state: Mapping[str, Any] | None,
) -> StimulusHistoryInfluence:
    context = _mapping(expression_context_state)
    qualia = _group(context, "qualia")
    qualia_structure = _first_mapping(
        context.get("qualia_structure"),
        context.get("qualia_structure_state"),
        qualia.get("structure_state"),
    )
    qualia_state = _first_mapping(
        context.get("qualia_state"),
        qualia.get("state"),
        qualia,
    )
    memory = _group(context, "memory")
    environment = _group(context, "environment")

    emergence = _max_float(
        qualia_structure.get("emergence"),
        _nested_axis_value(context.get("qualia_structure_axes"), "emergence"),
    )
    drift = _max_float(
        qualia_structure.get("drift"),
        _nested_axis_value(context.get("qualia_structure_axes"), "drift"),
    )
    intensity = _float01(qualia_structure.get("intensity"))
    stability = _max_float(
        qualia_structure.get("stability"),
        _nested_axis_value(context.get("qualia_structure_axes"), "stability"),
    )
    memory_resonance = _max_float(
        qualia_structure.get("memory_resonance"),
        _nested_axis_value(context.get("qualia_structure_axes"), "resonance"),
    )
    habituation = _mean_sequence(qualia_state.get("habituation"))
    normalization = _mapping(qualia_state.get("normalization_stats"))
    field_clarity = _derive_field_clarity(normalization, environment)
    gradient_confidence = _derive_gradient_confidence(normalization, field_clarity)

    memory_reentry_pressure = _max_float(
        memory.get("ignition_readiness"),
        memory.get("replay_priority"),
        memory.get("reconsolidation_priority"),
        memory.get("memory_tension"),
        memory_resonance,
    )
    novelty_pressure = _clamp01(
        drift * 0.38
        + emergence * 0.24
        + intensity * 0.18
        + (1.0 - habituation) * 0.12
        + (1.0 - field_clarity) * 0.08
    )
    stimulus_pressure = _clamp01(
        emergence * 0.3
        + drift * 0.24
        + intensity * 0.2
        + memory_reentry_pressure * 0.16
        + (1.0 - field_clarity) * 0.1
    )
    habituation_pressure = _clamp01(
        habituation * 0.46
        + stability * 0.34
        + memory_resonance * 0.12
        + field_clarity * 0.08
    )

    reasons: list[str] = []
    if stimulus_pressure >= 0.58:
        reasons.append("stimulus.pressure")
    if novelty_pressure >= 0.55:
        reasons.append("stimulus.novelty")
    if habituation_pressure >= 0.55:
        reasons.append("stimulus.habituated")
    if memory_reentry_pressure >= 0.45:
        reasons.append("memory.reentry")
    if field_clarity <= 0.42:
        reasons.append("field.low_clarity")

    response_bias = "steady"
    if stimulus_pressure >= 0.6 and field_clarity <= 0.48:
        response_bias = "hold_for_clarity"
    elif novelty_pressure >= 0.62 and habituation_pressure <= 0.38:
        response_bias = "minimal_first_contact"
    elif habituation_pressure >= 0.58 and field_clarity >= 0.52:
        response_bias = "habituated_small_response"
    elif memory_reentry_pressure >= 0.58:
        response_bias = "memory_reentry_caution"

    return StimulusHistoryInfluence(
        stimulus_pressure=round(stimulus_pressure, 4),
        novelty_pressure=round(novelty_pressure, 4),
        habituation_pressure=round(habituation_pressure, 4),
        memory_reentry_pressure=round(memory_reentry_pressure, 4),
        field_clarity=round(field_clarity, 4),
        gradient_confidence=round(gradient_confidence, 4),
        response_bias=response_bias,
        reasons=tuple(_dedupe(reasons)),
    )


def apply_stimulus_history_influence_to_contract_inputs(
    *,
    contract_inputs: Mapping[str, Any],
    influence: StimulusHistoryInfluence,
) -> dict[str, Any]:
    adjusted = _clone_mapping_tree(contract_inputs)
    packet = _ensure_dict(adjusted, "surface_context_packet")
    source_state = _ensure_dict(packet, "source_state")
    source_state["stimulus_pressure"] = influence.stimulus_pressure
    source_state["stimulus_novelty_pressure"] = influence.novelty_pressure
    source_state["stimulus_habituation_pressure"] = influence.habituation_pressure
    source_state["stimulus_field_clarity"] = influence.field_clarity
    source_state["stimulus_gradient_confidence"] = influence.gradient_confidence
    source_state["stimulus_response_bias"] = influence.response_bias

    if influence.response_bias in {"hold_for_clarity", "minimal_first_contact"}:
        _project_stimulus_hold(adjusted, influence)
    elif influence.response_bias == "memory_reentry_caution":
        _project_memory_reentry_caution(adjusted, influence)
    elif influence.response_bias == "habituated_small_response":
        _project_habituated_small_response(adjusted, influence)
    return adjusted


def _project_stimulus_hold(
    contract_inputs: dict[str, Any],
    influence: StimulusHistoryInfluence,
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
    source_state["utterance_reason_preserve"] = "leave_open"
    source_state["organism_protective_tension"] = max(
        _float01(source_state.get("organism_protective_tension")),
        influence.stimulus_pressure,
        1.0 - influence.field_clarity,
    )

    turn_delta = _ensure_dict(contract_inputs, "turn_delta")
    turn_delta["kind"] = "reopening_thread"
    turn_delta["preferred_act"] = "leave_return_point_from_anchor"


def _project_memory_reentry_caution(
    contract_inputs: dict[str, Any],
    influence: StimulusHistoryInfluence,
) -> None:
    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    source_state = _ensure_dict(packet, "source_state")
    source_state["utterance_reason_offer"] = ""
    source_state["utterance_reason_preserve"] = "leave_open"
    source_state["utterance_reason_memory_frame"] = "reentry_caution"
    source_state["organism_protective_tension"] = max(
        _float01(source_state.get("organism_protective_tension")),
        influence.memory_reentry_pressure,
    )
    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0
    constraints["prefer_return_point"] = True

    action_posture = _ensure_dict(contract_inputs, "action_posture")
    action_posture["boundary_mode"] = "soft_hold"
    action_posture["question_budget"] = 0

    discourse_shape = _ensure_dict(contract_inputs, "discourse_shape")
    discourse_shape["shape_id"] = "reflect_hold"
    discourse_shape["question_budget"] = 0


def _project_habituated_small_response(
    contract_inputs: dict[str, Any],
    influence: StimulusHistoryInfluence,
) -> None:
    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    source_state = _ensure_dict(packet, "source_state")
    source_state["utterance_reason_preserve"] = "keep_it_small"
    source_state["stimulus_habituated_response"] = influence.habituation_pressure

    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0
    constraints["prefer_return_point"] = False

    action_posture = _ensure_dict(contract_inputs, "action_posture")
    action_posture["boundary_mode"] = "soft_hold"
    action_posture["question_budget"] = 0

    actuation_plan = _ensure_dict(contract_inputs, "actuation_plan")
    if _text(actuation_plan.get("response_channel")) != "hold":
        actuation_plan["response_channel"] = "speak"


def _derive_field_clarity(
    normalization_stats: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> float:
    explicit = environment.get("field_clarity")
    if explicit is not None:
        return _float01(explicit)
    values: list[float] = []
    for stat in normalization_stats.values():
        if not isinstance(stat, Mapping):
            continue
        if stat.get("range_trust") is not None:
            values.append(_float01(stat.get("range_trust")))
        if stat.get("gradient_confidence") is not None:
            values.append(_float01(stat.get("gradient_confidence")))
    if not values:
        fog = _max_float(environment.get("fog_density"), environment.get("noise_fog"))
        return _clamp01(1.0 - fog)
    return _clamp01(sum(values) / len(values))


def _derive_gradient_confidence(
    normalization_stats: Mapping[str, Any],
    fallback: float,
) -> float:
    values: list[float] = []
    for stat in normalization_stats.values():
        if isinstance(stat, Mapping) and stat.get("gradient_confidence") is not None:
            values.append(_float01(stat.get("gradient_confidence")))
    if not values:
        return _float01(fallback)
    return _clamp01(sum(values) / len(values))


def _nested_axis_value(value: Any, axis_name: str) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    axis = value.get(axis_name)
    if isinstance(axis, Mapping):
        return _float01(axis.get("value"))
    return 0.0


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _group(context: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _mapping(context.get(name))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mean_sequence(value: Any) -> float:
    if not isinstance(value, (list, tuple)):
        return 0.0
    numbers = [_float01(item) for item in value]
    if not numbers:
        return 0.0
    return _clamp01(sum(numbers) / len(numbers))


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
