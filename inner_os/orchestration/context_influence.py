from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ContextInfluence:
    gate_pressure: float = 0.0
    surface_caution: float = 0.0
    support_permission: str = ""
    memory_reentry_pressure: float = 0.0
    safety_boundary: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_pressure": self.gate_pressure,
            "surface_caution": self.surface_caution,
            "support_permission": self.support_permission,
            "memory_reentry_pressure": self.memory_reentry_pressure,
            "safety_boundary": self.safety_boundary,
            "reasons": list(self.reasons),
        }


def derive_context_influence(
    expression_context_state: Mapping[str, Any] | None,
) -> ContextInfluence:
    context = _mapping(expression_context_state)
    safety = _group(context, "safety")
    body = _group(context, "body")
    homeostasis = _group(context, "homeostasis")
    memory = _group(context, "memory")
    temperament = _group(context, "temperament")
    identity = _group(context, "identity")
    green_kernel = _group(context, "green_kernel")
    norm = _group(context, "norm")
    culture = _group(context, "culture")
    environment = _group(context, "environment")

    support_permission = _text(safety.get("dialogue_permission"))
    safety_boundary = _text(safety.get("risk_state"))
    reasons: list[str] = []

    symbolic_gate_pressure = 0.0
    if support_permission == "boundary_only":
        symbolic_gate_pressure = max(symbolic_gate_pressure, 0.72)
        reasons.append("safety.boundary_only")
    if safety_boundary == "guarded_context":
        symbolic_gate_pressure = max(symbolic_gate_pressure, 0.66)
        reasons.append("safety.guarded_context")
    if _text(homeostasis.get("budget_state")) == "recovering":
        symbolic_gate_pressure = max(symbolic_gate_pressure, 0.58)
        reasons.append("homeostasis.recovering")

    numeric_gate_pressure = _max_float(
        body.get("stress"),
        body.get("recovery_need"),
        homeostasis.get("load"),
        memory.get("memory_tension"),
        memory.get("reconsolidation_priority"),
        temperament.get("protect_floor"),
        identity.get("boundary_need"),
        green_kernel.get("guardedness"),
    )
    if numeric_gate_pressure >= 0.55:
        reasons.append("state.guard_pressure")

    surface_caution = _max_float(
        norm.get("privacy_level"),
        norm.get("norm_pressure"),
        culture.get("politeness_pressure"),
        1.0 - _float01(culture.get("directness_ceiling"))
        if culture.get("directness_ceiling") is not None
        else 0.0,
        environment.get("social_pressure"),
    )
    if surface_caution >= 0.45:
        reasons.append("surface.caution")

    memory_reentry_pressure = _max_float(
        memory.get("ignition_readiness"),
        memory.get("replay_priority"),
        memory.get("reconsolidation_priority"),
        green_kernel.get("reopening_pull"),
    )
    if memory_reentry_pressure >= 0.45:
        reasons.append("memory.reentry")

    if support_permission == "allow_support":
        reasons.append("support.allowed")

    return ContextInfluence(
        gate_pressure=max(symbolic_gate_pressure, numeric_gate_pressure),
        surface_caution=surface_caution,
        support_permission=support_permission,
        memory_reentry_pressure=memory_reentry_pressure,
        safety_boundary=safety_boundary,
        reasons=tuple(_dedupe(reasons)),
    )


def apply_context_influence_to_contract_inputs(
    *,
    contract_inputs: Mapping[str, Any],
    influence: ContextInfluence,
) -> dict[str, Any]:
    adjusted = _clone_mapping_tree(contract_inputs)
    if (
        influence.gate_pressure >= 0.55
        and influence.support_permission != "allow_support"
    ):
        _project_guarded_context(adjusted, influence)
    elif (
        influence.support_permission == "allow_support"
        and influence.gate_pressure < 0.72
    ):
        _project_support_context(adjusted, influence)
    elif influence.surface_caution >= 0.45:
        _project_surface_caution(adjusted, influence)
    return adjusted


def _project_guarded_context(
    contract_inputs: dict[str, Any],
    influence: ContextInfluence,
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
        influence.gate_pressure,
    )

    turn_delta = _ensure_dict(contract_inputs, "turn_delta")
    turn_delta["kind"] = "reopening_thread"
    turn_delta["preferred_act"] = "leave_return_point_from_anchor"


def _project_support_context(
    contract_inputs: dict[str, Any],
    influence: ContextInfluence,
) -> None:
    interaction_policy = _ensure_dict(contract_inputs, "interaction_policy")
    interaction_policy["response_strategy"] = "user_led_support"

    action_posture = _ensure_dict(contract_inputs, "action_posture")
    action_posture["boundary_mode"] = "soft_hold"
    action_posture["question_budget"] = 0

    actuation_plan = _ensure_dict(contract_inputs, "actuation_plan")
    actuation_plan["execution_mode"] = "user_led_support"
    actuation_plan["response_channel"] = "speak"
    actuation_plan["wait_before_action"] = ""

    discourse_shape = _ensure_dict(contract_inputs, "discourse_shape")
    discourse_shape["shape_id"] = "reflect_step"
    discourse_shape["question_budget"] = 0

    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    packet["conversation_phase"] = "continuing_thread"
    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0
    constraints["prefer_return_point"] = False

    source_state = _ensure_dict(packet, "source_state")
    source_state["utterance_reason_offer"] = "user_led_support"
    source_state["utterance_reason_preserve"] = "keep_it_small"
    source_state["context_surface_caution"] = influence.surface_caution

    turn_delta = _ensure_dict(contract_inputs, "turn_delta")
    turn_delta["kind"] = "continuing_thread"
    turn_delta["preferred_act"] = "user_led_support"


def _project_surface_caution(
    contract_inputs: dict[str, Any],
    influence: ContextInfluence,
) -> None:
    packet = _ensure_dict(contract_inputs, "surface_context_packet")
    constraints = _ensure_dict(packet, "constraints")
    constraints["max_questions"] = 0

    source_state = _ensure_dict(packet, "source_state")
    source_state["context_surface_caution"] = influence.surface_caution
    source_state["utterance_reason_preserve"] = (
        source_state.get("utterance_reason_preserve") or "keep_it_small"
    )


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _group(context: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _mapping(context.get(name))


def _max_float(*values: Any) -> float:
    return max((_float01(value) for value in values), default=0.0)


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered
