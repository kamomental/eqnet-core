from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


BASIS_CONFIDENCE_LOW = 0.34
CLOSURE_TENSION_HIGH = 0.5
RECONSTRUCTION_RISK_CAUTION = 0.52
RECONSTRUCTION_RISK_HIGH = 0.7
SHARED_ANCHOR_THRESHOLD = 0.5


@dataclass(frozen=True)
class ClosurePacket:
    """生成閉包を判断ではなく、反応契約の根拠候補として束ねる packet。"""

    dominant_basis_keys: tuple[str, ...] = ()
    generated_constraints: tuple[str, ...] = ()
    generated_affordances: tuple[str, ...] = ()
    inhibition_reasons: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    basis_confidence: float = 0.0
    closure_tension: float = 0.0
    reconstruction_risk: float = 0.0
    contract_bias: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_basis_keys": list(self.dominant_basis_keys),
            "generated_constraints": list(self.generated_constraints),
            "generated_affordances": list(self.generated_affordances),
            "inhibition_reasons": list(self.inhibition_reasons),
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "basis_confidence": round(self.basis_confidence, 4),
            "closure_tension": round(self.closure_tension, 4),
            "reconstruction_risk": round(self.reconstruction_risk, 4),
            "contract_bias": dict(self.contract_bias),
        }

    def reason_tags(self) -> tuple[str, ...]:
        """reaction_contract に混ぜられる監査用タグを返す。"""

        tags: list[str] = []
        for value in (
            *self.generated_constraints,
            *self.generated_affordances,
            *self.inhibition_reasons,
            *self.uncertainty_reasons,
        ):
            text = _text(value)
            if text:
                tags.append(f"closure:{text}")
        return tuple(dict.fromkeys(tags))


def derive_closure_packet(
    *,
    memory_dynamics_state: Mapping[str, Any] | None = None,
    association_graph_state: Mapping[str, Any] | None = None,
    terrain_dynamics_state: Mapping[str, Any] | None = None,
    subjective_scene_state: Mapping[str, Any] | None = None,
    self_other_attribution_state: Mapping[str, Any] | None = None,
    shared_presence_state: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
) -> ClosurePacket:
    """記憶・連想・場の状態から、反応契約の根拠 packet を投影する。"""

    memory = dict(memory_dynamics_state or {})
    association = _association_payload(association_graph_state)
    terrain = dict(terrain_dynamics_state or {})
    scene = dict(subjective_scene_state or {})
    attribution = dict(self_other_attribution_state or {})
    presence = dict(shared_presence_state or {})
    joint = dict(joint_state or {})

    basis_keys = _dominant_basis_keys(
        memory=memory,
        association=association,
        scene=scene,
        presence=presence,
    )
    basis_confidence = _clamp01(
        _float01(memory.get("activation_confidence")) * 0.34
        + _float01(memory.get("monument_salience")) * 0.18
        + _float01(association.get("dominant_weight")) * 0.2
        + _float01(joint.get("common_ground")) * 0.14
        + _float01(presence.get("co_presence")) * 0.14
    )
    closure_tension = _clamp01(
        _float01(memory.get("memory_tension")) * 0.34
        + _float01(terrain.get("barrier_height")) * 0.18
        + _float01(scene.get("tension")) * 0.16
        + _float01(joint.get("shared_tension")) * 0.16
        + _float01(attribution.get("unknown_likelihood")) * 0.16
    )
    reconstruction_risk = _clamp01(
        _float01(memory.get("memory_tension")) * 0.28
        + max(0.0, 1.0 - basis_confidence) * 0.24
        + _float01(attribution.get("unknown_likelihood")) * 0.2
        + max(0.0, 1.0 - _float01(presence.get("boundary_stability"))) * 0.16
        + _float01(memory.get("forgetting_pressure")) * 0.12
    )

    generated_constraints = _derive_constraints(
        memory=memory,
        basis_confidence=basis_confidence,
        closure_tension=closure_tension,
        reconstruction_risk=reconstruction_risk,
        attribution=attribution,
        presence=presence,
    )
    generated_affordances = _derive_affordances(
        memory=memory,
        joint=joint,
        presence=presence,
        basis_confidence=basis_confidence,
        closure_tension=closure_tension,
    )
    inhibition_reasons = _derive_inhibition_reasons(
        attribution=attribution,
        presence=presence,
        closure_tension=closure_tension,
        reconstruction_risk=reconstruction_risk,
    )
    uncertainty_reasons = _derive_uncertainty_reasons(
        association=association,
        basis_confidence=basis_confidence,
        reconstruction_risk=reconstruction_risk,
    )
    contract_bias = _derive_contract_bias(
        generated_constraints=generated_constraints,
        generated_affordances=generated_affordances,
        inhibition_reasons=inhibition_reasons,
        uncertainty_reasons=uncertainty_reasons,
        reconstruction_risk=reconstruction_risk,
        basis_confidence=basis_confidence,
        closure_tension=closure_tension,
    )

    return ClosurePacket(
        dominant_basis_keys=basis_keys,
        generated_constraints=generated_constraints,
        generated_affordances=generated_affordances,
        inhibition_reasons=inhibition_reasons,
        uncertainty_reasons=uncertainty_reasons,
        basis_confidence=basis_confidence,
        closure_tension=closure_tension,
        reconstruction_risk=reconstruction_risk,
        contract_bias=contract_bias,
    )


def coerce_closure_packet(value: Mapping[str, Any] | ClosurePacket | None) -> ClosurePacket:
    if isinstance(value, ClosurePacket):
        return value
    payload = dict(value or {})
    return ClosurePacket(
        dominant_basis_keys=_text_tuple(payload.get("dominant_basis_keys")),
        generated_constraints=_text_tuple(payload.get("generated_constraints")),
        generated_affordances=_text_tuple(payload.get("generated_affordances")),
        inhibition_reasons=_text_tuple(payload.get("inhibition_reasons")),
        uncertainty_reasons=_text_tuple(payload.get("uncertainty_reasons")),
        basis_confidence=_float01(payload.get("basis_confidence")),
        closure_tension=_float01(payload.get("closure_tension")),
        reconstruction_risk=_float01(payload.get("reconstruction_risk")),
        contract_bias=dict(payload.get("contract_bias") or {}),
    )


def _derive_constraints(
    *,
    memory: Mapping[str, Any],
    basis_confidence: float,
    closure_tension: float,
    reconstruction_risk: float,
    attribution: Mapping[str, Any],
    presence: Mapping[str, Any],
) -> tuple[str, ...]:
    constraints: list[str] = []
    if (
        reconstruction_risk >= RECONSTRUCTION_RISK_CAUTION
        or closure_tension >= CLOSURE_TENSION_HIGH
        or _float01(attribution.get("unknown_likelihood")) >= 0.5
    ):
        constraints.append("do_not_overinterpret")
    if (
        closure_tension >= CLOSURE_TENSION_HIGH
        or _text(memory.get("dominant_relation_type")) == "unfinished_carry"
    ):
        constraints.append("leave_return_point")
    if basis_confidence <= BASIS_CONFIDENCE_LOW:
        constraints.append("keep_basis_visible")
    if reconstruction_risk >= RECONSTRUCTION_RISK_HIGH:
        constraints.append("do_not_reconstruct_memory")
    if _float01(presence.get("boundary_stability")) <= 0.34:
        constraints.append("preserve_boundary")
    return tuple(dict.fromkeys(constraints))


def _derive_affordances(
    *,
    memory: Mapping[str, Any],
    joint: Mapping[str, Any],
    presence: Mapping[str, Any],
    basis_confidence: float,
    closure_tension: float,
) -> tuple[str, ...]:
    affordances: list[str] = []
    relation_type = _text(memory.get("dominant_relation_type"))
    causal_type = _text(memory.get("dominant_causal_type"))
    shared_signal = max(
        _float01(joint.get("common_ground")),
        _float01(presence.get("co_presence")),
        _float01(presence.get("shared_scene_salience")),
    )
    if relation_type in {"same_anchor", "recurrent_association"} or shared_signal >= SHARED_ANCHOR_THRESHOLD:
        affordances.append("shared_anchor")
    if (
        shared_signal >= 0.55
        and basis_confidence >= 0.32
        and closure_tension < CLOSURE_TENSION_HIGH
    ):
        affordances.append("gentle_join")
    if causal_type in {"enabled_by", "reopened_by"} or _float01(joint.get("repair_readiness")) >= 0.45:
        affordances.append("repair_window")
    return tuple(dict.fromkeys(affordances))


def _derive_inhibition_reasons(
    *,
    attribution: Mapping[str, Any],
    presence: Mapping[str, Any],
    closure_tension: float,
    reconstruction_risk: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if closure_tension >= CLOSURE_TENSION_HIGH:
        reasons.append("memory_tension")
    if reconstruction_risk >= RECONSTRUCTION_RISK_CAUTION:
        reasons.append("reconstruction_risk")
    if _float01(attribution.get("unknown_likelihood")) >= 0.5:
        reasons.append("unknown_attribution")
    if _float01(presence.get("boundary_stability")) <= 0.4:
        reasons.append("unstable_boundary")
    return tuple(dict.fromkeys(reasons))


def _derive_uncertainty_reasons(
    *,
    association: Mapping[str, Any],
    basis_confidence: float,
    reconstruction_risk: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if basis_confidence <= BASIS_CONFIDENCE_LOW:
        reasons.append("low_basis_confidence")
    if _float01(association.get("winner_margin")) <= 0.08 and association:
        reasons.append("weak_association_margin")
    if reconstruction_risk >= RECONSTRUCTION_RISK_HIGH:
        reasons.append("high_reconstruction_risk")
    return tuple(dict.fromkeys(reasons))


def _derive_contract_bias(
    *,
    generated_constraints: tuple[str, ...],
    generated_affordances: tuple[str, ...],
    inhibition_reasons: tuple[str, ...],
    uncertainty_reasons: tuple[str, ...],
    reconstruction_risk: float,
    basis_confidence: float,
    closure_tension: float,
) -> dict[str, object]:
    bias: dict[str, object] = {}
    if "do_not_overinterpret" in generated_constraints:
        bias["interpretation_budget_bias"] = "none"
    if "leave_return_point" in generated_constraints:
        bias["closure_mode_bias"] = "leave_open"
    if "preserve_boundary" in generated_constraints or "unstable_boundary" in inhibition_reasons:
        bias["distance_mode_bias"] = "guarded"
    if reconstruction_risk >= RECONSTRUCTION_RISK_HIGH or closure_tension >= 0.68:
        bias["stance_bias"] = "hold"
        bias["response_channel_bias"] = "hold"
    elif "gentle_join" in generated_affordances and basis_confidence >= 0.38:
        bias["stance_bias"] = "join"
    if "low_basis_confidence" in uncertainty_reasons:
        bias["initiative_bias"] = "yield"
    return bias


def _dominant_basis_keys(
    *,
    memory: Mapping[str, Any],
    association: Mapping[str, Any],
    scene: Mapping[str, Any],
    presence: Mapping[str, Any],
) -> tuple[str, ...]:
    keys = [
        _text(memory.get("dominant_link_key")),
        _text(memory.get("recall_anchor")),
        _text(association.get("dominant_link_key") or association.get("dominant_link_id")),
        _text(memory.get("dominant_relation_type")),
        _text(memory.get("dominant_causal_type")),
        _text(scene.get("anchor_frame")),
        _text(presence.get("dominant_mode")),
    ]
    return tuple(dict.fromkeys(item for item in keys if item))


def _association_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    state_hint = payload.get("state_hint")
    if isinstance(state_hint, Mapping):
        merged = dict(payload)
        merged.update({key: item for key, item in state_hint.items() if key not in merged or not merged[key]})
        return merged
    return payload


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(text for item in value if (text := _text(item))))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _clamp01(number)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
