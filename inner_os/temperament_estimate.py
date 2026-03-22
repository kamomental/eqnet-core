from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TemperamentEstimate:
    risk_tolerance: float
    ambiguity_tolerance: float
    curiosity_drive: float
    bond_drive: float
    recovery_discipline: float
    protect_floor: float
    initiative_persistence: float
    leader_tendency: float
    hero_tendency: float
    forward_trace: float
    guard_trace: float
    bond_trace: float
    recovery_trace: float

    def to_dict(self) -> dict[str, float]:
        return {
            "risk_tolerance": round(self.risk_tolerance, 4),
            "ambiguity_tolerance": round(self.ambiguity_tolerance, 4),
            "curiosity_drive": round(self.curiosity_drive, 4),
            "bond_drive": round(self.bond_drive, 4),
            "recovery_discipline": round(self.recovery_discipline, 4),
            "protect_floor": round(self.protect_floor, 4),
            "initiative_persistence": round(self.initiative_persistence, 4),
            "leader_tendency": round(self.leader_tendency, 4),
            "hero_tendency": round(self.hero_tendency, 4),
            "forward_trace": round(self.forward_trace, 4),
            "guard_trace": round(self.guard_trace, 4),
            "bond_trace": round(self.bond_trace, 4),
            "recovery_trace": round(self.recovery_trace, 4),
        }


def derive_temperament_estimate(state: Mapping[str, Any] | None) -> TemperamentEstimate:
    payload = dict(state or {})
    caution_bias = _float01(payload.get("caution_bias"), 0.4)
    affiliation_bias = _float01(payload.get("affiliation_bias"), 0.45)
    exploration_bias = _float01(payload.get("exploration_bias"), 0.4)
    reflective_bias = _float01(payload.get("reflective_bias"), 0.45)
    trust_bias = _float01(payload.get("trust_bias"), 0.45)
    safety_bias = _float01(payload.get("safety_bias"), 0.35)
    continuity_score = _float01(payload.get("continuity_score"), 0.48)
    recent_strain = _float01(payload.get("recent_strain"), 0.0)
    stress = _float01(payload.get("stress"), 0.0)
    recovery_need = _float01(payload.get("recovery_need"), 0.0)

    forward_trace = _float01(payload.get("temperament_forward_trace"), 0.0)
    guard_trace = _float01(payload.get("temperament_guard_trace"), 0.0)
    bond_trace = _float01(payload.get("temperament_bond_trace"), 0.0)
    recovery_trace = _float01(payload.get("temperament_recovery_trace"), 0.0)
    forward_bias = _float01(payload.get("temperament_forward_bias"), 0.0)
    guard_bias = _float01(payload.get("temperament_guard_bias"), 0.0)
    bond_bias = _float01(payload.get("temperament_bond_bias"), 0.0)
    recovery_bias = _float01(payload.get("temperament_recovery_bias"), 0.0)

    forward_signal = _clamp01(forward_trace + forward_bias * 0.35)
    guard_signal = _clamp01(guard_trace + guard_bias * 0.35)
    bond_signal = _clamp01(bond_trace + bond_bias * 0.35)
    recovery_signal = _clamp01(recovery_trace + recovery_bias * 0.35)

    risk_tolerance = _clamp01(
        0.16
        + 0.34 * exploration_bias
        + 0.2 * forward_signal
        + 0.08 * trust_bias
        - 0.28 * caution_bias
        - 0.14 * safety_bias
        - 0.12 * recovery_signal
        - 0.08 * recent_strain
    )
    ambiguity_tolerance = _clamp01(
        0.18
        + 0.24 * exploration_bias
        + 0.18 * reflective_bias
        + 0.1 * forward_signal
        - 0.2 * caution_bias
        - 0.08 * guard_signal
    )
    curiosity_drive = _clamp01(
        0.18
        + 0.34 * exploration_bias
        + 0.18 * forward_signal
        + 0.08 * trust_bias
        - 0.12 * stress
        - 0.1 * recovery_need
    )
    bond_drive = _clamp01(
        0.2
        + 0.34 * affiliation_bias
        + 0.2 * bond_signal
        + 0.06 * trust_bias
        - 0.08 * caution_bias
    )
    recovery_discipline = _clamp01(
        0.18
        + 0.26 * caution_bias
        + 0.16 * reflective_bias
        + 0.18 * recovery_signal
        + 0.08 * safety_bias
        + 0.08 * recovery_need
    )
    protect_floor = _clamp01(
        0.18
        + 0.28 * caution_bias
        + 0.2 * guard_signal
        + 0.12 * recovery_signal
        + 0.08 * safety_bias
        - 0.12 * forward_signal
    )
    initiative_persistence = _clamp01(
        0.16
        + 0.24 * forward_signal
        + 0.18 * exploration_bias
        + 0.12 * bond_signal
        + 0.08 * continuity_score
        - 0.1 * recovery_signal
        - 0.08 * caution_bias
    )
    leader_tendency = _clamp01(
        0.08
        + 0.28 * bond_drive
        + 0.22 * recovery_discipline
        + 0.18 * protect_floor
        + 0.16 * bond_signal
        + 0.1 * continuity_score
        - 0.08 * risk_tolerance
    )
    hero_tendency = _clamp01(
        0.06
        + 0.3 * risk_tolerance
        + 0.22 * curiosity_drive
        + 0.18 * initiative_persistence
        + 0.14 * forward_signal
        - 0.14 * recovery_discipline
        - 0.1 * protect_floor
    )

    return TemperamentEstimate(
        risk_tolerance=risk_tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
        curiosity_drive=curiosity_drive,
        bond_drive=bond_drive,
        recovery_discipline=recovery_discipline,
        protect_floor=protect_floor,
        initiative_persistence=initiative_persistence,
        leader_tendency=leader_tendency,
        hero_tendency=hero_tendency,
        forward_trace=forward_trace,
        guard_trace=guard_trace,
        bond_trace=bond_trace,
        recovery_trace=recovery_trace,
    )


def advance_temperament_traces(
    state: Mapping[str, Any] | None,
    *,
    protection_mode: str,
    protection_strength: float,
    body_recovery_guard: str,
    initiative_readiness: str,
    initiative_followup_state: str,
    memory_write_class: str,
) -> dict[str, float]:
    payload = dict(state or {})
    prev_forward = _float01(payload.get("temperament_forward_trace"), 0.0)
    prev_guard = _float01(payload.get("temperament_guard_trace"), 0.0)
    prev_bond = _float01(payload.get("temperament_bond_trace"), 0.0)
    prev_recovery = _float01(payload.get("temperament_recovery_trace"), 0.0)
    alpha = 0.12

    forward_target = _clamp01(
        (0.48 if initiative_readiness == "ready" else 0.0)
        + (0.24 if initiative_followup_state == "offer_next_step" else 0.0)
        + (0.14 if protection_mode in {"monitor", "repair"} else 0.0)
        - (0.22 if body_recovery_guard == "recovery_first" else 0.0)
        - (0.14 if protection_mode in {"shield", "stabilize"} else 0.0)
    )
    guard_target = _clamp01(
        (0.42 if body_recovery_guard in {"guarded", "recovery_first"} else 0.0)
        + (0.26 if protection_mode in {"contain", "stabilize", "shield"} else 0.0)
        + 0.12 * _clamp01(protection_strength)
    )
    bond_target = _clamp01(
        (0.4 if memory_write_class == "bond_protection" else 0.0)
        + (0.22 if memory_write_class == "repair_trace" else 0.0)
        + (0.18 if initiative_followup_state == "reopen_softly" else 0.0)
    )
    recovery_target = _clamp01(
        (0.44 if body_recovery_guard == "recovery_first" else 0.0)
        + (0.22 if protection_mode == "stabilize" else 0.0)
        + (0.12 if memory_write_class == "body_risk" else 0.0)
    )

    return {
        "temperament_forward_trace": round(_ema(prev_forward, forward_target, alpha), 4),
        "temperament_guard_trace": round(_ema(prev_guard, guard_target, alpha), 4),
        "temperament_bond_trace": round(_ema(prev_bond, bond_target, alpha), 4),
        "temperament_recovery_trace": round(_ema(prev_recovery, recovery_target, alpha), 4),
    }


def _ema(previous: float, target: float, alpha: float) -> float:
    return _clamp01((1.0 - alpha) * previous + alpha * target)


def _float01(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
