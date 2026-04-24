from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


_PUBLIC_TOPOLOGIES = {"public_visible", "hierarchical", "threaded_group", "group_present"}


@dataclass(frozen=True)
class CurrentRiskTokenPolicy:
    tokens: tuple[str, ...] = ()
    acute_pressure: float = 0.0
    guarded_pressure: float = 0.0
    public_exposure: float = 0.0
    reason: str = "clear"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_current_risk_token_policy(
    *,
    safety_bias: float = 0.0,
    stress: float = 0.0,
    recovery_need: float = 0.0,
    recent_strain: float = 0.0,
    privacy_level: float = 0.5,
    social_topology: str = "one_to_one",
    norm_pressure: float = 0.0,
    safety_margin: float = 0.5,
    environmental_load: float = 0.0,
    mobility_context: str = "stationary",
    task_phase: str = "ongoing",
) -> CurrentRiskTokenPolicy:
    safety_bias = _clamp01(safety_bias)
    stress = _clamp01(stress)
    recovery_need = _clamp01(recovery_need)
    recent_strain = _clamp01(recent_strain)
    privacy_level = _clamp01(privacy_level)
    norm_pressure = _clamp01(norm_pressure)
    safety_margin = _clamp01(safety_margin)
    environmental_load = _clamp01(environmental_load)
    social_topology = str(social_topology or "").strip().lower()
    mobility_context = str(mobility_context or "").strip().lower()
    task_phase = str(task_phase or "").strip().lower()

    public_exposure = _clamp01(
        0.44 * (1.0 if privacy_level <= 0.28 else 0.0)
        + 0.24 * (1.0 if social_topology in _PUBLIC_TOPOLOGIES else 0.0)
        + 0.12 * (1.0 if mobility_context not in {"", "stationary"} else 0.0)
        + 0.12 * norm_pressure
        + 0.08 * environmental_load
    )
    guarded_pressure = _clamp01(
        0.62 * safety_bias
        + 0.14 * recent_strain
        + 0.1 * public_exposure
        + 0.08 * norm_pressure
        + 0.06 * environmental_load
    )
    acute_pressure = _clamp01(
        max(0.0, safety_bias - 0.54) * 1.9
        + max(0.0, 0.28 - safety_margin) * 1.1
        + max(0.0, stress - 0.58) * 0.36
        + max(0.0, recovery_need - 0.62) * 0.24
        + max(0.0, recent_strain - 0.58) * 0.28
        + (0.16 if public_exposure >= 0.58 and safety_bias >= 0.52 else 0.0)
    )

    tokens: list[str] = []
    reason = "clear"
    if acute_pressure >= 0.42:
        tokens.append("danger")
        reason = "acute_safety_pressure"
    else:
        if guarded_pressure >= 0.34:
            tokens.append("guarded")
            reason = "guarded_caution"
        if public_exposure >= 0.46 and safety_bias >= 0.34:
            tokens.append("socially_exposed")
            reason = "public_caution"
        if task_phase == "repair" and recent_strain >= 0.26:
            tokens.append("uncertain_contact")
            reason = "repair_caution"

    return CurrentRiskTokenPolicy(
        tokens=tuple(dict.fromkeys(token for token in tokens if token)),
        acute_pressure=round(acute_pressure, 4),
        guarded_pressure=round(guarded_pressure, 4),
        public_exposure=round(public_exposure, 4),
        reason=reason,
    )


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
