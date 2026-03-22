from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ForgettingSnapshot:
    forgetting_pressure: float = 0.0
    replay_horizon: int = 2
    persona_halflife_tau: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "forgetting_pressure": round(self.forgetting_pressure, 4),
            "replay_horizon": max(1, int(self.replay_horizon)),
            "persona_halflife_tau": round(self.persona_halflife_tau, 4),
        }


@dataclass(frozen=True)
class _Advice:
    lstm: dict[str, Any]
    ssm: dict[str, Any]
    intero: dict[str, Any]
    replay: dict[str, Any]
    persona: dict[str, Any]


class _HeuristicForgettingController:
    def advise(
        self,
        *,
        tau_rate: float,
        inflammation: float,
        uncertainty: float,
        novelty: float,
        cfl: float,
    ) -> _Advice:
        del tau_rate, cfl
        forget_bias_delta = _clamp01(inflammation * 0.38 + uncertainty * 0.34 + novelty * 0.18)
        replay_horizon = 3
        if uncertainty >= 0.42 or inflammation >= 0.52:
            replay_horizon = 2
        if uncertainty >= 0.72 or inflammation >= 0.78:
            replay_horizon = 1
        halflife_tau = max(8.0, 24.0 - inflammation * 8.0 - uncertainty * 5.0 - novelty * 3.0)
        return _Advice(
            lstm={"forget_bias_delta": round(forget_bias_delta, 4)},
            ssm={},
            intero={},
            replay={"horizon": replay_horizon},
            persona={"halflife_tau": round(halflife_tau, 4)},
        )


class ForgettingCore:
    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def snapshot(
        self,
        *,
        stress: float,
        recovery_need: float,
        terrain_transition_roughness: float,
        transition_intensity: float,
        recent_strain: float,
    ) -> ForgettingSnapshot:
        controller = self._controller or self._build_default_controller()
        advice = controller.advise(
            tau_rate=1.0 + _clamp01(recent_strain) * 0.35,
            inflammation=_clamp01(max(stress, recovery_need)),
            uncertainty=_clamp01(max(terrain_transition_roughness, transition_intensity)),
            novelty=_clamp01(transition_intensity),
            cfl=0.5 + _clamp01(stress) * 0.2,
        )
        replay_horizon = max(1, int((getattr(advice, "replay", {}) or {}).get("horizon", 2) or 2))
        forget_bias_delta = max(0.0, _float_from(getattr(advice, "lstm", {}), "forget_bias_delta", default=0.0))
        halflife_tau = max(6.0, _float_from(getattr(advice, "persona", {}), "halflife_tau", default=24.0))
        forgetting_pressure = _clamp01(
            forget_bias_delta * 0.9
            + max(0.0, (3.0 - replay_horizon) / 3.0) * 0.35
            + max(0.0, (24.0 - halflife_tau) / 24.0) * 0.18
        )
        return ForgettingSnapshot(
            forgetting_pressure=forgetting_pressure,
            replay_horizon=replay_horizon,
            persona_halflife_tau=halflife_tau,
        )

    @staticmethod
    def _build_default_controller() -> Any:
        return _HeuristicForgettingController()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _float_from(mapping: Optional[Mapping[str, Any]], key: str, *, default: float = 0.0) -> float:
    if not mapping:
        return default
    try:
        return float(mapping.get(key, default) or 0.0)
    except (TypeError, ValueError, AttributeError):
        return default
