from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EpistemicState:
    """知識の鮮度と再検証圧をまとめる小さな状態契約。"""

    freshness: float = 0.5
    source_confidence: float = 0.5
    verification_pressure: float = 0.35
    change_likelihood: float = 0.3
    stale_risk: float = 0.35
    epistemic_caution: float = 0.4
    dominant_posture: str = "carry_forward"

    def to_dict(self) -> dict[str, Any]:
        return {
            "freshness": round(self.freshness, 4),
            "source_confidence": round(self.source_confidence, 4),
            "verification_pressure": round(self.verification_pressure, 4),
            "change_likelihood": round(self.change_likelihood, 4),
            "stale_risk": round(self.stale_risk, 4),
            "epistemic_caution": round(self.epistemic_caution, 4),
            "dominant_posture": self.dominant_posture,
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "EpistemicState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_epistemic_state(previous)
        current_axes = _packet_axis_values(self)
        previous_axes = _packet_axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def coerce_epistemic_state(value: Mapping[str, Any] | EpistemicState | None) -> EpistemicState:
    if isinstance(value, EpistemicState):
        return value
    payload = dict(value or {})
    return EpistemicState(
        freshness=_float01(payload.get("freshness"), 0.5),
        source_confidence=_float01(payload.get("source_confidence"), 0.5),
        verification_pressure=_float01(payload.get("verification_pressure"), 0.35),
        change_likelihood=_float01(payload.get("change_likelihood"), 0.3),
        stale_risk=_float01(payload.get("stale_risk"), 0.35),
        epistemic_caution=_float01(payload.get("epistemic_caution"), 0.4),
        dominant_posture=str(payload.get("dominant_posture") or "carry_forward").strip() or "carry_forward",
    )


def _packet_axis_values(state: EpistemicState) -> dict[str, float]:
    return {
        "grounding": _clamp01(
            state.freshness * 0.52
            + state.source_confidence * 0.4
            + (1.0 - state.epistemic_caution) * 0.08
        ),
        "volatility": _clamp01(
            state.change_likelihood * 0.58
            + state.stale_risk * 0.42
        ),
        "verification": _clamp01(
            state.verification_pressure * 0.62
            + state.epistemic_caution * 0.38
        ),
    }


def _float01(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
