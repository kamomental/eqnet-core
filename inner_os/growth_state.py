from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GrowthState:
    """共生的な長期成長を表す slow-state。"""

    relational_trust: float = 0.45
    epistemic_maturity: float = 0.35
    expressive_range: float = 0.4
    residue_integration: float = 0.35
    playfulness_range: float = 0.3
    self_coherence: float = 0.45
    social_update_strength: float = 1.0
    identity_update_strength: float = 1.0
    dominant_transition: str = "steady"

    def to_dict(self) -> dict[str, Any]:
        return {
            "relational_trust": round(self.relational_trust, 4),
            "epistemic_maturity": round(self.epistemic_maturity, 4),
            "expressive_range": round(self.expressive_range, 4),
            "residue_integration": round(self.residue_integration, 4),
            "playfulness_range": round(self.playfulness_range, 4),
            "self_coherence": round(self.self_coherence, 4),
            "social_update_strength": round(self.social_update_strength, 4),
            "identity_update_strength": round(self.identity_update_strength, 4),
            "dominant_transition": self.dominant_transition,
        }

    def to_replay_axes(
        self,
        previous: Mapping[str, Any] | "GrowthState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_growth_state(previous)
        current_axes = _replay_axis_values(self)
        previous_axes = _replay_axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def coerce_growth_state(value: Mapping[str, Any] | GrowthState | None) -> GrowthState:
    if isinstance(value, GrowthState):
        return value
    payload = dict(value or {})
    return GrowthState(
        relational_trust=_float01(payload.get("relational_trust"), 0.45),
        epistemic_maturity=_float01(payload.get("epistemic_maturity"), 0.35),
        expressive_range=_float01(payload.get("expressive_range"), 0.4),
        residue_integration=_float01(payload.get("residue_integration"), 0.35),
        playfulness_range=_float01(payload.get("playfulness_range"), 0.3),
        self_coherence=_float01(payload.get("self_coherence"), 0.45),
        social_update_strength=_float01(payload.get("social_update_strength"), 1.0),
        identity_update_strength=_float01(payload.get("identity_update_strength"), 1.0),
        dominant_transition=str(payload.get("dominant_transition") or "steady").strip() or "steady",
    )


def _replay_axis_values(state: GrowthState) -> dict[str, float]:
    return {
        "bond": _clamp01(
            state.relational_trust * 0.62
            + state.residue_integration * 0.24
            + state.playfulness_range * 0.14
        ),
        "stability": _clamp01(
            state.self_coherence * 0.58
            + state.epistemic_maturity * 0.28
            + state.residue_integration * 0.14
        ),
        "curiosity": _clamp01(
            state.expressive_range * 0.48
            + state.playfulness_range * 0.36
            + state.epistemic_maturity * 0.16
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
