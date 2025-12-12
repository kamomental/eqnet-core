from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Mapping


def _coerce_float(payload: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not payload:
        return default
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class ValueGradient:
    """Represents which value axes dominated a decision."""

    survival_bias: float = 0.5
    physiological_bias: float = 0.5
    social_bias: float = 0.5
    exploration_bias: float = 0.5
    attachment_bias: float = 0.5

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ValueGradient":
        return cls(
            survival_bias=_coerce_float(payload, "survival_bias", 0.5),
            physiological_bias=_coerce_float(payload, "physiological_bias", 0.5),
            social_bias=_coerce_float(payload, "social_bias", 0.5),
            exploration_bias=_coerce_float(payload, "exploration_bias", 0.5),
            attachment_bias=_coerce_float(payload, "attachment_bias", 0.5),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "survival_bias": float(self.survival_bias),
            "physiological_bias": float(self.physiological_bias),
            "social_bias": float(self.social_bias),
            "exploration_bias": float(self.exploration_bias),
            "attachment_bias": float(self.attachment_bias),
        }

    def blend(self, other: "ValueGradient", ratio: float = 0.5) -> "ValueGradient":
        r = max(0.0, min(1.0, ratio))
        inv = 1.0 - r
        return ValueGradient(
            survival_bias=inv * self.survival_bias + r * other.survival_bias,
            physiological_bias=inv * self.physiological_bias + r * other.physiological_bias,
            social_bias=inv * self.social_bias + r * other.social_bias,
            exploration_bias=inv * self.exploration_bias + r * other.exploration_bias,
            attachment_bias=inv * self.attachment_bias + r * other.attachment_bias,
        )


@dataclass
class EmotionVector:
    """Lightweight affect snapshot with convenience helpers."""

    valence: float = 0.0
    arousal: float = 0.0
    love: float = 0.0
    stress: float = 0.0
    mask: float = 0.0
    heart_rate_norm: float = 0.0
    breath_ratio_norm: float = 0.0
    value_gradient: ValueGradient = field(default_factory=ValueGradient)

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any] | None) -> "EmotionVector":
        metrics = metrics or {}

        def _coerce(name: str) -> float:
            value = metrics.get(name)
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        vg_payload = None
        if isinstance(metrics, Mapping):
            vg_payload = metrics.get("value_gradient")

        return cls(
            valence=_coerce("valence"),
            arousal=_coerce("arousal"),
            love=_coerce("love"),
            stress=_coerce("stress"),
            mask=_coerce("mask"),
            heart_rate_norm=_coerce("heart_rate_norm"),
            breath_ratio_norm=_coerce("breath_ratio_norm"),
            value_gradient=ValueGradient.from_mapping(vg_payload),
        )

    def magnitude(self) -> float:
        """Return a scalar intensity for quick thresholding."""

        return sqrt(self.valence ** 2 + self.arousal ** 2 + self.stress ** 2)

    def salience_score(self) -> float:
        """Bias toward love/stress so positive ties also mark episodes."""

        return max(abs(self.valence), abs(self.arousal)) + max(self.love, self.stress)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "love": float(self.love),
            "stress": float(self.stress),
            "mask": float(self.mask),
            "heart_rate_norm": float(self.heart_rate_norm),
            "breath_ratio_norm": float(self.breath_ratio_norm),
        }
        payload["value_gradient"] = self.value_gradient.to_dict()
        return payload

    def blend(self, other: "EmotionVector", ratio: float = 0.5) -> "EmotionVector":
        """Return a simple linear blend with ``other``."""

        r = max(0.0, min(1.0, ratio))
        inv = 1.0 - r
        return EmotionVector(
            valence=inv * self.valence + r * other.valence,
            arousal=inv * self.arousal + r * other.arousal,
            love=inv * self.love + r * other.love,
            stress=inv * self.stress + r * other.stress,
            mask=inv * self.mask + r * other.mask,
            heart_rate_norm=inv * self.heart_rate_norm + r * other.heart_rate_norm,
            breath_ratio_norm=inv * self.breath_ratio_norm + r * other.breath_ratio_norm,
            value_gradient=self.value_gradient.blend(other.value_gradient, ratio),
        )
