from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float32]


@dataclass(frozen=True)
class FieldNormalizationConfig:
    global_range: float = 1.0
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    cv_low: float = 0.25
    cv_high: float = 1.0
    fog_density: float = 0.0
    eps: float = 1.0e-6


@dataclass(frozen=True)
class FieldNormalizationStats:
    local_range: float = 0.0
    global_range: float = 1.0
    coefficient_variation: float = 0.0
    range_trust: float = 0.0
    effective_range: float = 1.0
    fog_density: float = 0.0
    gradient_confidence: float = 0.0
    lower_anchor: float = 0.0
    upper_anchor: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_range": round(self.local_range, 6),
            "global_range": round(self.global_range, 6),
            "coefficient_variation": round(self.coefficient_variation, 6),
            "range_trust": round(self.range_trust, 6),
            "effective_range": round(self.effective_range, 6),
            "fog_density": round(self.fog_density, 6),
            "gradient_confidence": round(self.gradient_confidence, 6),
            "lower_anchor": round(self.lower_anchor, 6),
            "upper_anchor": round(self.upper_anchor, 6),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FieldNormalizationResult:
    values: Vector
    stats: FieldNormalizationStats


def normalize_field_values(
    values: Sequence[float] | Vector,
    config: FieldNormalizationConfig | None = None,
) -> FieldNormalizationResult:
    cfg = config or FieldNormalizationConfig()
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return FieldNormalizationResult(
            values=vector.astype(np.float32),
            stats=FieldNormalizationStats(
                global_range=_positive(cfg.global_range, 1.0),
                effective_range=_positive(cfg.global_range, 1.0),
                fog_density=_clamp01(cfg.fog_density),
                reasons=("empty_field",),
            ),
        )

    transformed = np.log1p(np.maximum(vector, 0.0)).astype(np.float32)
    stats = derive_field_normalization_stats(transformed, config=cfg)
    normalized = (transformed - stats.lower_anchor) / max(stats.effective_range, cfg.eps)
    normalized = np.clip(normalized, 0.0, 1.0)
    return FieldNormalizationResult(values=normalized.astype(np.float32), stats=stats)


def derive_field_normalization_stats(
    transformed_values: Sequence[float] | Vector,
    config: FieldNormalizationConfig | None = None,
) -> FieldNormalizationStats:
    cfg = config or FieldNormalizationConfig()
    vector = np.asarray(transformed_values, dtype=np.float32).reshape(-1)
    global_range = _positive(cfg.global_range, 1.0)
    fog_density = _clamp01(cfg.fog_density)
    reasons: list[str] = []

    if vector.size == 0:
        return FieldNormalizationStats(
            global_range=global_range,
            effective_range=global_range,
            fog_density=fog_density,
            reasons=("empty_field",),
        )

    lower_q = _clamp01(cfg.lower_quantile)
    upper_q = _clamp01(cfg.upper_quantile)
    if upper_q <= lower_q:
        lower_q, upper_q = 0.1, 0.9
        reasons.append("invalid_quantile_reset")

    lower_anchor = float(np.quantile(vector, lower_q))
    upper_anchor = float(np.quantile(vector, upper_q))
    local_range = max(0.0, upper_anchor - lower_anchor)

    cv = _coefficient_variation(vector, eps=cfg.eps)
    range_trust = 1.0 - _smoothstep(cfg.cv_low, cfg.cv_high, cv)
    range_trust *= 1.0 - fog_density

    if local_range <= cfg.eps:
        range_trust = 0.0
        reasons.append("flat_local_range")
    if cv >= cfg.cv_high:
        reasons.append("noisy_local_range")
    if fog_density > 0.0:
        reasons.append("fog_reduced_trust")

    effective_range = (
        range_trust * max(local_range, cfg.eps)
        + (1.0 - range_trust) * global_range
    )
    gradient_confidence = _clamp01(range_trust * (1.0 - fog_density))

    return FieldNormalizationStats(
        local_range=float(local_range),
        global_range=float(global_range),
        coefficient_variation=float(cv),
        range_trust=float(_clamp01(range_trust)),
        effective_range=float(max(effective_range, cfg.eps)),
        fog_density=float(fog_density),
        gradient_confidence=float(gradient_confidence),
        lower_anchor=float(lower_anchor),
        upper_anchor=float(upper_anchor),
        reasons=tuple(reasons),
    )


def _coefficient_variation(vector: Vector, *, eps: float) -> float:
    if vector.size <= 2:
        return 0.0
    sorted_values = np.sort(vector.astype(np.float32))
    spacing = np.diff(sorted_values).astype(np.float32)
    mean_spacing = float(np.mean(np.abs(spacing)))
    if mean_spacing <= eps:
        return 0.0
    return max(0.0, float(np.std(spacing)) / mean_spacing)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = _clamp01((float(value) - float(edge0)) / (float(edge1) - float(edge0)))
    return x * x * (3.0 - 2.0 * x)


def _positive(value: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(float(numeric), 1.0e-6)


def _clamp01(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
