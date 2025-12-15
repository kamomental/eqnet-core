from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from emot_terrain_lab.hub.perception import AffectSample
from emot_terrain_lab.mind.inner_replay import ReplayStats


@dataclass(frozen=True)
class ShadowEstimatorConfig:
    alpha_min: float = 0.05
    alpha_max: float = 0.85
    prior_floor: float = 0.05
    entropy_weight: float = 0.6
    alignment_weight: float = 0.3
    story_weight: float = 0.3
    residual_weight: float = 0.6
    entropy_to_uncertainty: float = 0.3
    obs_uncertainty_weight: float = 0.4


@dataclass(frozen=True)
class ShadowEstimate:
    completed_valence: float
    completed_arousal: float
    pred_valence: float
    pred_arousal: float
    alpha: float
    mood_uncertainty: float
    residual: float
    evidence: dict[str, float]


class ShadowEstimator:
    def __init__(self, config: ShadowEstimatorConfig | None = None) -> None:
        self.cfg = config or ShadowEstimatorConfig()

    def estimate(
        self,
        affect: AffectSample,
        *,
        prospective: Mapping[str, Any] | None = None,
        replay_stats: ReplayStats | None = None,
        sensor_confidence: Optional[float] = None,
        culture_bias: Optional[float] = None,
    ) -> ShadowEstimate:
        story_energy = float(prospective.get("E_story", 0.0)) if prospective else 0.0
        pdc_temp = float(prospective.get("T", 0.65)) if prospective else 0.65
        alignment = replay_stats.alignment if replay_stats else 0.5
        entropy = replay_stats.entropy_proxy if replay_stats else 0.5
        top_signal = replay_stats.top_signal if replay_stats else 0.0

        shadow_valence = story_energy
        shadow_valence += 0.5 * (alignment * 2.0 - 1.0)
        if culture_bias is not None:
            shadow_valence += 0.2 * culture_bias
        shadow_valence = _clamp(shadow_valence, -1.0, 1.0)

        shadow_arousal = 1.0 - _clamp(pdc_temp, 0.0, 1.5) / 1.5
        shadow_arousal = 0.6 * shadow_arousal + 0.4 * _clamp(top_signal, 0.0, 1.0)
        shadow_arousal = _clamp(shadow_arousal, 0.0, 1.0)

        sensor_conf = sensor_confidence
        if sensor_conf is None:
            sensor_conf = getattr(affect, "confidence", None)
        if sensor_conf is None:
            sensor_conf = 0.5
        sensor_conf = _clamp(sensor_conf, 0.0, 1.0)

        obs_sigma = max(1e-3, 1.0 - sensor_conf)
        prior_sigma = self.cfg.prior_floor
        prior_sigma += self.cfg.entropy_weight * entropy
        prior_sigma += self.cfg.alignment_weight * (1.0 - alignment)
        prior_sigma += self.cfg.story_weight * (1.0 - abs(story_energy))
        prior_sigma = max(1e-3, prior_sigma)

        alpha = obs_sigma / (obs_sigma + prior_sigma)
        alpha = _clamp(alpha, self.cfg.alpha_min, self.cfg.alpha_max)

        pred_valence = shadow_valence
        pred_arousal = shadow_arousal

        completed_valence = (1.0 - alpha) * affect.valence + alpha * pred_valence
        completed_arousal = (1.0 - alpha) * affect.arousal + alpha * pred_arousal
        completed_valence = _clamp(completed_valence, -1.0, 1.0)
        completed_arousal = _clamp(completed_arousal, 0.0, 1.0)

        residual = math.sqrt(
            (affect.valence - pred_valence) ** 2 + (affect.arousal - pred_arousal) ** 2
        )
        norm_residual = _clamp(residual / math.sqrt(2.0), 0.0, 1.0)

        mood_uncertainty = self.cfg.residual_weight * norm_residual
        mood_uncertainty += self.cfg.entropy_to_uncertainty * entropy
        mood_uncertainty += self.cfg.obs_uncertainty_weight * (1.0 - sensor_conf)
        mood_uncertainty = _clamp(mood_uncertainty, 0.0, 1.0)

        evidence = {
            "story": story_energy,
            "alignment": alignment,
            "entropy": entropy,
            "sensor_confidence": sensor_conf,
            "alpha": alpha,
        }

        return ShadowEstimate(
            completed_valence=completed_valence,
            completed_arousal=completed_arousal,
            pred_valence=pred_valence,
            pred_arousal=pred_arousal,
            alpha=alpha,
            mood_uncertainty=mood_uncertainty,
            residual=norm_residual,
            evidence=evidence,
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


__all__ = ["ShadowEstimator", "ShadowEstimatorConfig", "ShadowEstimate"]
