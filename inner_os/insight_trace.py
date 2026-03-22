from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class InsightTrace:
    insight_class: str
    association_link_key: str
    linked_seed_ids: tuple[str, ...] = ()
    linked_seed_keys: tuple[str, ...] = ()
    insight_score: float = 0.0
    novelty: float = 0.0
    coherence_gain: float = 0.0
    prediction_drop: float = 0.0
    reframed_topic: str = ""
    followup_bias: float = 0.0
    confidence: float = 0.0
    anchor_center: tuple[float, ...] = ()
    anchor_dispersion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_class": self.insight_class,
            "association_link_key": self.association_link_key,
            "linked_seed_ids": list(self.linked_seed_ids),
            "linked_seed_keys": list(self.linked_seed_keys),
            "insight_score": float(self.insight_score),
            "novelty": float(self.novelty),
            "coherence_gain": float(self.coherence_gain),
            "prediction_drop": float(self.prediction_drop),
            "reframed_topic": self.reframed_topic,
            "followup_bias": float(self.followup_bias),
            "confidence": float(self.confidence),
            "anchor_center": list(self.anchor_center),
            "anchor_dispersion": float(self.anchor_dispersion),
        }


def derive_insight_trace(
    *,
    insight_event: Mapping[str, Any] | None,
    qualia_planner_view: Mapping[str, Any] | None = None,
    protection_mode: Mapping[str, Any] | None = None,
    affective_position: Mapping[str, Any] | None = None,
    terrain_readout: Mapping[str, Any] | None = None,
) -> Optional[InsightTrace]:
    event = dict(insight_event or {})
    if not bool(event.get("triggered", False)):
        return None
    score = dict(event.get("score") or {})
    linked_seed_keys = tuple(str(item) for item in event.get("connected_seed_keys") or [] if str(item).strip())
    linked_seed_ids = tuple(str(item) for item in event.get("connected_seed_ids") or [] if str(item).strip())
    dominant_seed_label = str(event.get("dominant_seed_label") or "").strip()
    summary = str(event.get("summary") or "").strip()
    reframed_topic = dominant_seed_label or summary
    insight_class = _derive_insight_class(
        linked_seed_keys=linked_seed_keys,
        novelty_gain=_clamp01(score.get("novelty_gain")),
        source_diversity=_clamp01(score.get("source_diversity")),
    )
    trust = _clamp01((qualia_planner_view or {}).get("trust"))
    protection_strength = _clamp01((protection_mode or {}).get("strength"))
    orient_bias = _clamp01(event.get("orient_bias"))
    stabilizing_bias = _clamp01(event.get("stabilizing_bias"))
    anchor_center = _coerce_anchor_center((affective_position or {}).get("z_aff"))
    anchor_dispersion = _derive_anchor_dispersion(
        cov_value=(affective_position or {}).get("cov"),
        position_confidence=_clamp01((affective_position or {}).get("confidence")),
        terrain_readout=terrain_readout,
    )
    confidence = _clamp01(
        _clamp01(score.get("total")) * 0.62
        + trust * 0.26
        + (1.0 - protection_strength) * 0.12
    )
    return InsightTrace(
        insight_class=insight_class,
        association_link_key=str(event.get("link_key") or "").strip(),
        linked_seed_ids=linked_seed_ids,
        linked_seed_keys=linked_seed_keys,
        insight_score=_clamp01(score.get("total")),
        novelty=_clamp01(score.get("novelty_gain")),
        coherence_gain=_clamp01(
            _clamp01(score.get("link_weight")) * 0.58
            + _clamp01(score.get("source_diversity")) * 0.42
        ),
        prediction_drop=_clamp01(score.get("tension_relief")),
        reframed_topic=reframed_topic[:160],
        followup_bias=_clamp01(orient_bias * 0.62 + stabilizing_bias * 0.38),
        confidence=confidence,
        anchor_center=anchor_center,
        anchor_dispersion=anchor_dispersion,
    )


def _derive_insight_class(
    *,
    linked_seed_keys: Sequence[str],
    novelty_gain: float,
    source_diversity: float,
) -> str:
    keys = tuple(str(item).strip() for item in linked_seed_keys if str(item).strip())
    has_bond = any(item.startswith("bond:") for item in keys)
    has_memory = any(item.startswith("memory:") for item in keys)
    has_external = any(item.startswith("external:") for item in keys)
    if has_bond and (has_memory or has_external):
        return "reframed_relation"
    if novelty_gain >= 0.34 and source_diversity >= 0.2:
        return "new_link_hypothesis"
    return "insight_trace"


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _coerce_anchor_center(value: Any) -> tuple[float, ...]:
    vector = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return ()
    return tuple(float(item) for item in vector.tolist())


def _derive_anchor_dispersion(
    *,
    cov_value: Any,
    position_confidence: float,
    terrain_readout: Mapping[str, Any] | None,
) -> float:
    dispersion = 0.0
    cov = np.asarray(cov_value if cov_value is not None else [], dtype=np.float32)
    if cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] > 0:
        diagonal = np.diag(cov)
        if diagonal.size > 0:
            dispersion = float(np.sqrt(max(0.0, float(np.mean(diagonal)))))
    if dispersion <= 0.0:
        dispersion = 0.16 + (1.0 - position_confidence) * 0.28
    protect_bias = _clamp01((terrain_readout or {}).get("protect_bias"))
    avoid_bias = _clamp01((terrain_readout or {}).get("avoid_bias"))
    dispersion += protect_bias * 0.04 + avoid_bias * 0.03
    return max(0.05, float(dispersion))
