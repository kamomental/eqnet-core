from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _to_array(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


@dataclass(slots=True)
class StreamingSensorState:
    """Aggregated sensor snapshot ready for Hub / Nightly consumption."""

    fused_vec: np.ndarray
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "StreamingSensorState":
        """Build ``StreamingSensorState`` from a raw sensor dictionary."""

        pose_vec = _to_array(raw.get("pose_vec"))
        voice_level = float(raw.get("voice_level", 0.0))
        breath_rate = float(raw.get("breath_rate", 0.0))
        inner_emotion_score = float(raw.get("inner_emotion_score", 0.0))

        activity_level = compute_activity_level(
            pose_vec=pose_vec,
            voice_level=voice_level,
            breath_rate=breath_rate,
        )

        heart_rate_raw = raw.get("heart_rate_raw")
        heart_rate_baseline = raw.get("heart_rate_baseline")
        heart_rate_motion = 0.0
        heart_rate_emotion = 0.0
        delta_hr: Optional[float] = None

        if heart_rate_raw is not None and heart_rate_baseline is not None:
            delta_hr = max(0.0, float(heart_rate_raw) - float(heart_rate_baseline))
            heart_rate_motion, heart_rate_emotion = decompose_heart_rate(
                delta_hr=delta_hr,
                activity_level=activity_level,
                inner_emotion_score=inner_emotion_score,
            )

        body_metrics = compute_body_channel_metrics(
            raw,
            heart_rate_motion,
            heart_rate_emotion,
        )
        body_state_flag = classify_body_state(raw, body_metrics)

        metrics: Dict[str, Any] = {
            "activity_level": activity_level,
            "voice_level": voice_level,
            "breath_rate": breath_rate,
            "heart_rate_raw": heart_rate_raw,
            "heart_rate_baseline": heart_rate_baseline,
            "delta_hr": delta_hr,
            "heart_rate_motion": heart_rate_motion,
            "heart_rate_emotion": heart_rate_emotion,
            **body_metrics,
        }
        metrics["body_state_flag"] = body_state_flag
        metrics["body_flag_private"] = 1.0 if body_state_flag == "private_high_arousal" else 0.0
        metrics["body_flag_overloaded"] = 1.0 if body_state_flag == "overloaded" else 0.0

        fused_vec = fuse_streams_to_vec(raw, metrics)
        return cls(fused_vec=fused_vec, metrics=metrics)


def compute_activity_level(
    *,
    pose_vec: Optional[np.ndarray],
    voice_level: float,
    breath_rate: float,
) -> float:
    """Rough scalar summarising how much the user is moving / exerting."""

    pose_component = 0.0
    if pose_vec is not None and pose_vec.size:
        pose_component = float(np.linalg.norm(pose_vec))

    score = 0.5 * pose_component + 0.3 * voice_level + 0.2 * breath_rate
    return clamp(score, 0.0, 1.5)


def decompose_heart_rate(
    delta_hr: float,
    activity_level: float,
    inner_emotion_score: float,
    *,
    a_low: float = 0.2,
    a_high: float = 0.8,
    hr_threshold: float = 1.5,
) -> Tuple[float, float]:
    """Split ``delta_hr`` into motion vs. emotion contributions."""

    if delta_hr <= hr_threshold:
        return 0.0, 0.0

    if activity_level >= a_high:
        base_alpha = 0.0
    elif activity_level <= a_low:
        base_alpha = 1.0
    else:
        base_alpha = (a_high - activity_level) / max(1e-6, (a_high - a_low))

    alpha = clamp(base_alpha + 0.3 * inner_emotion_score)
    hr_motion = delta_hr * (1.0 - alpha)
    hr_emotion = delta_hr * alpha
    return hr_motion, hr_emotion


def compute_body_channel_metrics(
    raw: Dict[str, Any],
    heart_rate_motion: float,
    heart_rate_emotion: float,
) -> Dict[str, Any]:
    """Estimate body stress / autonomic balance metrics."""

    stress_hint = raw.get("body_stress_index")
    balance_hint = raw.get("autonomic_balance")

    if stress_hint is None:
        stress_hint = clamp(0.6 * heart_rate_motion + 0.4 * heart_rate_emotion, 0.0, 1.5)
    if balance_hint is None:
        balance_hint = clamp(0.5 + 0.3 * np.tanh(heart_rate_motion - heart_rate_emotion))

    return {
        "body_stress_index": float(stress_hint),
        "autonomic_balance": float(balance_hint),
    }


def classify_body_state(raw: Dict[str, Any], body_metrics: Dict[str, Any]) -> str:
    """Heuristic privacy / overload classifier."""

    place_hint = str(raw.get("place_id", "") or "").lower()
    privacy_tags = raw.get("privacy_tags") or []
    privacy_tags = [str(tag).lower() for tag in privacy_tags]

    if "private" in privacy_tags or place_hint in {"bedroom", "bath", "changing_room"}:
        return "private_high_arousal"

    stress = float(body_metrics.get("body_stress_index", 0.0))
    if stress > 1.0:
        return "overloaded"
    return "normal"


def fuse_streams_to_vec(raw: Dict[str, Any], metrics: Dict[str, Any]) -> np.ndarray:
    """Build a fused observation vector for downstream models."""

    features: list[float] = []
    for key in (
        "activity_level",
        "voice_level",
        "breath_rate",
        "heart_rate_motion",
        "heart_rate_emotion",
        "body_stress_index",
        "autonomic_balance",
    ):
        val = metrics.get(key)
        features.append(0.0 if val is None else float(val))

    features.append(float(bool(raw.get("has_face", True))))
    features.append(float(raw.get("inner_emotion_score", 0.0)))

    extra = raw.get("vlm_features")
    if extra is not None:
        arr = _to_array(extra)
        if arr is not None:
            features.extend(float(v) for v in arr.flatten())

    if not features:
        features.append(0.0)

    return np.asarray(features, dtype=float)

