"""Qualia field helpers: construction + replay forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from emot_terrain_lab.terrain.emotion import AXES, AXIS_BOUNDS
try:
    from eqnet.hub.streaming_sensor import StreamingSensorState
except Exception:  # pragma: no cover - optional import
    StreamingSensorState = Any  # type: ignore

from .runtime.state import QualiaState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qualia construction helpers (v0: simple concatenation)
# ---------------------------------------------------------------------------

# Axis helpers for emotion blending
_AXIS_INDEX = {name: idx for idx, name in enumerate(AXES)}


def _empty_axis_vec() -> np.ndarray:
    return np.zeros(len(AXES), dtype=float)


def _axis_vec(values: Optional[Iterable[float]]) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=float)
    arr = _empty_axis_vec()
    data = np.asarray(list(values), dtype=float).reshape(-1)
    limit = min(arr.size, data.size)
    if limit > 0:
        arr[:limit] = data[:limit]
    return arr


def _clamp_axes(vec: np.ndarray) -> np.ndarray:
    if vec.size != len(AXES):
        return vec
    clamped = vec.copy()
    for name, idx in _AXIS_INDEX.items():
        lo, hi = AXIS_BOUNDS.get(name, (0.0, 1.0))
        clamped[idx] = float(np.clip(clamped[idx], lo, hi))
    return clamped


def sensor_to_emotion_axes(snapshot: Optional[StreamingSensorState]) -> np.ndarray:
    axes = _empty_axis_vec()
    if snapshot is None:
        return axes
    metrics = getattr(snapshot, 'metrics', {}) or {}
    activity = float(metrics.get('activity_level', 0.0))
    motion = float(metrics.get('heart_rate_motion', 0.0))
    emotion = float(metrics.get('heart_rate_emotion', 0.0))
    breath = float(metrics.get('breath_rate', 0.0))
    stress = float(metrics.get('body_stress_index', 0.0))
    balance = float(metrics.get('autonomic_balance', 0.5))
    privacy = float(metrics.get('body_flag_private', 0.0))
    overload = float(metrics.get('body_flag_overloaded', 0.0))
    voice = float(metrics.get('voice_level', 0.0))
    people = float(metrics.get('person_count', 0.0))
    motion_score = float(metrics.get('motion_score', 0.0))

    def _set(axis: str, value: float) -> None:
        lo, hi = AXIS_BOUNDS.get(axis, (0.0, 1.0))
        axes[_AXIS_INDEX[axis]] = float(np.clip(value, lo, hi))

    _set('sensory', 0.25 + 0.45 * np.clip(breath, 0.0, 1.0) + 0.2 * np.clip(stress, 0.0, 1.0))
    _set('temporal', 0.4 + 0.3 * (1.0 - np.clip(activity, 0.0, 1.5)) - 0.1 * abs(emotion))
    _set('spatial', 0.3 + 0.2 * np.clip(motion_score, 0.0, 1.0) + 0.1 * np.clip(people, 0.0, 1.0))
    axes[_AXIS_INDEX['affective']] = float(np.clip(np.tanh(0.5 * emotion - 0.25 * stress), -1.0, 1.0))
    _set('cognitive', 0.5 - 0.2 * overload + 0.1 * balance)
    axes[_AXIS_INDEX['social']] = float(np.clip(0.2 * people - 0.3 * privacy + 0.15 * voice, -1.0, 1.0))
    _set('meta', 0.3 + 0.4 * balance - 0.1 * activity)
    axes[_AXIS_INDEX['agency']] = float(np.clip(np.tanh(0.35 * emotion - 0.15 * motion + 0.25 * (0.5 - activity)), -1.0, 1.0))
    _set('recursion', 0.4 + 0.2 * (1.0 - overload) - 0.1 * stress)
    return _clamp_axes(axes)


def blend_emotion_axes(
    text_axes: Sequence[float],
    sensor_axes: Sequence[float],
    *,
    activity_level: float,
    privacy_level: float,
    fog_level: float,
) -> np.ndarray:
    text_vec = _axis_vec(text_axes)
    if text_vec.size == 0:
        text_vec = _empty_axis_vec()
    sensor_vec = _axis_vec(sensor_axes)
    if sensor_vec.size == 0:
        sensor_vec = _empty_axis_vec()
    activity = float(np.clip(activity_level, 0.0, 1.5))
    privacy = float(np.clip(privacy_level, 0.0, 1.0))
    fog = float(np.clip(fog_level, 0.0, 1.0))
    w_sensor = np.clip(0.2 + 0.6 * (1.0 - activity), 0.0, 0.95)
    w_sensor *= 0.85 + 0.15 * fog
    w_sensor *= 1.0 - 0.4 * privacy
    blended = (1.0 - w_sensor) * text_vec + w_sensor * sensor_vec
    return _clamp_axes(blended)

def build_qualia_vec_v0(
    emotion_metrics: Dict[str, float],
    culture_metrics: Dict[str, Any],
    awareness_stage: int,
    text_embedding: Iterable[float],
    emotion_axes: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Construct the v0 qualia vector by concatenating basic metrics.

    Parameters
    ----------
    emotion_metrics:
        Dict containing ``mask``, ``love``, ``stress``, ``heart_rate_norm``,
        ``breath_ratio_norm`` (missing keys default to 0).
    culture_metrics:
        Dict containing ``rho``, ``politeness``, ``intimacy`` and
        ``culture_tag_embed`` (iterable, defaults to empty).
    awareness_stage:
        Integer stage (0-3). It is normalised to [0,1].
    text_embedding:
        Pre-computed sentence embedding (already projected to a compact dim).
    emotion_axes:
        Optional blended axes vector (typically text+sensor). When provided it
        is concatenated ahead of the scalar metrics.
    """

    axes_vec = _axis_vec(emotion_axes)
    components = []
    if axes_vec.size:
        components.append(axes_vec)

    emo_vec = np.array([
        float(emotion_metrics.get("mask", 0.0)),
        float(emotion_metrics.get("love", 0.0)),
        float(emotion_metrics.get("stress", 0.0)),
        float(emotion_metrics.get("heart_rate_norm", 0.0)),
        float(emotion_metrics.get("breath_ratio_norm", 0.0)),
    ], dtype=float)

    tag_vec = np.asarray(culture_metrics.get("culture_tag_embed", []), dtype=float).reshape(-1)
    cult_vec = np.concatenate([
        np.array([
            float(culture_metrics.get("rho", 0.0)),
            float(culture_metrics.get("politeness", 0.0)),
            float(culture_metrics.get("intimacy", 0.0)),
        ], dtype=float),
        tag_vec,
    ])

    aware_vec = np.array([float(awareness_stage) / 3.0], dtype=float)
    text_vec = np.asarray(text_embedding, dtype=float).reshape(-1)

    components.extend([emo_vec, cult_vec, aware_vec, text_vec])
    return np.concatenate(components)


def extract_emotion_features(moment_entry: Any) -> Dict[str, float]:
    """Map a moment entry to emotion metrics (v0 placeholder)."""

    emo = getattr(moment_entry, "emotion", None)
    if emo is None:
        return {k: 0.0 for k in ("mask", "love", "stress", "heart_rate_norm", "breath_ratio_norm")}
    return {
        "mask": float(getattr(emo, "mask", 0.0)),
        "love": float(getattr(emo, "love", 0.0)),
        "stress": float(getattr(emo, "stress", 0.0)),
        "heart_rate_norm": float(getattr(emo, "heart_rate_norm", 0.0)),
        "breath_ratio_norm": float(getattr(emo, "breath_ratio_norm", 0.0)),
    }


def extract_culture_features(moment_entry: Any) -> Dict[str, Any]:
    """Map a moment entry to culture metrics (v0 placeholder)."""

    cult = getattr(moment_entry, "culture", None)
    if cult is None:
        return {
            "rho": 0.0,
            "politeness": 0.5,
            "intimacy": 0.0,
            "culture_tag_embed": [],
        }
    return {
        "rho": float(getattr(cult, "rho", 0.0)),
        "politeness": float(getattr(cult, "politeness", 0.5)),
        "intimacy": float(getattr(cult, "intimacy", 0.0)),
        "culture_tag_embed": list(getattr(cult, "culture_tag_embed", [])),
    }


def update_qualia_state(
    prev_state: Optional[QualiaState],
    moment_entry: Any,
    text_embedding: Iterable[float],
) -> QualiaState:
    """Build the next ``QualiaState`` from the latest moment entry.

    ``prev_state`` is reserved for future use (flux calculations, etc.).
    """

    ts = getattr(moment_entry, "timestamp")
    awareness_stage = int(getattr(moment_entry, "awareness_stage", 0))

    axes = getattr(moment_entry, 'emotion_axes_blended', None) or getattr(moment_entry, 'emotion_axes_sensor', None)
    q_vec = build_qualia_vec_v0(
        emotion_metrics=extract_emotion_features(moment_entry),
        culture_metrics=extract_culture_features(moment_entry),
        awareness_stage=awareness_stage,
        text_embedding=text_embedding,
        emotion_axes=axes,
    )

    return QualiaState(
        timestamp=ts,
        qualia_vec=q_vec,
        membrane_state={},
        flux={},
        narrative_ref=None,
    )


# ---------------------------------------------------------------------------
# Future replay (predictive vs imagery)
# ---------------------------------------------------------------------------


class ReplayMode(str, Enum):
    """Modes for future replay simulation."""

    PREDICTIVE = "predictive"
    IMAGERY = "imagery"


@dataclass
class FutureReplayConfig:
    """Configuration for :func:`simulate_future`."""

    steps: int = 5
    noise_scale: float = 0.3
    window: int = 10


def _estimate_delta_from_history(
    history: List[QualiaState],
    window: int,
) -> np.ndarray:
    """Return the averaged Deltaqualia_vec across the trailing window."""

    if len(history) < window + 1:
        logger.info(
            "simulate_future: history too short (len=%d < %d); using zero delta.",
            len(history),
            window + 1,
        )
        return np.zeros_like(history[-1].qualia_vec)
    tail = history[-(window + 1) :]
    deltas = [
        tail[i + 1].qualia_vec - tail[i].qualia_vec
        for i in range(len(tail) - 1)
    ]
    return np.mean(deltas, axis=0)


def simulate_future(
    history: List[QualiaState],
    mode: ReplayMode,
    cfg: FutureReplayConfig,
    intention_vec: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Sample one future qualia trajectory.

    Notes
    -----
    - 現状は ``np.random`` による非決定的動作。再現性が必要になったら
      ``rng`` に PRNG を渡す形で seed 対応を拡張する。
    """

    if not history:
        raise ValueError("history must not be empty")

    rng = rng or np.random.default_rng()
    delta = _estimate_delta_from_history(history, cfg.window)
    last = history[-1]

    if mode is ReplayMode.IMAGERY and intention_vec is not None:
        intention_vec = np.asarray(intention_vec, dtype=float).reshape(-1)
        if intention_vec.shape != last.qualia_vec.shape:
            raise ValueError(
                "intention_vec shape %s != qualia_vec shape %s"
                % (intention_vec.shape, last.qualia_vec.shape)
            )
        start_vec = last.qualia_vec + intention_vec
    else:
        start_vec = last.qualia_vec.copy()

    traj = [start_vec]
    fog = float(last.membrane_state.get("fog_level", 0.5))

    for _ in range(cfg.steps):
        noise = rng.normal(
            scale=cfg.noise_scale * (0.5 + fog), size=delta.shape
        )
        next_vec = traj[-1] + delta + noise
        traj.append(next_vec)

    return traj

def compute_future_risk(
    history: Sequence[QualiaState],
    cfg: FutureReplayConfig,
    *,
    stress_index: int | None = None,
    stress_threshold: float = 0.6,
    body_index: int | None = None,
    body_threshold: float = 0.4,
) -> float:
    """Return the fraction of predicted steps that exceed risk thresholds.

    Parameters
    ----------
    history:
        Qualia history used to seed :func:`simulate_future`.
    cfg:
        Configuration for the predictive rollout.
    stress_index:
        Index in the qualia vector representing stress/tension (>= threshold -> risky).
    body_index:
        Index representing body.R or similar success proxy (<= threshold -> risky).
    """

    if len(history) < 2:
        return 0.0

    vectors = simulate_future(list(history), ReplayMode.PREDICTIVE, cfg)
    if len(vectors) <= 1:
        return 0.0

    risky = 0
    total = max(1, len(vectors) - 1)
    for vec in vectors[1:]:
        triggered = False
        if stress_index is not None and stress_index < vec.shape[0]:
            triggered = triggered or (float(vec[stress_index]) >= stress_threshold)
        if body_index is not None and body_index < vec.shape[0]:
            triggered = triggered or (float(vec[body_index]) <= body_threshold)
        if triggered:
            risky += 1
    return float(risky / total)

def compute_future_hopefulness(
    history: Sequence[QualiaState],
    cfg: FutureReplayConfig,
    *,
    intention_vec: Optional[np.ndarray],
    valence_index: int = 0,
    love_index: int = 1,
    scale: float = 2.0,
) -> float:
    """Return a scalar hopefulness estimate from an imagery rollout.

    ``history`` should contain low-dimensional qualia states (e.g., [valence, love])
    and ``intention_vec`` nudges the starting point toward the desired future.
    """

    if len(history) < 1 or intention_vec is None:
        return 0.0
    last = history[-1]
    qualia_dim = last.qualia_vec.shape[0]
    intention_vec = np.asarray(intention_vec, dtype=float).reshape(-1)
    if intention_vec.size != qualia_dim:
        raise ValueError(
            "intention_vec shape %s != qualia_vec shape %s"
            % (intention_vec.shape, last.qualia_vec.shape)
        )
    traj = simulate_future(list(history), ReplayMode.IMAGERY, cfg, intention_vec=intention_vec)
    if len(traj) <= 1:
        return 0.0
    val_seq: list[float] = []
    love_seq: list[float] = []
    for vec in traj:
        if valence_index < vec.shape[0]:
            val_seq.append(float(vec[valence_index]))
        if love_index < vec.shape[0]:
            love_seq.append(float(vec[love_index]))
    if not val_seq or not love_seq:
        return 0.0
    base_v = val_seq[0]
    base_l = love_seq[0]
    dv = float(np.mean(val_seq) - base_v)
    dl = float(np.mean(love_seq) - base_l)
    raw = max(0.0, dv + dl)
    return float(np.tanh(raw * scale))

