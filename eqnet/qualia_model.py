"""Qualia field helpers: construction + replay forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .runtime.state import QualiaState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qualia construction helpers (v0: simple concatenation)
# ---------------------------------------------------------------------------

def build_qualia_vec_v0(
    emotion_metrics: Dict[str, float],
    culture_metrics: Dict[str, Any],
    awareness_stage: int,
    text_embedding: Iterable[float],
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
    """

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

    return np.concatenate([emo_vec, cult_vec, aware_vec, text_vec])


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

    q_vec = build_qualia_vec_v0(
        emotion_metrics=extract_emotion_features(moment_entry),
        culture_metrics=extract_culture_features(moment_entry),
        awareness_stage=awareness_stage,
        text_embedding=text_embedding,
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

