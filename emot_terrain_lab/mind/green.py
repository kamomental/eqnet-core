# -*- coding: utf-8 -*-
"""Green-function response mapping qualia to affect deltas and control tweaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

AFFECT_AXES: Tuple[str, ...] = ("v", "a", "d", "n", "c", "e", "s")
QUALIA_AXES: Tuple[str, ...] = ("sensation", "meaning")

DEFAULT_KERNEL = np.array(
    [
        [0.25, 0.55],  # valence (v)
        [0.35, 0.10],  # arousal (a)
        [0.15, 0.45],  # dominance (d)
        [0.30, 0.25],  # novelty (n)
        [-0.40, -0.30],  # certainty (c)
        [-0.25, 0.05],  # effort (e)
        [0.20, 0.50],  # social (s)
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GreenResponse:
    delta_mood: Dict[str, float]
    controls: Dict[str, float]
    qualia_vector: Dict[str, float]


def green_response(
    qualia: Mapping[str, Any],
    *,
    culture_resonance: float = 0.3,
    culture_kernel: Optional[Mapping[str, float]] = None,
    memory_trace: Optional[Mapping[str, float]] = None,
    affect_state: Optional[Mapping[str, float]] = None,
    kernel: Optional[Sequence[Sequence[float]]] = None,
    tau_rate: float = 1.0,
) -> GreenResponse:
    """Map qualia + culture/memory context into affect deltas and control tweaks."""
    q_vec, q_dict = _qualia_vector(qualia)
    kernel_mat = np.asarray(
        kernel if kernel is not None else DEFAULT_KERNEL, dtype=np.float32
    )
    delta = kernel_mat @ q_vec  # shape (7,)

    if culture_kernel:
        gains = np.array(
            [float(culture_kernel.get(axis, 1.0)) for axis in AFFECT_AXES],
            dtype=np.float32,
        )
        delta *= gains

    if memory_trace:
        trace = np.array(
            [float(memory_trace.get(axis, 0.0)) for axis in AFFECT_AXES],
            dtype=np.float32,
        )
        delta += 0.2 * trace

    delta *= float(culture_resonance)
    mood_delta = {
        axis: float(np.clip(delta[idx], -0.5, 0.5))
        for idx, axis in enumerate(AFFECT_AXES)
    }

    controls = _controls_from_delta(mood_delta, affect_state, tau_rate=tau_rate)
    return GreenResponse(mood_delta, controls, q_dict)


def _qualia_vector(qualia: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    tone = str(qualia.get("tone", "neutral")).lower()
    tone_map = {
        "soft": -0.4,
        "warm": -0.25,
        "neutral": 0.0,
        "bright": 0.25,
        "sharp": 0.4,
        "harsh": 0.45,
    }
    sensation = tone_map.get(tone, 0.0)

    tempo = str(qualia.get("tempo", qualia.get("rhythm", "steady"))).lower()
    tempo_map = {"slow": -0.2, "steady": 0.0, "fast": 0.25, "staccato": 0.35}
    sensation += tempo_map.get(tempo, 0.0)

    sensor_intensity = qualia.get("sensor_intensity")
    if sensor_intensity is not None:
        sensation += float(np.clip(sensor_intensity, -1.0, 1.0)) * 0.3

    semantic_valence = qualia.get("semantic_valence")
    meaning = (
        float(np.clip(semantic_valence, -1.0, 1.0))
        if semantic_valence is not None
        else 0.0
    )

    coherence = qualia.get("coherence")
    if coherence is not None:
        meaning += 0.4 * float(np.clip(coherence, -1.0, 1.0))

    trust_hint = qualia.get("trust_hint")
    if trust_hint is not None:
        meaning += 0.3 * float(np.clip(trust_hint, -1.0, 1.0))

    sensation = float(np.clip(sensation, -1.0, 1.0))
    meaning = float(np.clip(meaning, -1.0, 1.0))
    vector = np.array([sensation, meaning], dtype=np.float32)
    return vector, {"sensation": sensation, "meaning": meaning}


def _controls_from_delta(
    delta: Mapping[str, float],
    affect_state: Optional[Mapping[str, float]],
    tau_rate: float = 1.0,
) -> Dict[str, float]:
    valence = float(delta.get("v", 0.0))
    arousal = float(delta.get("a", 0.0))
    dominance = float(delta.get("d", 0.0))
    novelty = float(delta.get("n", 0.0))
    social = float(delta.get("s", 0.0))

    rate = float(_clip(tau_rate, 0.5, 1.5))
    inv_rate = 1.0 / rate

    directness_add = (0.05 * dominance - 0.04 * valence) * rate
    pause_ms_add = int((180 * max(0.0, -arousal) + 80 * max(0.0, -novelty)) * inv_rate)
    warmth_add = (0.08 * social + 0.04 * valence) * rate
    exploration_bias = 0.06 * novelty * rate

    if affect_state:
        current_a = float(affect_state.get("a", affect_state.get("arousal", 0.0)))
        if current_a > 0.6 and pause_ms_add > 0:
            pause_ms_add = int(pause_ms_add * 0.6)

    return {
        "directness_add": directness_add,
        "pause_ms_add": pause_ms_add,
        "warmth_add": warmth_add,
        "exploration_bias": exploration_bias,
    }


__all__ = ["green_response", "GreenResponse", "AFFECT_AXES", "QUALIA_AXES"]
