# -*- coding: utf-8 -*-
"""Emit affect telemetry including Plutchik wheel projection."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

import numpy as np

DEFAULT_MATRIX = np.array(
    [
        [0.7, 0.2, 0.1, 0.0, 0.2, -0.2, 0.2],   # joy
        [0.5, -0.2, 0.1, -0.1, 0.6, -0.1, 0.4],  # trust
        [-0.6, 0.6, -0.5, 0.1, -0.4, 0.2, -0.1],  # fear
        [0.0, 0.5, 0.0, 0.8, -0.3, 0.0, 0.0],    # surprise
        [-0.7, -0.3, -0.2, -0.1, 0.2, 0.2, -0.2], # sadness
        [-0.5, 0.2, 0.1, -0.2, 0.3, 0.2, -0.3],   # disgust
        [-0.6, 0.7, 0.5, 0.1, -0.2, 0.5, -0.2],   # anger
        [0.2, 0.4, 0.2, 0.6, 0.1, 0.1, 0.1],      # anticipation
    ],
    dtype=np.float32,
)

DEFAULT_BIAS = np.zeros(8, dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def plutchik_from_g_axis(
    g_axis: Mapping[str, float],
    matrix: np.ndarray = DEFAULT_MATRIX,
    bias: np.ndarray = DEFAULT_BIAS,
) -> Dict[str, float]:
    vector = np.array(
        [
            float(g_axis.get("v", 0.0)),
            float(g_axis.get("a", 0.0)),
            float(g_axis.get("d", 0.0)),
            float(g_axis.get("n", 0.0)),
            float(g_axis.get("c", 0.0)),
            float(g_axis.get("e", 0.0)),
            float(g_axis.get("s", 0.0)),
        ],
        dtype=np.float32,
    )
    raw = matrix @ vector + bias
    probs = sigmoid(raw)
    labels = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    return {label: float(np.clip(probs[idx], 0.0, 1.0)) for idx, label in enumerate(labels)}


class AffectEmitter:
    """Simple emitter that publishes affect ticks over a callback or stdout."""

    def __init__(
        self,
        *,
        transport: Optional[Callable[[Dict[str, object]], None]] = None,
        logfile: Optional[Path] = None,
    ) -> None:
        self.transport = transport
        self.logfile = logfile
        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        g_axis: Mapping[str, float],
        qualia2d: Mapping[str, float],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        payload = {
            "ts": time.time(),
            "g_axis": {k: float(v) for k, v in g_axis.items()},
            "plutchik": plutchik_from_g_axis(g_axis),
            "qualia2d": {
                "x": float(qualia2d.get("x", qualia2d.get("sensation", 0.0))),
                "y": float(qualia2d.get("y", qualia2d.get("meaning", 0.0))),
            },
            "meta": dict(metadata or {}),
        }
        if self.transport:
            self.transport(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False))
        if self.logfile:
            with self.logfile.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload


__all__ = ["AffectEmitter", "plutchik_from_g_axis", "DEFAULT_MATRIX", "DEFAULT_BIAS"]
