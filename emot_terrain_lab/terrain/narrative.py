# -*- coding: utf-8 -*-
"""Narrative projection adaptation and story graph logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import load_culture_projection
from .emotion import AXES


@dataclass
class NarrativeParams:
    learning_rate: float = 0.05
    decay: float = 0.995
    clip: float = 0.75
    max_examples: int = 5


class NarrativeProjection:
    """Maintain adjustable cultural projection matrices."""

    def __init__(self, config_path: Path | str = Path("config/culture.yaml"), params: NarrativeParams | None = None) -> None:
        self.config_path = Path(config_path)
        self.loader = load_culture_projection(self.config_path)
        self.params = params or NarrativeParams()
        self.adjustments: Dict[str, np.ndarray] = {}

    def to_json(self) -> Dict:
        return {
            "params": {
                "learning_rate": self.params.learning_rate,
                "decay": self.params.decay,
                "clip": self.params.clip,
                "max_examples": self.params.max_examples,
            },
            "adjustments": {loc: adj.tolist() for loc, adj in self.adjustments.items()},
        }

    @staticmethod
    def from_json(payload: Dict, config_path: Path | str = Path("config/culture.yaml")) -> "NarrativeProjection":
        params = NarrativeParams(
            learning_rate=float(payload.get("params", {}).get("learning_rate", 0.05)),
            decay=float(payload.get("params", {}).get("decay", 0.995)),
            clip=float(payload.get("params", {}).get("clip", 0.75)),
            max_examples=int(payload.get("params", {}).get("max_examples", 5)),
        )
        np_proj = NarrativeProjection(config_path=config_path, params=params)
        adjustments = payload.get("adjustments", {})
        for locale, mat in adjustments.items():
            np_proj.adjustments[locale] = np.array(mat, dtype=float)
        return np_proj

    def _get_adjustment(self, locale: str) -> np.ndarray:
        if locale not in self.adjustments:
            base = self.loader.matrix(locale)
            self.adjustments[locale] = np.zeros_like(base)
        return self.adjustments[locale]

    def matrix(self, locale: str) -> np.ndarray:
        base = self.loader.matrix(locale)
        adj = self._get_adjustment(locale)
        return base + adj

    def project(self, emotion_vec: np.ndarray, locale: str) -> np.ndarray:
        return self.matrix(locale) @ emotion_vec

    def update(self, emotion_vec: np.ndarray, locale: str, empowerment: float = 0.0) -> None:
        adj = self._get_adjustment(locale)
        adj *= self.params.decay
        norm = np.linalg.norm(emotion_vec)
        if norm <= 1e-6:
            return
        normalized = emotion_vec / norm
        top_indices = np.argsort(-np.abs(normalized))[:3]
        scale = self.params.learning_rate * (0.5 + np.clip(empowerment, 0.0, 2.0))
        for idx in top_indices:
            direction = np.sign(normalized[idx])
            adj[:, idx] += scale * direction
        np.clip(adj, -self.params.clip, self.params.clip, out=adj)


class StoryGraph:
    """Simple story graph built from dominant emotion transitions."""

    def __init__(self, max_examples: int = 5) -> None:
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], int] = {}
        self.previous_key: str | None = None
        self.max_examples = max_examples
        self.loop_run = 0
        self.loop_threshold = 4
        self.loop_alert = False

    def to_json(self) -> Dict:
        return {
            "nodes": self.nodes,
            "edges": [
                {"source": src, "target": dst, "count": count}
                for (src, dst), count in self.edges.items()
            ],
            "previous_key": self.previous_key,
            "max_examples": self.max_examples,
            "loop_alert": self.loop_alert,
            "loop_run": self.loop_run,
            "loop_threshold": self.loop_threshold,
        }

    @staticmethod
    def from_json(payload: Dict) -> "StoryGraph":
        sg = StoryGraph(max_examples=int(payload.get("max_examples", 5)))
        sg.nodes = payload.get("nodes", {})
        sg.edges = {
            (entry["source"], entry["target"]): int(entry["count"])
            for entry in payload.get("edges", [])
        }
        sg.previous_key = payload.get("previous_key")
        sg.loop_alert = payload.get("loop_alert", False)
        sg.loop_run = int(payload.get("loop_run", 0))
        sg.loop_threshold = int(payload.get("loop_threshold", sg.loop_threshold))
        return sg

    def log_event(
        self,
        timestamp: str,
        emotion_vec: np.ndarray,
        projection: np.ndarray,
        membrane_state: Dict[str, float],
        dialogue: str,
        qualia: Optional[Dict[str, float]] = None,
    ) -> bool:
        dominant_idx = int(np.argmax(np.abs(emotion_vec)))
        dominant_axis = AXES[dominant_idx]
        sign = "+" if emotion_vec[dominant_idx] >= 0 else "-"
        key = f"{dominant_axis}{sign}"

        node = self.nodes.setdefault(
            key,
            {
                "axis": dominant_axis,
                "sign": sign,
                "count": 0,
                "last_projection": [],
                "examples": [],
            },
        )
        node["count"] += 1
        node["last_projection"] = [float(projection[0]), float(projection[1])]
        if self.previous_key == key:
            self.loop_run += 1
        else:
            self.loop_run = 0
        loop_flag = self.loop_run >= self.loop_threshold
        self.loop_alert = loop_flag
        if loop_flag:
            alerts = node.setdefault("alerts", {})
            alerts["loop"] = alerts.get("loop", 0) + 1
        if len(node["examples"]) < self.max_examples and not loop_flag:
            node["examples"].append(
                {
                    "timestamp": timestamp,
                    "projection": [float(projection[0]), float(projection[1])],
                    "membrane": membrane_state,
                    "snippet": dialogue[:80],
                }
            )
            if qualia:
                node["examples"][-1]["qualia"] = qualia

        if self.previous_key is not None:
            edge = (self.previous_key, key)
            self.edges[edge] = self.edges.get(edge, 0) + 1
        self.previous_key = key
        return loop_flag
