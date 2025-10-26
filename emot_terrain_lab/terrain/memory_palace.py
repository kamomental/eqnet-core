# -*- coding: utf-8 -*-
"""Memory palace nodes with adaptive forgetting operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .emotion import AXES, AXIS_BOUNDS
from .config import project_emotion


@dataclass
class MemoryNode:
    name: str
    locale: str
    position: List[float]
    capacity: int = 64
    decay: float = 0.96
    reinforcement: float = 0.12
    threshold: float = 0.35

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "locale": self.locale,
            "position": self.position,
            "capacity": self.capacity,
            "decay": self.decay,
            "reinforcement": self.reinforcement,
            "threshold": self.threshold,
        }

    @staticmethod
    def from_json(payload: Dict) -> "MemoryNode":
        return MemoryNode(
            name=payload["name"],
            locale=payload.get("locale", "default"),
            position=list(payload.get("position", [0.5, 0.5])),
            capacity=int(payload.get("capacity", 64)),
            decay=float(payload.get("decay", 0.96)),
            reinforcement=float(payload.get("reinforcement", 0.12)),
            threshold=float(payload.get("threshold", 0.35)),
        )


class MemoryPalace:
    """Collection of memory nodes storing trace values on a 2D map."""

    def __init__(self, nodes: Optional[List[MemoryNode]] = None) -> None:
        self.nodes: Dict[str, MemoryNode] = {node.name: node for node in (nodes or [])}
        self.traces: Dict[str, np.ndarray] = {
            node.name: np.zeros(node.capacity, dtype=float) for node in self.nodes.values()
        }
        self.labels: Dict[str, List[str]] = {
            node.name: [""] * node.capacity for node in self.nodes.values()
        }
        self.qualia_state: Dict[str, Dict[str, float | List[float] | int]] = self._default_qualia_state()
        self.overload_threshold = 0.82
        self.overload_decay = 0.7

    def to_json(self) -> Dict:
        return {
            "nodes": [node.to_json() for node in self.nodes.values()],
            "traces": {name: trace.tolist() for name, trace in self.traces.items()},
            "labels": self.labels,
            "qualia_state": self.qualia_state,
        }

    @staticmethod
    def from_json(payload: Dict) -> "MemoryPalace":
        nodes = [MemoryNode.from_json(item) for item in payload.get("nodes", [])]
        palace = MemoryPalace(nodes)
        for name, trace in payload.get("traces", {}).items():
            palace.traces[name] = np.array(trace, dtype=float)
        palace.labels = payload.get("labels", palace.labels)
        palace.qualia_state = payload.get("qualia_state", palace._default_qualia_state())
        palace._ensure_qualia_state()
        return palace

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.name] = node
        self.traces[node.name] = np.zeros(node.capacity, dtype=float)
        self.labels[node.name] = [""] * node.capacity
        self.qualia_state[node.name] = self._baseline_qualia_entry()
        self._ensure_qualia_state()

    def update(
        self,
        emotion_vec: np.ndarray,
        locale: str,
        dialogue: str,
        weight: float = 1.0,
        qualia: Optional[Dict[str, float | List[float]]] = None,
    ) -> bool:
        overload_triggered = False
        for node in self.nodes.values():
            if node.locale != locale:
                continue
            x, y = project_emotion(emotion_vec, locale)
            pos = np.clip(np.array([x, y]), 0.0, 1.0)
            dist = np.linalg.norm(pos - np.array(node.position))
            closeness = float(np.exp(-4.0 * dist * dist))
            if closeness < node.threshold:
                continue
            trace = self.traces[node.name]
            trace *= node.decay
            trace += node.reinforcement * closeness * weight
            self.labels[node.name] = self.labels[node.name][-node.capacity :]
            self.labels[node.name].append(dialogue[:80])
            self.labels[node.name] = self.labels[node.name][-node.capacity :]
            np.clip(trace, 0.0, 1.0, out=trace)
            if qualia:
                self._update_qualia_state(node.name, qualia, weight)
            load = float(np.mean(trace))
            if load >= self.overload_threshold:
                trace *= self.overload_decay
                np.clip(trace, 0.0, 1.0, out=trace)
                overload_triggered = True
        return overload_triggered

    def get_heatmap(self, locale: str, resolution: int = 64) -> np.ndarray:
        heatmap = np.zeros((resolution, resolution), dtype=float)
        xs = np.linspace(0, 1, resolution)
        ys = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        for node in self.nodes.values():
            if node.locale != locale:
                continue
            total = float(self.traces[node.name].sum())
            if total <= 1e-6:
                continue
            pos = np.array(node.position)
            sigma = 0.08 + 0.02 * np.sqrt(total)
            heatmap += total * np.exp(-((X - pos[0]) ** 2 + (Y - pos[1]) ** 2) / (2 * sigma ** 2))
        return heatmap

    def strongest_alignment(self, locale: str, position: np.ndarray) -> Tuple[str | None, float]:
        best_node = None
        best_score = 0.0
        for node in self.nodes.values():
            if node.locale != locale:
                continue
            total = float(self.traces[node.name].sum())
            if total <= 1e-6:
                continue
            pos = np.array(node.position)
            dist = np.linalg.norm(position - pos)
            closeness = float(np.exp(-4.0 * dist * dist))
            score = closeness * total
            if score > best_score:
                best_score = score
                best_node = node.name
        return best_node, best_score

    def _default_qualia_state(self) -> Dict[str, Dict[str, float | List[float] | int]]:
        state: Dict[str, Dict[str, float | List[float] | int]] = {}
        for node in self.nodes.values():
            state[node.name] = self._baseline_qualia_entry()
        return state

    @staticmethod
    def _baseline_qualia_entry() -> Dict[str, float | List[float] | int]:
        return {
            "energy": 0.0,
            "magnitude": 0.0,
            "phase": 0.0,
            "memory": 0.0,
            "enthalpy": 0.0,
            "flow": [0.0, 0.0],
            "samples": 0,
        }

    def _ensure_qualia_state(self) -> None:
        for node in self.nodes.values():
            state = self.qualia_state.setdefault(node.name, self._baseline_qualia_entry())
            template = self._baseline_qualia_entry()
            for key, default in template.items():
                if key not in state:
                    state[key] = default if not isinstance(default, list) else list(default)

    def _update_qualia_state(
        self,
        node_name: str,
        qualia: Dict[str, float | List[float]],
        weight: float,
    ) -> None:
        state = self.qualia_state.setdefault(node_name, self._baseline_qualia_entry())
        beta = min(0.5, 0.1 + 0.1 * max(float(weight), 0.0))
        flow_current = np.array(state.get("flow", [0.0, 0.0]), dtype=float)
        flow_update = np.array(qualia.get("flow", [0.0, 0.0]), dtype=float)
        state["energy"] = float((1 - beta) * float(state.get("energy", 0.0)) + beta * float(qualia.get("energy", 0.0)))
        state["magnitude"] = float(
            (1 - beta) * float(state.get("magnitude", 0.0)) + beta * float(qualia.get("magnitude", 0.0))
        )
        state["phase"] = float((1 - beta) * float(state.get("phase", 0.0)) + beta * float(qualia.get("phase", 0.0)))
        state["memory"] = float((1 - beta) * float(state.get("memory", 0.0)) + beta * float(qualia.get("memory", 0.0)))
        state["enthalpy"] = float(
            (1 - beta) * float(state.get("enthalpy", 0.0)) + beta * float(qualia.get("enthalpy", 0.0))
        )
        flow_blend = (1 - beta) * flow_current + beta * flow_update
        state["flow"] = flow_blend.tolist()
        state["samples"] = int(state.get("samples", 0)) + 1
