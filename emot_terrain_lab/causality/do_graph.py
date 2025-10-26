# -*- coding: utf-8 -*-
"""Dependent-origination (五蘊) causal flow tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import math

NODE_KEYS = ["色", "受", "想", "行", "識"]
EDGE_KEYS = [
    ("色", "受"),
    ("受", "想"),
    ("想", "行"),
    ("行", "識"),
    ("識", "想"),
]


@dataclass
class DOEdgeState:
    """Per-edge gain + exponential moving average."""

    w: float = 0.2
    ema: float = 0.0
    last: float = 0.0


@dataclass
class DOgraph:
    """Causal flow tracker across 五蘊 nodes."""

    tau_ema: float = 6.0
    contrib_cap: float = 1.0
    normalize_topk: bool = False
    node_act: Dict[str, float] = field(
        default_factory=lambda: {k: 0.0 for k in NODE_KEYS}
    )
    edges: Dict[Tuple[str, str], DOEdgeState] = field(
        default_factory=lambda: {edge: DOEdgeState() for edge in EDGE_KEYS}
    )

    def set_contrib_cap(self, cap: float) -> None:
        self.contrib_cap = max(0.0, float(cap))

    def load_weights(self, weights: Dict[str, float]) -> None:
        """Load edge weights from config: {'色→受': 0.25, ...}."""

        for key, value in (weights or {}).items():
            if "→" not in key:
                continue
            src, dst = key.split("→", 1)
            edge = (src, dst)
            if edge in self.edges:
                try:
                    self.edges[edge].w = float(value)
                except Exception:
                    continue

    def update(self, signals: Dict[str, float], d_tau: float) -> None:
        """Update node activations and edge contributions."""

        for node in NODE_KEYS:
            if node in signals:
                self.node_act[node] = float(
                    max(0.0, min(1.0, signals.get(node, 0.0)))
                )
        alpha = math.exp(-max(d_tau, 0.0) / max(self.tau_ema, 1e-6))
        for (src, _dst), state in self.edges.items():
            contrib = self.node_act[src] * state.w * max(d_tau, 0.0)
            if self.contrib_cap > 0.0:
                contrib = min(contrib, self.contrib_cap)
            state.last = contrib
            state.ema = alpha * state.ema + (1.0 - alpha) * contrib

    def topk(self, k: int = 3, *, use_ema: bool = True) -> List[Tuple[str, float]]:
        """Return top-k edge contributions."""

        pairs: List[Tuple[str, float]] = []
        for (src, dst), state in self.edges.items():
            value = state.ema if use_ema else state.last
            pairs.append((f"{src}→{dst}", float(value)))
        if self.normalize_topk:
            total = sum(v for _name, v in pairs if v > 0)
            if total > 0:
                pairs = [(name, val / total) for name, val in pairs]
        pairs.sort(key=lambda item: -item[1])
        return pairs[:k]

    def snapshot(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return raw node and edge state for logging/debug."""

        edges = {
            f"{src}→{dst}": {"w": st.w, "ema": st.ema, "last": st.last}
            for (src, dst), st in self.edges.items()
        }
        return {"nodes": dict(self.node_act), "edges": edges}


__all__ = ["DOgraph", "NODE_KEYS", "EDGE_KEYS"]
