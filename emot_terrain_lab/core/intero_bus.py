# -*- coding: utf-8 -*-
"""Interoceptive signal bus handling diffusive fields and tonic values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


class InteroBus:
    """Maintain slow-changing internal signals and diffusive fields."""

    def __init__(
        self,
        nodes: Iterable[str],
        adj: Mapping[Tuple[str, str], float],
        *,
        alpha: float = 0.3,
        beta_inject: float = 0.2,
    ) -> None:
        self.nodes = list(nodes)
        self.adj = {tuple(edge): float(weight) for edge, weight in adj.items()}
        self.alpha = float(alpha)
        self.beta_inject = float(beta_inject)
        self.diffusion_scale = 1.0
        self.tonic: Dict[str, float] = {}
        self.field: Dict[str, float] = {node: 0.0 for node in self.nodes}

    def publish(self, kind: str, value: float, node: str | None = None) -> None:
        """Inject tonic/global signals or localized field sources."""
        if kind == "inflammation" and node and node in self.field:
            self.field[node] = _clip01(self.field[node] + self.beta_inject * float(value))
            return
        self.tonic[kind] = max(-1.0, min(1.0, float(value)))

    def step(self, d_tau: float) -> None:
        """Advance field diffusion by subjective-time step Δτ."""
        if not self.field:
            return
        current = self.field.copy()
        for i in self.nodes:
            diff = -self.alpha * current[i]
            for j in self.nodes:
                if (i, j) in self.adj:
                    diff += self.diffusion_scale * self.adj[(i, j)] * (current[j] - current[i])
            self.field[i] = _clip01(current[i] + d_tau * diff)

    def effective(self) -> Dict[str, float]:
        """Return aggregated signals usable by modulation layers."""
        if self.field:
            avg_field = sum(self.field.values()) / max(1, len(self.field))
        else:
            avg_field = 0.0
        out = dict(self.tonic)
        out["inflammation_global"] = avg_field
        return out

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return current state for logging/receipts."""
        return {
            "tonic": dict(self.tonic),
            "field": dict(self.field),
            "alpha": self.alpha,
            "beta_inject": self.beta_inject,
            "diffusion_scale": self.diffusion_scale,
        }

    def restore(self, snapshot: Mapping[str, Mapping[str, float]]) -> None:
        """Restore bus state from a snapshot mapping."""
        tonic = snapshot.get("tonic", {})
        if isinstance(tonic, Mapping):
            for key, value in tonic.items():
                self.tonic[str(key)] = float(value)
        field = snapshot.get("field", {})
        if isinstance(field, Mapping):
            for key, value in field.items():
                if key in self.field:
                    self.field[key] = _clip01(value)
        if "alpha" in snapshot:
            try:
                self.alpha = float(snapshot["alpha"])
            except Exception:
                pass
        if "beta_inject" in snapshot:
            try:
                self.beta_inject = float(snapshot["beta_inject"])
            except Exception:
                pass
        self.diffusion_scale = float(snapshot.get("diffusion_scale", self.diffusion_scale))

    def save(self, path: str | Path, state: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Persist current state to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(state) if state is not None else self.snapshot()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def load(self, path: str | Path) -> Dict[str, Any] | None:
        """Load state from JSON if available."""
        target = Path(path)
        if not target.exists():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, Mapping):
            self.restore(data)
            return dict(data)
        return None

    def set_decay_params(self, params: Mapping[str, float]) -> None:
        """Adjust decay parameters in response to forgetting advice."""
        if not isinstance(params, Mapping):
            return
        if "alpha" in params:
            try:
                self.alpha = float(params["alpha"])
            except Exception:
                pass
        if "beta_inject" in params:
            try:
                self.beta_inject = float(params["beta_inject"])
            except Exception:
                pass
        if "D_scale" in params:
            try:
                self.diffusion_scale = max(0.0, float(params["D_scale"]))
            except Exception:
                pass


def _clip01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


__all__ = ["InteroBus"]
