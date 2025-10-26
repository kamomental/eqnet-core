# -*- coding: utf-8 -*-
"""Dragon Hatchling (BDH) inference engine adapter."""

from __future__ import annotations

from typing import Any, Dict

from .inference import InferenceEngine


def _map_controls(controls: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "temperature_mul": float(controls.get("temp_mul", 1.0)),
        "top_p_mul": float(controls.get("top_p_mul", 1.0)),
    }


class BDHEngine(InferenceEngine):
    """Thin adapter around an optional BDH model implementation."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = None
        loader = cfg.get("loader")
        if callable(loader):
            self.model = loader(cfg)
            return
        try:
            from bdh import load_model  # type: ignore

            path = cfg.get("model_path", "models/bdh-small")
            device = cfg.get("device", "cpu")
            self.model = load_model(path, device=device)
        except Exception:
            self.model = None

    def generate(
        self,
        prompt: str,
        controls: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        mapped = _map_controls(controls)
        if self.model is None:
            text = f"[BDH-STUB] {prompt}"
            trace = {
                "engine": "bdh",
                "sparsity": None,
                "hebbian_updates": None,
                "active_modules": None,
            }
            return {"text": text, "trace": trace}

        try:
            output = self.model.infer(
                prompt,
                hebbian=self.cfg.get("hebbian", True),
                sparse=self.cfg.get("sparse", True),
                controls=mapped,
            )
        except Exception:
            text = f"[BDH-ERROR] {prompt}"
            trace = {
                "engine": "bdh",
                "sparsity": None,
                "hebbian_updates": None,
                "active_modules": None,
            }
            return {"text": text, "trace": trace}

        stats = getattr(output, "stats", None)
        trace = {
            "engine": "bdh",
            "sparsity": getattr(stats, "sparsity", None) if stats else None,
            "hebbian_updates": getattr(stats, "hebbian_delta", None) if stats else None,
            "active_modules": getattr(stats, "active_modules", None) if stats else None,
        }
        return {"text": getattr(output, "text", ""), "trace": trace}


__all__ = ["BDHEngine"]
