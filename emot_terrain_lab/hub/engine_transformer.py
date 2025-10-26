# -*- coding: utf-8 -*-
"""Transformer-backed inference engine wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .inference import InferenceEngine


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _map_controls(controls: Dict[str, Any]) -> Dict[str, Any]:
    temp = float(controls.get("temp_mul", 1.0))
    top_p = float(controls.get("top_p_mul", 1.0))
    return {
        "temperature": _clamp(temp, 0.1, 1.5),
        "top_p": _clamp(top_p, 0.05, 1.0),
        "max_tokens": int(controls.get("max_tokens", 256)),
        "stop": controls.get("stop"),
    }


class TransformerEngine(InferenceEngine):
    """Wrap a transformer-style text generator to match the hub interface."""

    def __init__(self, model: Any, tokenizer: Optional[Any] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        controls: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        kwargs = _map_controls(controls)
        # Allow models that expect tokenizer inputs vs raw text.
        if hasattr(self.model, "generate_text"):
            text = self.model.generate_text(prompt, **kwargs)
        elif self.tokenizer is not None and hasattr(self.model, "generate"):
            tokens = self.tokenizer(prompt, return_tensors="pt")
            gen_tokens = self.model.generate(**tokens, **kwargs)
            text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        else:
            # Fallback for stub models used in tests.
            text = str(self.model(prompt))
        return {"text": text, "trace": {"engine": "transformer"}}


__all__ = ["TransformerEngine"]
