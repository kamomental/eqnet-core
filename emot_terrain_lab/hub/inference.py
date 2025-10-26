# -*- coding: utf-8 -*-
"""Inference engine abstraction for EQNet hub."""

from __future__ import annotations

from typing import Any, Dict


class InferenceEngine:
    """Minimal interface for interchangeable inference engines."""

    def generate(
        self,
        prompt: str,
        controls: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce text and an optional trace payload."""
        raise NotImplementedError


__all__ = ["InferenceEngine"]
