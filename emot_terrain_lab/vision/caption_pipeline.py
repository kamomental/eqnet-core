# -*- coding: utf-8 -*-
"""
Optional image captioning helpers for persona profiling.

This module intentionally avoids hard dependencies on ``transformers`` or
other heavy packages.  To enable captioning support install:

    pip install transformers pillow

The wrapper below relies on ``transformers.pipeline`` with BLIP or similar
image-captioning models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class CaptionConfig:
    model_id: str = "Salesforce/blip-image-captioning-large"
    device: str = "cpu"
    max_new_tokens: int = 32
    prompt: Optional[str] = None


class ImageCaptioner:
    def __init__(self, config: Optional[CaptionConfig] = None) -> None:
        self.config = config or CaptionConfig()
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "transformers is required for image captioning. "
                "Install it via `pip install transformers`."
            ) from exc
        self._pipeline = pipeline(
            "image-to-text",
            model=self.config.model_id,
            device=self.config.device,
        )

    def caption(self, image_paths: Iterable[str | Path]) -> List[str]:
        self._ensure_pipeline()
        assert self._pipeline is not None
        captions: List[str] = []
        for path in image_paths:
            p = Path(path)
            if not p.exists():
                continue
            result = self._pipeline(
                str(p),
                max_new_tokens=self.config.max_new_tokens,
                prompt=self.config.prompt,
            )
            if not result:
                continue
            text = result[0].get("generated_text") if isinstance(result, list) else None
            if text:
                captions.append(str(text).strip())
        return captions


__all__ = ["CaptionConfig", "ImageCaptioner"]
