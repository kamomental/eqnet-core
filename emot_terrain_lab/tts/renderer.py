# -*- coding: utf-8 -*-
"""Render text for specific TTS backends."""

from __future__ import annotations

import re
from html import unescape
from typing import List

BRK_TOKEN = re.compile(r"\[\[BRK:(\d{2,4})\]\]")


def _ms_to_commas(ms: int, *, unit_ms: int = 90, max_commas: int = 4) -> str:
    n = max(1, min(max_commas, round(ms / max(40, unit_ms))))
    return "、" * n


def render_for_tts(
    text: str,
    *,
    backend: str = "stylebert",
    unit_ms: int = 90,
    max_commas: int = 4,
    wrap_ssml: bool = True,
) -> str:
    """Render text for a given TTS backend, handling placeholder breaks."""
    s = unescape(text or "")
    # Remove stray <break> tags to avoid them being spoken verbatim
    s = re.sub(r"<\s*break[^>]*>", "", s, flags=re.IGNORECASE)

    backend_lower = (backend or "").lower()
    if backend_lower in {"stylebert", "style-bert-vits2", "avisspeech", "jp_commas"}:
        return BRK_TOKEN.sub(lambda m: _ms_to_commas(int(m.group(1)), unit_ms=unit_ms, max_commas=max_commas) + " ", s)

    if backend_lower in {"generic-ssml", "azure", "polly", "google"}:
        body = BRK_TOKEN.sub(
            lambda m: f'<break time="{max(20, min(1000, int(m.group(1))))}ms"/>',
            s,
        )
        if wrap_ssml and not body.strip().startswith("<speak"):
            body = f"<speak>{body}</speak>"
        return body

    # Default: strip placeholders entirely
    return BRK_TOKEN.sub(" ", s)


__all__ = ["render_for_tts"]
