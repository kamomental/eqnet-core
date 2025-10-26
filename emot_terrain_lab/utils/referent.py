# -*- coding: utf-8 -*-
"""Utilities for referent/shared-attention hints."""

from __future__ import annotations

from typing import Optional


FORBIDDEN_TOKENS = ("face", "coord", "bbox", "x=", "y=", "id_", "track")


def sanitize_referent_label(label: Optional[str]) -> Optional[str]:
    """Return label stripped of obvious PII tokens, or None."""

    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    lowered = text.lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        return None
    return text


__all__ = ["sanitize_referent_label"]
