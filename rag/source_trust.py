# -*- coding: utf-8 -*-
"""Heuristics for estimating retrieval source trustworthiness."""

from __future__ import annotations

from typing import Any, Dict


_SOURCE_TRUST_TABLE = {
    "official": 0.95,
    "scientific": 0.9,
    "news": 0.75,
    "gov": 0.85,
    "ngo": 0.8,
    "blog": 0.55,
    "forum": 0.45,
    "social": 0.35,
}


def trust_score(metadata: Dict[str, Any] | None) -> float:
    """Return a soft trust estimate in [0, 1]."""
    if not metadata:
        return 0.5
    source_type = str(metadata.get("source_type") or metadata.get("domain") or "").lower()
    if source_type:
        for key, value in _SOURCE_TRUST_TABLE.items():
            if key in source_type:
                return value
    site = str(metadata.get("site") or "").lower()
    if "wikipedia" in site:
        return 0.7
    if site.endswith(".gov") or site.endswith(".go.jp"):
        return 0.85
    if site.endswith(".ac.jp") or ".edu" in site:
        return 0.82
    return 0.5


__all__ = ["trust_score"]

