# -*- coding: utf-8 -*-
"""Metaphor/imagery helpers for bridging non-shared sensations."""

from __future__ import annotations

from typing import Dict, Iterable, List


def pick_bridges(
    targets: Iterable[str],
    metaphors_cfg: Dict[str, object],
    *,
    max_items: int = 2,
) -> List[str]:
    """Choose at most `max_items` metaphorical phrases for the targets."""

    phrases: List[str] = []
    mapping = metaphors_cfg.get("missing_to_phrase", {}) if metaphors_cfg else {}
    if not isinstance(mapping, dict):
        return phrases

    for target in targets:
        candidates = mapping.get(target)
        if not isinstance(candidates, list):
            continue
        for phrase in candidates:
            if phrase and phrase not in phrases:
                phrases.append(phrase)
                break
        if len(phrases) >= max_items:
            break
    return phrases


__all__ = ["pick_bridges"]
