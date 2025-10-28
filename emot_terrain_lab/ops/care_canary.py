from __future__ import annotations

from random import Random
from typing import Iterable, Sequence, Set


def _clean_candidates(candidates: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).strip()
        if not value:
            continue
        cleaned.append(value)
    # ensure deterministic ordering before sampling
    cleaned = sorted(set(cleaned))
    return cleaned


def select_canary_ids(
    candidates: Sequence[str],
    ratio: float,
    seed: int = 0,
) -> Set[str]:
    """Return a deterministic subset of candidates for canary application."""
    cleaned = _clean_candidates(candidates)
    if not cleaned:
        return set()
    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= 0.0:
        return set()
    target_count = int(round(len(cleaned) * ratio))
    if target_count <= 0:
        target_count = 1
    target_count = min(len(cleaned), target_count)
    rng = Random(int(seed))
    selected = rng.sample(cleaned, target_count)
    return set(selected)


__all__ = ["select_canary_ids"]
