# -*- coding: utf-8 -*-
"""Fast-path cleanup summary using registered task profiles."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from .task_fastpath import summarize_task_fastpath
from .task_profiles import TASK_PROFILES


def summarize_cleanup(
    J_index: Sequence[float],
    projector_map: Mapping[str, Callable[[float], Any]],
) -> Dict[str, Any]:
    """Helper dedicated to the cleanup profile (still shared by legacy tests)."""

    profile = TASK_PROFILES["cleanup"]
    summary = summarize_task_fastpath(profile, J_index, projector_map, include_full_labels=False)
    summary.pop("profile", None)
    return summary


__all__ = ["summarize_cleanup"]
