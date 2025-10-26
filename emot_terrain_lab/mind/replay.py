# -*- coding: utf-8 -*-
"""Unified replay scaffold."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple


class UnifiedReplay:
    """Minimal rollout stub to unblock integrations."""

    def rollout(
        self,
        *,
        z0: Any,
        horizon: int,
        value_fn: Callable[[Any], float],
    ) -> List[Tuple[Any, float]]:
        state = z0
        trace: List[Tuple[Any, float]] = []
        for _ in range(max(0, int(horizon))):
            state = self._transition(state)
            trace.append((state, value_fn(state)))
        return trace

    def _transition(self, state: Any) -> Any:
        return state


__all__ = ["UnifiedReplay"]
