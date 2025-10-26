"""Delayed report evaluation for assessing working memory in the field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class DelayedReportMetrics:
    accuracy: float
    chance: float
    window: int
    trials: int


def evaluate_delayed_report(
    targets: Sequence[int],
    responses: Sequence[int],
    delay_steps: int,
    chance_level: float = 0.25,
) -> DelayedReportMetrics:
    """Compare delayed responses to ground truth."""
    targets_arr = np.asarray(targets)
    responses_arr = np.asarray(responses)
    if targets_arr.size == 0 or responses_arr.size == 0:
        return DelayedReportMetrics(accuracy=0.0, chance=chance_level, window=delay_steps, trials=0)
    trials = min(targets_arr.size, responses_arr.size)
    acc = float(np.mean(targets_arr[:trials] == responses_arr[:trials]))
    return DelayedReportMetrics(accuracy=acc, chance=chance_level, window=delay_steps, trials=trials)
