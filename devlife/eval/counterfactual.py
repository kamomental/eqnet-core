"""Counterfactual simulation alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CounterfactualResult:
    match_rate: float
    random_rate: float
    trials: int


def counterfactual_match(predicted: Sequence[int], executed: Sequence[int], num_options: int) -> CounterfactualResult:
    """Compute how often predicted branches match executed outcome."""
    pred = np.asarray(predicted)
    exec_arr = np.asarray(executed)
    trials = min(pred.size, exec_arr.size)
    if trials == 0 or num_options <= 0:
        return CounterfactualResult(match_rate=0.0, random_rate=0.0, trials=0)
    match = float(np.mean(pred[:trials] == exec_arr[:trials]))
    random = 1.0 / float(num_options)
    return CounterfactualResult(match_rate=match, random_rate=random, trials=trials)
