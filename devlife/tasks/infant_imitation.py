"""Prototype imitation task (T2) for developmental loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass
class ImitationPattern:
    name: str
    sequence: List[int]


PATTERNS = [
    ImitationPattern("wave_hand", [0, 1, 2, 1, 0]),
    ImitationPattern("reach_object", [2, 3, 3, 1, 0]),
    ImitationPattern("rhythm_clap", [1, 0, 1, 0, 1]),
]


def pattern_stream(seed: int | None = None) -> Iterator[ImitationPattern]:
    rng = np.random.default_rng(seed)
    while True:
        yield rng.choice(PATTERNS)
