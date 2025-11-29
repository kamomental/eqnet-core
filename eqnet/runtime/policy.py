"""Policy prior helpers that interact with qualia replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass
class PolicyPrior:
    """Lightweight style vector that nudges downstream responses."""

    warmth: float = 0.5
    directness: float = 0.5
    self_disclosure: float = 0.5
    calmness: float = 0.5

    def as_array(self) -> Array:
        return np.array(
            [self.warmth, self.directness, self.self_disclosure, self.calmness],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: Iterable[float]) -> "PolicyPrior":
        vector = list(arr)
        if len(vector) != 4:
            raise ValueError("PolicyPrior vector must have length 4")
        return cls(
            warmth=float(vector[0]),
            directness=float(vector[1]),
            self_disclosure=float(vector[2]),
            calmness=float(vector[3]),
        )


def apply_imagery_update(
    prior: PolicyPrior,
    imagined_traj: Sequence[Array] | None,
    avg_potential: float,
    avg_life_indicator: float,
    lr: float = 0.05,
) -> PolicyPrior:
    """Return a slightly adjusted prior based on imagined trajectories.

    TODO: 将来的には ``imagined_traj`` の方向（例: fog が下がったか）を使って
    更新方向を推定する。現在はスカラー reward のみで暖かさ／落ち着きを
    わずかに調整する。

    Note: avg_life_indicator を reward に使う場合、Nightly 側で加える meta
    ボーナスと二重計上にならないよう設計すること。
    """

    _ = imagined_traj  # v0 では未使用

    reward = (-avg_potential + avg_life_indicator) * 0.5
    reward = float(np.clip(reward, -1.0, 1.0))

    arr = prior.as_array()
    update = np.array([reward, 0.0, 0.0, reward], dtype=float)
    new_arr = np.clip(arr + lr * update, 0.0, 1.0)
    return PolicyPrior.from_array(new_arr)

