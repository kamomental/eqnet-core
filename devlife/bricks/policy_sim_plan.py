"""Counterfactual planning stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PlanConfig:
    horizon: int = 30
    num_plans: int = 3
    seed: int = 0


@dataclass
class PolicySimPlan:
    config: PlanConfig = field(default_factory=PlanConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)

    def simulate(self, field_state: np.ndarray, goal: Dict[str, float] | None = None) -> Dict[str, List[int]]:
        plans = []
        scores = []
        for _ in range(self.config.num_plans):
            plan = self.rng.integers(low=0, high=5, size=self.config.horizon).tolist()
            score = float(self.rng.random())
            plans.append(plan)
            scores.append(score)
        best = int(np.argmax(scores))
        rejected = [i for i in range(len(plans)) if i != best]
        return {
            "plan_chosen": plans[best],
            "plans_rejected": [plans[i] for i in rejected],
            "scores": scores,
            "best_index": best,
        }

