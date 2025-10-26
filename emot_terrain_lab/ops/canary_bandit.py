"""Canary bandit deployment scaffold for EQNet auto-tuning.

This module coordinates online experiments: a small portion of traffic
receives candidate θ configurations (from replay tuning), Thompson Sampling
updates the allocation, and Lean-based repair counters serve as guardrails.

The implementation keeps dependencies minimal; production systems should
replace the stubs with robust logging/metrics backends.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import yaml


@dataclass
class Arm:
    name: str
    theta: Mapping[str, float]
    successes: float = 1.0
    failures: float = 1.0
    active: bool = True
    notes: List[str] = field(default_factory=list)


class ThompsonBandit:
    def __init__(self, arms: List[Arm], canary_ratio: float = 0.1) -> None:
        self.arms = arms
        self.control_arm = arms[0] if arms else None
        self.canary_ratio = canary_ratio

    def sample_arm(self) -> Arm:
        if random.random() > self.canary_ratio and self.control_arm:
            return self.control_arm
        active_arms = [arm for arm in self.arms if arm.active]
        if not active_arms:
            return self.control_arm
        samples = [(random.betavariate(arm.successes, arm.failures), arm) for arm in active_arms]
        return max(samples, key=lambda x: x[0])[1]

    def update(self, arm: Arm, reward: float, penalty: float) -> None:
        if reward >= penalty:
            arm.successes += reward
        else:
            arm.failures += penalty - reward

        if arm.failures > arm.successes * 3:
            arm.active = False
            arm.notes.append("deactivated_due_to_penalty")


def load_candidates(path: Path) -> List[Arm]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    arms = []
    for idx, (name, theta) in enumerate(data.items()):
        arms.append(Arm(name=name, theta=theta, active=(idx > 0)))
    return arms


def simulate_feedback(arm: Arm) -> Tuple[float, float]:
    """Placeholder reward model: reward ~ QoR, penalty ~ repairs."""
    base_reward = random.uniform(0.5, 1.0) if arm.active else random.uniform(0.3, 0.6)
    penalty = random.uniform(0.0, 0.2)
    if "deactivated_due_to_penalty" in arm.notes:
        penalty += 0.5
    return base_reward, penalty


def log_event(path: Path, arm: Arm, reward: float, penalty: float) -> None:
    record = {
        "arm": arm.name,
        "theta": dict(arm.theta),
        "reward": reward,
        "penalty": penalty,
        "active": arm.active,
        "notes": list(arm.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet canary bandit scaffold")
    parser.add_argument("--theta", type=Path, default=Path("data/tuning/best_theta.yaml"))
    parser.add_argument("--log", type=Path, default=Path("data/tuning/bandit_log.jsonl"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--canary", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.theta.exists():
        print(f"⚠️  candidate file not found: {args.theta}")
        return 1

    arms = load_candidates(args.theta)
    if not arms:
        print("⚠️  no arms available")
        return 1

    bandit = ThompsonBandit(arms, canary_ratio=args.canary)

    for _ in range(args.steps):
        arm = bandit.sample_arm()
        reward, penalty = simulate_feedback(arm)
        bandit.update(arm, reward, penalty)
        log_event(args.log, arm, reward, penalty)

    print(f"Bandit log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
