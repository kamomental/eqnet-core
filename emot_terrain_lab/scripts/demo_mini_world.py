from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from emot_terrain_lab.mind.inner_replay import ReplayConfig
from emot_terrain_lab.sim.mini_world import (
    DEFAULT_SCENARIOS,
    MiniWorldScenario,
    MiniWorldSimulator,
    ScenarioStats,
)


def _select_scenarios(requested: List[str] | None) -> List[MiniWorldScenario]:
    if not requested:
        return list(DEFAULT_SCENARIOS.values())
    missing = [name for name in requested if name not in DEFAULT_SCENARIOS]
    if missing:
        raise SystemExit(f"Unknown scenarios: {', '.join(missing)}")
    return [DEFAULT_SCENARIOS[name] for name in requested]


def _print_summary(stats: ScenarioStats) -> None:
    if stats.steps == 0:
        print(f"[mini-world] {stats.scenario}: no steps executed")
        return
    print(
        f"[mini-world] {stats.scenario}: {stats.steps} steps | "
        f"avg hazard {stats.mean_hazard:.2f} | anchors {stats.anchors} | conscious {stats.conscious_count} | "
        f"execute {stats.execute_count} / cancel {stats.cancel_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EQNet mini-world scenarios.")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Name of a default scenario to run (commute, family_roles, workplace_safety). "
        "Repeat flag to run multiple; default is all.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("logs/mini_world_trace.jsonl"),
        help="Where to append telemetry jsonl records.",
    )
    parser.add_argument(
        "--diary-path",
        type=Path,
        default=Path("logs/mini_world_diary.jsonl"),
        help="Where to write conscious diary entries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Seed forwarded to the inner replay controller.",
    )
    args = parser.parse_args()

    scenarios = _select_scenarios(args.scenario)
    replay_config = ReplayConfig(seed=args.seed)
    simulator = MiniWorldSimulator(
        replay_config=replay_config,
        diary_path=args.diary_path,
        telemetry_path=args.trace_path,
    )

    for scenario in scenarios:
        _, stats = simulator.run_scenario(scenario)
        _print_summary(stats)

    print(f"[mini-world] trace appended to {args.trace_path}")
    print(f"[mini-world] diary entries stored at {args.diary_path}")


if __name__ == "__main__":
    main()
