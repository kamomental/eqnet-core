"""Replay-based auto-tuning scaffold for EQNet.

This script demonstrates how to iterate over historical session logs,
evaluate candidate parameter sets (θ), and record the best-performing
configuration per segment. Heavy optimisers (Optuna, Ax, CMA-ES) are kept
optional; the default implementation uses a simple random/Bayesian hybrid
stub so it runs even without external dependencies.

Usage:
    python -m scripts.autotune_replay --logs data/logs --out data/tuning

The output directory receives:
    - candidates.jsonl  : score breakdown per θ
    - best_theta.yaml   : segment → best θ mapping
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Configuration structures


@dataclass
class TunableParam:
    name: str
    min: float
    max: float
    prior: float
    rule: str


@dataclass
class Segment:
    persona: str
    socials: List[str]


# --------------------------------------------------------------------------- #
# Core logic


def load_tunable_config(path: Path) -> Tuple[List[TunableParam], List[Segment], Mapping[str, float]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = [
        TunableParam(name=name, min=spec["min"], max=spec["max"], prior=spec["prior"], rule=spec["rule"])
        for name, spec in cfg.get("tunable", {}).items()
    ]
    segments = [
        Segment(persona=seg["persona"], socials=seg.get("socials", ["solo"]))
        for seg in cfg.get("segments", [])
    ]
    weights = cfg.get("objective", {}).get("weights", {})
    return params, segments, weights


def sample_logs(log_dir: Path, segment: Segment) -> Iterable[Mapping[str, any]]:
    """Yield log entries matching the segment (placeholder implementation)."""
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                data = json.loads(line)
                if data.get("persona") != segment.persona:
                    continue
                if data.get("social") not in segment.socials:
                    continue
                yield data


def score_session(entry: Mapping[str, any], weights: Mapping[str, float]) -> Tuple[float, float]:
    """Compute a scalar objective and penalty from a log entry."""
    def pull(key: str, default: float = 0.0) -> float:
        value = entry.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    qor = pull("qor_score")
    natural = 1.0 - abs(pull("beta_gap"))
    resilience = 1.0 / (1.0 + pull("recovery_time"))
    slo = 1.0 if pull("response_p95") <= 180 else 0.0
    repairs = pull("lean_repairs")

    score = (
        weights.get("qor", 0.0) * qor
        + weights.get("natural", 0.0) * natural
        + weights.get("resilience", 0.0) * resilience
        + weights.get("slo", 0.0) * slo
    )
    penalty = weights.get("repairs", 0.0) * repairs
    return score, penalty


def random_theta(params: List[TunableParam]) -> Dict[str, float]:
    return {
        p.name: random.uniform(p.min, p.max)
        for p in params
    }


def nudge_theta(theta: Dict[str, float], params: List[TunableParam], scale: float = 0.2) -> Dict[str, float]:
    proposal = {}
    for p in params:
        span = p.max - p.min
        delta = random.uniform(-span * scale, span * scale)
        proposal[p.name] = float(min(p.max, max(p.min, theta[p.name] + delta)))
    return proposal


def evaluate_theta(theta: Dict[str, float], logs: Iterable[Mapping[str, any]], weights: Mapping[str, float]) -> Tuple[float, float]:
    total = 0.0
    penalty = 0.0
    count = 0
    for entry in logs:
        score, pen = score_session(entry, weights)
        total += score
        penalty += pen
        count += 1
    if count == 0:
        return -math.inf, math.inf
    return total / count, penalty / max(1, count)


def autotune_segment(
    segment: Segment,
    params: List[TunableParam],
    logs_dir: Path,
    weights: Mapping[str, float],
    rounds: int = 30,
) -> Tuple[Dict[str, float], Dict[str, any]]:
    logs = list(sample_logs(logs_dir, segment))
    if not logs:
        return {}, {"status": "no_data"}

    best_theta = random_theta(params)
    best_value, best_penalty = evaluate_theta(best_theta, logs, weights)

    history = []

    for _ in range(rounds):
        candidate = nudge_theta(best_theta, params)
        value, penalty = evaluate_theta(candidate, logs, weights)
        effective_score = value - penalty
        history.append({"theta": candidate, "value": value, "penalty": penalty})

        if effective_score > best_value - best_penalty:
            best_theta = candidate
            best_value, best_penalty = value, penalty

    return best_theta, {
        "status": "ok",
        "value": best_value,
        "penalty": best_penalty,
        "samples": len(logs),
        "history": history,
    }


def save_results(out_dir: Path, segment: Segment, theta: Dict[str, float], report: Dict[str, any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "segment": dataclass_to_dict(segment),
        "theta": theta,
        "report": report,
    }
    (out_dir / "candidates.jsonl").open("a", encoding="utf-8").write(json.dumps(record) + "\n")


def dataclass_to_dict(obj: Segment) -> Dict[str, any]:
    return {"persona": obj.persona, "socials": obj.socials}


# --------------------------------------------------------------------------- #
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet replay autotuner scaffold")
    parser.add_argument("--config", type=Path, default=Path("config/tunable.yaml"))
    parser.add_argument("--logs", type=Path, required=True, help="Directory with *.jsonl session logs")
    parser.add_argument("--out", type=Path, default=Path("data/tuning"))
    parser.add_argument("--rounds", type=int, default=30, help="Iterations per segment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params, segments, weights = load_tunable_config(args.config)
    best_map = {}
    summaries = {}

    for segment in segments:
        theta, report = autotune_segment(segment, params, args.logs, weights, rounds=args.rounds)
        best_map[segment.persona] = theta
        summaries[segment.persona] = report
        save_results(args.out, segment, theta, report)

    (args.out / "best_theta.yaml").write_text(yaml.safe_dump(best_map), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
