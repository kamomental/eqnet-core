#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore resonance gain (k_res) and record outcomes.

Example:
    python scripts/tune_resonance.py \
        --mode grid \
        --run-cmd "python scripts/run_quick_loop.py --steps {steps} --seed {seed}" \
        --nightly-cmd "python -m emot_terrain_lab.ops.nightly --telemetry_log telemetry/ignition-*.jsonl" \
        --logs telemetry/ignition-*.jsonl telemetry/resonance-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _run_command(template: str, **fmt) -> None:
    command = template.format(**fmt)
    subprocess.run(shlex.split(command), check=True)


def _update_resonance_cfg(cfg_path: Path, k_res: float, logs: Sequence[str] | None) -> None:
    cfg = _load_yaml(cfg_path)
    resonance = cfg.setdefault("resonance", {})
    resonance["k_res"] = float(k_res)
    if logs is not None:
        resonance["logs"] = list(logs)
    _dump_yaml(cfg_path, cfg)


def _restore_cfg(cfg_path: Path, snapshot: dict | None) -> None:
    if snapshot is None:
        return
    _dump_yaml(cfg_path, snapshot)


def _append_bayes_trace(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_objective(nightly_path: Path) -> dict:
    if not nightly_path.exists():
        return {}
    report = json.loads(nightly_path.read_text(encoding="utf-8"))
    resonance = report.get("resonance") or {}
    summary = resonance.get("summary")
    if summary:
        base = summary.copy()
    else:
        pairs = resonance.get("pairs") or []
        base = pairs[0] if pairs else {}
    base["objective"] = base.get("objective", resonance.get("objective"))
    base["corr"] = base.get("corr", resonance.get("corr"))
    base["lag"] = base.get("lag", resonance.get("lag"))
    base["energy"] = base.get("energy", resonance.get("energy"))
    if "partial_corr" in resonance and "partial_corr" not in base:
        base["partial_corr"] = resonance.get("partial_corr")
    base["partial_corr"] = base.get("partial_corr")
    if "granger" in resonance and "granger" not in base:
        base["granger"] = resonance.get("granger")
    base["granger"] = base.get("granger")
    return base


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid/Random/Bayesian search for resonance gain (k_res).")
    parser.add_argument("--mode", choices=["grid", "random", "bayes"], default="grid", help="Exploration mode.")
    parser.add_argument("--grid", default="0.00,0.02,0.04,0.06,0.08,0.10", help="Comma-separated k_res for grid mode.")
    parser.add_argument("--trials", type=int, default=6, help="Number of samples for random mode.")
    parser.add_argument("--bayes-trials", type=int, default=10, help="Total trials for Bayesian mode.")
    parser.add_argument("--init-samples", type=int, default=3, help="Initial random samples before Bayesian updates.")
    parser.add_argument("--steps", type=int, default=300, help="Steps for the run command.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed offset.")
    parser.add_argument("--k-min", type=float, default=0.0, help="Lower bound for k_res search.")
    parser.add_argument("--k-max", type=float, default=0.12, help="Upper bound for k_res search.")
    parser.add_argument(
        "--run-cmd",
        default="python scripts/run_quick_loop.py --steps {steps} --seed {seed}",
        help="Command to generate telemetry (format placeholders: {steps}, {seed}, {k_res}).",
    )
    parser.add_argument(
        "--nightly-cmd",
        default="python -m emot_terrain_lab.ops.nightly --telemetry_log telemetry/ignition-*.jsonl",
        help="Command to produce nightly summary (placeholders: {k_res}).",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Resonance log patterns to write into runtime.yaml (leave unset to keep existing).",
    )
    parser.add_argument("--config", default="config/runtime.yaml", help="Runtime config path.")
    parser.add_argument("--nightly", default="reports/nightly.json", help="Nightly JSON path.")
    parser.add_argument("--persist", action="store_true", help="Persist best k_res in config (default restores snapshot).")
    parser.add_argument(
        "--candidate-points",
        type=int,
        default=200,
        help="Number of candidate points for Bayesian EI maximisation.",
    )
    parser.add_argument(
        "--exploration-xi",
        type=float,
        default=0.01,
        help="Exploration parameter xi for expected improvement.",
    )
    return parser.parse_args(argv)


def _unique_random(rng: random.Random, low: float, high: float, used: List[float], attempts: int = 128) -> float:
    candidate = rng.uniform(low, high)
    for _ in range(attempts):
        candidate = rng.uniform(low, high)
        if all(abs(candidate - val) > 1e-4 for val in used):
            break
    return float(max(low, min(high, candidate)))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)
    improvement = mu - best - xi
    z = np.zeros_like(improvement)
    valid = sigma > 1e-12
    z[valid] = improvement[valid] / sigma[valid]
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * np.square(z))
    erf_inputs = z / math.sqrt(2.0)
    erf_vals = np.array([math.erf(val) for val in erf_inputs])
    cdf = 0.5 * (1.0 + erf_vals)
    ei = improvement * cdf + sigma * pdf
    ei[~valid] = 0.0
    ei[np.isnan(ei)] = 0.0
    return ei


def _bayes_next_candidate(
    results: List[dict],
    used: List[float],
    *,
    k_min: float,
    k_max: float,
    candidate_points: int,
    xi: float,
    rng: random.Random,
) -> float:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Bayesian mode requires scikit-learn. Install it via 'pip install scikit-learn'."
        ) from exc

    finite_results = [
        (float(r["k_res"]), float(r["objective"]))
        for r in results
        if r.get("objective") is not None and math.isfinite(r.get("objective"))
    ]
    if len(finite_results) < 2:
        return _unique_random(rng, k_min, k_max, used)

    X = np.array([row[0] for row in finite_results], dtype=float).reshape(-1, 1)
    y = np.array([row[1] for row in finite_results], dtype=float)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=0.05, nu=2.5) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2)
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=rng.randint(0, 2**32 - 1))
    gp.fit(X, y)

    candidates = np.linspace(k_min, k_max, max(candidate_points, 5))
    for val in used:
        idx = np.abs(candidates - val).argmin()
        candidates[idx] = np.nan
    candidates = candidates[np.isfinite(candidates)]
    if candidates.size == 0:
        return _unique_random(rng, k_min, k_max, used)

    mu, sigma = gp.predict(candidates.reshape(-1, 1), return_std=True)
    best = float(np.max(y))
    ei = _expected_improvement(mu, sigma, best, xi)
    if np.all(ei <= 0):
        return _unique_random(rng, k_min, k_max, used)
    chosen = float(candidates[np.argmax(ei)])
    if any(abs(chosen - val) <= 1e-4 for val in used):
        return _unique_random(rng, k_min, k_max, used)
    return chosen


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config)
    nightly_path = Path(args.nightly)
    trace_path: Path | None = None
    if not cfg_path.exists():
        print(f"[tune] config not found: {cfg_path}")
        return 1

    original_cfg = _load_yaml(cfg_path)
    logs = args.logs

    rng = random.Random(args.seed)
    k_min = float(args.k_min)
    k_max = float(args.k_max)

    if args.mode == "bayes":
        trace_path = Path("reports/resonance_bayes_trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)

    if k_min >= k_max:
        print("[tune] invalid range: k_min must be less than k_max")
        return 2

    if args.mode == "grid":
        k_values = [float(x) for x in args.grid.split(",") if x]
        total_trials = len(k_values)
    elif args.mode == "random":
        total_trials = args.trials
        k_values = []
    else:  # bayes
        total_trials = args.bayes_trials
        k_values = []

    results: List[dict] = []
    used_k: List[float] = []
    try:
        for idx in range(total_trials):
            if args.mode == "grid":
                k_res = k_values[idx]
            elif args.mode == "random":
                k_res = _unique_random(rng, k_min, k_max, used_k)
            else:  # bayes
                if idx < max(1, args.init_samples):
                    k_res = _unique_random(rng, k_min, k_max, used_k)
                else:
                    try:
                        k_res = _bayes_next_candidate(
                            results,
                            used_k,
                            k_min=k_min,
                            k_max=k_max,
                            candidate_points=args.candidate_points,
                            xi=args.exploration_xi,
                            rng=rng,
                        )
                    except RuntimeError as exc:
                        print(f"[tune] {exc}")
                        return 3
            used_k.append(k_res)
            seed = args.seed + idx
            print(f"[tune] trial {idx+1}/{total_trials}  k_res={k_res:.4f}")
            _update_resonance_cfg(cfg_path, k_res, logs)
            _run_command(args.run_cmd, steps=args.steps, seed=seed, k_res=k_res)
            _run_command(args.nightly_cmd, k_res=k_res)
            metrics = _read_objective(nightly_path)
            metrics["k_res"] = k_res
            print(f"[tune] result {metrics}")
            results.append(metrics)
            if trace_path is not None:
                trace_entry = {
                    "ts": time.time(),
                    "trial": idx + 1,
                    "k_res": k_res,
                    "objective": metrics.get("objective"),
                    "corr": metrics.get("corr"),
                    "lag": metrics.get("lag"),
                    "mode": args.mode,
                }
                _append_bayes_trace(trace_path, trace_entry)
        if not results:
            print("[tune] no results produced.")
            return 0
        best = max(results, key=lambda r: (r.get("objective") if r.get("objective") is not None else float("-inf")))
        print(f"[tune] best => {best}")
        if nightly_path.exists():
            report = json.loads(nightly_path.read_text(encoding="utf-8"))
            suggestion = report.setdefault("tuning_suggestion", {})
            suggestion["k_res"] = best.get("k_res")
            nightly_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[tune] tuning_suggestion.k_res -> {best.get('k_res')}")
        if args.persist:
            _update_resonance_cfg(cfg_path, best.get("k_res", 0.0), logs)
        else:
            _restore_cfg(cfg_path, original_cfg)
    finally:
        if not args.persist:
            _restore_cfg(cfg_path, original_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
