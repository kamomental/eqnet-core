from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from eqnet.orchestrators.hub_adapter import run_hub_turn
from eqnet.orchestrators.runtime_adapter import run_runtime_turn
from eqnet.policy.core_invariants import (
    evaluate_core_invariants,
    load_core_invariants,
)
from eqnet.runtime.turn import CoreState, SafetyConfig

COMPARE_PATHS = [
    "boundary.score",
    "boundary.reasons",
    "self.winner",
    "self.tie_flag",
    "self.fatigue",
    "prospection.jerk",
    "prospection.temperature",
    "prospection.accepted",
    "policy.throttles",
    "qualia.before",
    "qualia.after",
]
DEFAULT_INVARIANTS_PATH = Path("eqnet/policy/invariants.yaml")
BOUNDARY_THRESHOLD = 0.7
SUPPORTED_SUFFIXES = {".json", ".jsonl"}
FATAL_INVARIANTS = {"CORE_TRACE_001", "CORE_POLICY_001", "CORE_PROSPECT_001"}


def _expand_inputs(sources: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"[compare_loops] warning: path not found: {path}")
    return files


def _pair_inputs(paths: Iterable[Path]) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    groups: dict[str, dict[str, Path]] = {}
    for path in paths:
        stem = path.stem
        lower = stem.lower()
        if lower.startswith("hub_"):
            key = stem.split("hub_", 1)[1]
            groups.setdefault(key, {})["hub"] = path
        elif lower.startswith("runtime_"):
            key = stem.split("runtime_", 1)[1]
            groups.setdefault(key, {})["runtime"] = path
        else:
            key = stem
            group = groups.setdefault(key, {})
            group.setdefault("hub", path)
            group.setdefault("runtime", path)
    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for key in sorted(groups):
        pair = groups[key]
        hub_path = pair.get("hub")
        runtime_path = pair.get("runtime")
        if not hub_path or not runtime_path:
            print(f"[compare_loops] warning: missing pair for '{key}', skipping")
            missing.append(key)
            continue
        pairs.append((key, hub_path, runtime_path))
        
    return pairs, missing


def _load_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict) and "events" in payload:
            entries = payload["events"]
        else:
            entries = [payload]
    events: list[dict[str, Any]] = []
    for item in entries:
        if isinstance(item, dict):
            events.append(dict(item))
    return events


def _prepare_events(
    events: list[dict[str, Any]],
    scenario_id: str,
    *,
    deterministic: bool,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        evt = dict(event)
        if deterministic:
            evt["scenario_id"] = scenario_id
            evt["turn_id"] = f"{scenario_id}-turn-{idx:04d}"
            evt["timestamp_ms"] = idx * 1000
            evt["seed"] = idx + 1
        else:
            evt.setdefault("scenario_id", scenario_id)
            evt.setdefault("turn_id", evt.get("turn_id", f"turn-{idx:04d}"))
            evt.setdefault("timestamp_ms", evt.get("timestamp_ms", idx * 1000))
            evt.setdefault("seed", evt.get("seed", idx + 1))
        prepared.append(evt)
    return prepared


def _read_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _get_path(payload: dict[str, Any] | None, dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _diff_records(
    hub: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    hub_invariants: list[dict[str, bool]],
    runtime_invariants: list[dict[str, bool]],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for idx, (h_record, r_record) in enumerate(zip_longest(hub, runtime, fillvalue=None)):
        entry: dict[str, Any] = {"turn_index": idx}
        for path in COMPARE_PATHS:
            entry[path] = {
                "hub": _get_path(h_record, path),
                "runtime": _get_path(r_record, path),
            }
        entry["invariants"] = {
            "hub": hub_invariants[idx] if idx < len(hub_invariants) else {},
            "runtime": runtime_invariants[idx] if idx < len(runtime_invariants) else {},
        }
        diffs.append(entry)
    return diffs


def _summarize_diffs(diffs: list[dict[str, Any]]) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for entry in diffs:
        for path in COMPARE_PATHS:
            values = entry.get(path)
            if isinstance(values, dict) and values.get("hub") != values.get("runtime"):
                summary[path] += 1
    return dict(summary)


def _summarize_invariants(
    hub_results: list[dict[str, bool]],
    runtime_results: list[dict[str, bool]],
) -> dict[str, dict[str, Counter[str]]]:
    summary = {
        "hub": {"fatal": Counter(), "warn": Counter()},
        "runtime": {"fatal": Counter(), "warn": Counter()},
    }
    for label, results in (("hub", hub_results), ("runtime", runtime_results)):
        for record in results:
            for inv_id, ok in record.items():
                if ok:
                    continue
                severity = "fatal" if inv_id in FATAL_INVARIANTS else "warn"
                summary[label][severity][inv_id] += 1
    return summary


def _boundary_spans(records: list[dict[str, Any]], threshold: float) -> list[int]:
    spans: list[int] = []
    current = 0
    for record in records:
        score = _get_path(record, "boundary.score")
        if isinstance(score, (int, float)) and float(score) >= threshold:
            current += 1
        else:
            if current > 0:
                spans.append(current)
                current = 0
    if current > 0:
        spans.append(current)
    return spans


def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    boundary_scores = [
        float(score)
        for score in (_get_path(r, "boundary.score") for r in records)
        if isinstance(score, (int, float))
    ]
    boundary_high = sum(1 for score in boundary_scores if score >= BOUNDARY_THRESHOLD)
    spans = _boundary_spans(records, BOUNDARY_THRESHOLD)
    span_total = sum(spans)
    prospection_states = [
        _get_path(r, "prospection.accepted") for r in records
    ]
    prospection_total = sum(1 for state in prospection_states if state is not None)
    prospection_reject = sum(1 for state in prospection_states if state is False)
    throttle_count = 0
    for record in records:
        cap = _get_path(record, "policy.throttles.directiveness_cap")
        if isinstance(cap, (int, float)) and float(cap) < 1.0:
            throttle_count += 1
    metrics = {
        "count": count,
        "average_boundary": sum(boundary_scores) / len(boundary_scores) if boundary_scores else 0.0,
        "boundary_high_count": boundary_high,
        "boundary_high_rate": boundary_high / count if count else 0.0,
        "boundary_span_max": max(spans) if spans else 0,
        "boundary_span_count": len(spans),
        "boundary_span_total_length": span_total,
        "prospection_total": prospection_total,
        "prospection_reject_count": prospection_reject,
        "prospection_reject_rate": prospection_reject / prospection_total if prospection_total else 0.0,
        "offer_throttle_count": throttle_count,
        "offer_throttle_rate": throttle_count / count if count else 0.0,
    }
    return metrics


def _init_metric_totals() -> dict[str, Any]:
    return {
        "count": 0,
        "boundary_sum": 0.0,
        "boundary_high": 0,
        "boundary_span_max": 0,
        "boundary_span_total": 0,
        "boundary_span_count": 0,
        "prospection_total": 0,
        "prospection_reject": 0,
        "throttle_count": 0,
    }


def _update_metric_totals(totals: dict[str, Any], metrics: dict[str, Any]) -> None:
    count = metrics.get("count", 0)
    totals["count"] += count
    totals["boundary_sum"] += metrics.get("average_boundary", 0.0) * count
    totals["boundary_high"] += metrics.get("boundary_high_count", 0)
    totals["boundary_span_max"] = max(totals["boundary_span_max"], metrics.get("boundary_span_max", 0))
    totals["boundary_span_total"] += metrics.get("boundary_span_total_length", 0)
    totals["boundary_span_count"] += metrics.get("boundary_span_count", 0)
    totals["prospection_total"] += metrics.get("prospection_total", 0)
    totals["prospection_reject"] += metrics.get("prospection_reject_count", 0)
    totals["throttle_count"] += metrics.get("offer_throttle_count", 0)


def _finalize_metric_totals(totals: dict[str, Any]) -> dict[str, Any]:
    count = totals["count"] or 1
    span_count = totals["boundary_span_count"] or 1
    prospection_total = totals["prospection_total"] or 1
    return {
        "count": totals["count"],
        "average_boundary": totals["boundary_sum"] / count,
        "boundary_high_rate": totals["boundary_high"] / count,
        "boundary_high_count": totals["boundary_high"],
        "boundary_span_max": totals["boundary_span_max"],
        "boundary_span_avg": totals["boundary_span_total"] / span_count,
        "prospection_reject_rate": totals["prospection_reject"] / prospection_total,
        "prospection_reject_count": totals["prospection_reject"],
        "offer_throttle_rate": totals["throttle_count"] / count,
        "offer_throttle_count": totals["throttle_count"],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_comparisons(
    sources: Sequence[str],
    *,
    out_dir: Path,
    invariants_path: Path,
    deterministic: bool,
    strict_pairs: bool,
) -> int:
    files = _expand_inputs(sources)
    pairs, missing_pairs = _pair_inputs(files)
    if not pairs:
        print("[compare_loops] error: no scenario inputs were found", file=sys.stderr)
        return 1

    invariants = load_core_invariants(invariants_path)
    safety = SafetyConfig()

    aggregate_diff = Counter()
    aggregate_invariants = {
        "hub": {"fatal": Counter(), "warn": Counter()},
        "runtime": {"fatal": Counter(), "warn": Counter()},
    }
    aggregate_metrics = {"hub": _init_metric_totals(), "runtime": _init_metric_totals()}
    scenario_reports: list[dict[str, Any]] = []

    for key, hub_path, runtime_path in pairs:
        scenario_dir = out_dir / key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        hub_trace = scenario_dir / "trace_hub.jsonl"
        runtime_trace = scenario_dir / "trace_runtime.jsonl"
        for file in (hub_trace, runtime_trace):
            if file.exists():
                file.unlink()

        hub_events = _prepare_events(_load_events(hub_path), key, deterministic=deterministic)
        runtime_events = _prepare_events(_load_events(runtime_path), key, deterministic=deterministic)

        for event in hub_events:
            run_hub_turn(event, CoreState(), safety, hub_trace)
        for event in runtime_events:
            run_runtime_turn(event, CoreState(), safety, runtime_trace)

        hub_records = _read_traces(hub_trace)
        runtime_records = _read_traces(runtime_trace)
        hub_invariants = [evaluate_core_invariants(record, invariants) for record in hub_records]
        runtime_invariants = [evaluate_core_invariants(record, invariants) for record in runtime_records]
        diffs = _diff_records(hub_records, runtime_records, hub_invariants, runtime_invariants)
        diff_summary = _summarize_diffs(diffs)
        invariant_summary = _summarize_invariants(hub_invariants, runtime_invariants)
        metrics = {
            "hub": _compute_metrics(hub_records),
            "runtime": _compute_metrics(runtime_records),
        }

        aggregate_diff.update(diff_summary)
        for label in ("hub", "runtime"):
            for severity in ("fatal", "warn"):
                aggregate_invariants[label][severity].update(invariant_summary[label][severity])
            _update_metric_totals(aggregate_metrics[label], metrics[label])

        scenario_reports.append(
            {
                "id": key,
                "paths": {"hub": str(hub_path), "runtime": str(runtime_path)},
                "diff_summary": diff_summary,
                "invariants_summary": {
                    loop: {severity: dict(counter) for severity, counter in inv.items()}
                    for loop, inv in invariant_summary.items()
                },
                "metrics": metrics,
                "diffs": diffs,
            }
        )

    _write_json(
        out_dir / "trace_diff.json",
        {
            "scenarios": scenario_reports,
            "aggregate": {
                "diff_summary": dict(aggregate_diff),
                "missing_pairs": missing_pairs,
            },
        },
    )
    _write_json(
        out_dir / "invariant_summary.json",
        {
            loop: {severity: dict(counter) for severity, counter in summary.items()}
            for loop, summary in aggregate_invariants.items()
        },
    )
    _write_json(
        out_dir / "metric_summary.json",
        {k: _finalize_metric_totals(v) for k, v in aggregate_metrics.items()},
    )

    print(f"Processed {len(scenario_reports)} scenario pairs. Output -> {out_dir}")
    fatal_present = any(aggregate_invariants[label]["fatal"] for label in aggregate_invariants)
    missing_error = strict_pairs and bool(missing_pairs)
    if missing_error:
        print(
            f"[compare_loops] error: missing pairs detected: {', '.join(missing_pairs)}",
            file=sys.stderr,
        )
    return 1 if fatal_present or missing_error else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hub/runtime traces from scenarios")
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="Scenario files or directories (supports hub_*/runtime_* pairing)",
    )
    parser.add_argument("--out", default="sim_out", help="Directory for trace outputs")
    parser.add_argument(
        "--invariants",
        default=str(DEFAULT_INVARIANTS_PATH),
        help="Path to invariants YAML",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic scenario_id/turn_id/seed/timestamp",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Exit with failure if hub/runtime pairs are missing",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    exit_code = run_comparisons(
        args.scenarios,
        out_dir=out_dir,
        invariants_path=Path(args.invariants),
        deterministic=args.deterministic,
        strict_pairs=args.strict_pairs,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
