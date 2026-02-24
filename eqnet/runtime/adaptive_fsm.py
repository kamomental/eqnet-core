from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


class AdaptiveMode(str, Enum):
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class AdaptiveStateSnapshot:
    mode: AdaptiveMode
    evidence: Dict[str, float]
    reason_codes: List[str]
    policy_fingerprint: str
    policy_version: str
    policy_source: str


def load_fsm_policy(path: Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "fsm_policy_v0.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("fsm policy must be a mapping")
    if str(raw.get("schema_version") or "") != "fsm_policy_v0":
        raise ValueError("fsm policy schema_version must be fsm_policy_v0")
    raw["policy_source"] = str(path.as_posix())
    return raw


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    canonical = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def reduce_mode_sequence(
    events: List[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> List[AdaptiveStateSnapshot]:
    active_policy = policy if policy is not None else load_fsm_policy()
    mode = _safe_mode(str(active_policy.get("initial_mode") or "STABLE"))
    fp = policy_fingerprint(active_policy)
    version = str(active_policy.get("policy_version") or "fsm_policy_v0")
    source = str(active_policy.get("policy_source") or "configs/fsm_policy_v0.yaml")
    transitions = active_policy.get("transitions")
    if not isinstance(transitions, list):
        transitions = []

    snapshots: List[AdaptiveStateSnapshot] = []
    for event in events:
        metrics = _extract_metrics(event)
        next_mode = mode
        reason_codes: List[str] = []
        for row in transitions:
            if not isinstance(row, dict):
                continue
            from_mode = _safe_mode(str(row.get("from") or mode.value))
            if from_mode != mode:
                continue
            all_conditions = row.get("all")
            if not isinstance(all_conditions, list):
                continue
            if _conditions_match(all_conditions, metrics):
                next_mode = _safe_mode(str(row.get("to") or mode.value))
                reason = str(row.get("reason_code") or "").strip()
                if reason:
                    reason_codes.append(reason)
                break
        mode = next_mode
        snapshots.append(
            AdaptiveStateSnapshot(
                mode=mode,
                evidence=metrics,
                reason_codes=reason_codes,
                policy_fingerprint=fp,
                policy_version=version,
                policy_source=source,
            )
        )
    return snapshots


def reduce_latest_mode(
    events: List[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> AdaptiveStateSnapshot:
    snapshots = reduce_mode_sequence(events, policy=policy)
    if snapshots:
        return snapshots[-1]
    active_policy = policy if policy is not None else load_fsm_policy()
    return AdaptiveStateSnapshot(
        mode=_safe_mode(str(active_policy.get("initial_mode") or "STABLE")),
        evidence={},
        reason_codes=[],
        policy_fingerprint=policy_fingerprint(active_policy),
        policy_version=str(active_policy.get("policy_version") or "fsm_policy_v0"),
        policy_source=str(active_policy.get("policy_source") or "configs/fsm_policy_v0.yaml"),
    )


def _extract_metrics(event: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric_name in (
        "pending_ratio",
        "shadow_pass_rate",
        "contract_errors_ratio",
        "memory_entropy_delta",
        "repair_active",
        "repair_trigger",
        "budget_throttle_applied",
        "output_control_cautious",
    ):
        out[metric_name] = _read_metric(event, metric_name)
    return out


def _read_metric(event: Mapping[str, Any], metric_name: str) -> float:
    direct = event.get(metric_name)
    if isinstance(direct, (int, float)):
        return float(direct)
    if isinstance(direct, bool):
        return 1.0 if direct else 0.0
    metrics = event.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get(metric_name)
        if isinstance(nested, (int, float)):
            return float(nested)
        if isinstance(nested, bool):
            return 1.0 if nested else 0.0
    return 0.0


def _conditions_match(conditions: List[Any], metrics: Mapping[str, float]) -> bool:
    for condition in conditions:
        if not isinstance(condition, Mapping):
            return False
        metric = str(condition.get("metric") or "").strip()
        op = str(condition.get("op") or "").strip()
        threshold = condition.get("threshold")
        if not metric or metric not in metrics:
            return False
        try:
            value = float(metrics.get(metric, 0.0))
            bound = float(threshold)
        except (TypeError, ValueError):
            return False
        if not _compare(value, op, bound):
            return False
    return True


def _compare(value: float, op: str, bound: float) -> bool:
    if op == ">=":
        return value >= bound
    if op == "<=":
        return value <= bound
    if op == ">":
        return value > bound
    if op == "<":
        return value < bound
    if op == "==":
        return value == bound
    return False


def _safe_mode(raw: str) -> AdaptiveMode:
    normalized = raw.strip().upper()
    if normalized in AdaptiveMode.__members__:
        return AdaptiveMode[normalized]
    return AdaptiveMode.STABLE
