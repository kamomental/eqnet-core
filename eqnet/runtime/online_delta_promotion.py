from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_POLICY: Dict[str, Any] = {
    "schema_version": "online_delta_promotion_v0",
    "window_turns": 200,
    "min_samples": 20,
    "min_success_delta": 0.05,
    "min_cost_delta": 0.02,
    "max_block_rate": 0.8,
    "max_forced_confirm_rate": 0.5,
    "success_signal_path": "prospection.accepted",
    "cost_signal_path": "cost",
    "max_candidates": 10,
}


@dataclass(frozen=True)
class PromotionDecision:
    online_delta_id: str
    status: str
    reason_codes: List[str]
    sample_count: int
    success_rate: float
    success_rate_baseline: float
    success_delta: float
    mean_cost: float | None
    mean_cost_baseline: float | None
    cost_delta: float | None
    block_rate: float
    forced_confirm_rate: float
    top_action_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "online_delta_id": self.online_delta_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "sample_count": self.sample_count,
            "success_rate": round(self.success_rate, 6),
            "success_rate_baseline": round(self.success_rate_baseline, 6),
            "success_delta": round(self.success_delta, 6),
            "mean_cost": (round(self.mean_cost, 6) if isinstance(self.mean_cost, (int, float)) else None),
            "mean_cost_baseline": (
                round(self.mean_cost_baseline, 6)
                if isinstance(self.mean_cost_baseline, (int, float))
                else None
            ),
            "cost_delta": (round(self.cost_delta, 6) if isinstance(self.cost_delta, (int, float)) else None),
            "block_rate": round(self.block_rate, 6),
            "forced_confirm_rate": round(self.forced_confirm_rate, 6),
            "top_action_type": self.top_action_type,
        }


def load_promotion_policy(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_POLICY)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return dict(DEFAULT_POLICY)
    raw = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(raw, Mapping):
        return dict(DEFAULT_POLICY)
    out = dict(DEFAULT_POLICY)
    out.update(dict(raw))
    return out


def summarize_online_delta_effectiveness(
    records: List[Dict[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    window_turns = int(policy.get("window_turns", DEFAULT_POLICY["window_turns"]) or DEFAULT_POLICY["window_turns"])
    max_candidates = int(policy.get("max_candidates", DEFAULT_POLICY["max_candidates"]) or DEFAULT_POLICY["max_candidates"])
    success_signal_path = str(policy.get("success_signal_path") or DEFAULT_POLICY["success_signal_path"])
    cost_signal_path = str(policy.get("cost_signal_path") or DEFAULT_POLICY["cost_signal_path"])
    ordered = sorted(records, key=lambda row: int(_as_int(_get_path(row, "timestamp_ms")) or 0))
    if window_turns > 0 and len(ordered) > window_turns:
        ordered = ordered[-window_turns:]

    baseline_successes: List[float] = []
    baseline_costs: List[float] = []
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in ordered:
        policy_obs = _hub_policy_obs(row)
        online_applied = bool(policy_obs.get("online_delta_applied"))
        success = _as_success(_get_path(row, success_signal_path))
        cost = _as_float(_get_path(row, cost_signal_path))
        ids = [str(x) for x in (policy_obs.get("online_delta_ids") or []) if isinstance(x, str) and x]
        if online_applied and ids:
            for online_delta_id in ids:
                bucket = grouped.setdefault(
                    online_delta_id,
                    {
                        "samples": 0,
                        "success_values": [],
                        "cost_values": [],
                        "block_count": 0,
                        "forced_count": 0,
                        "action_types": {},
                    },
                )
                bucket["samples"] += 1
                if success is not None:
                    bucket["success_values"].append(success)
                if cost is not None:
                    bucket["cost_values"].append(cost)
                if _is_tool_blocked(row):
                    bucket["block_count"] += 1
                if _is_forced_confirm(row):
                    bucket["forced_count"] += 1
                for action_type in policy_obs.get("online_delta_action_types") or []:
                    if not isinstance(action_type, str) or not action_type:
                        continue
                    action_counts = bucket["action_types"]
                    action_counts[action_type] = int(action_counts.get(action_type, 0)) + 1
            continue
        if success is not None:
            baseline_successes.append(success)
        if cost is not None:
            baseline_costs.append(cost)

    baseline_success_rate = _mean(baseline_successes) or 0.0
    baseline_cost = _mean(baseline_costs)
    candidates: List[Dict[str, Any]] = []
    for online_delta_id, bucket in grouped.items():
        samples = int(bucket["samples"])
        success_rate = _mean(bucket["success_values"]) or 0.0
        mean_cost = _mean(bucket["cost_values"])
        block_rate = float(bucket["block_count"] / samples) if samples > 0 else 0.0
        forced_rate = float(bucket["forced_count"] / samples) if samples > 0 else 0.0
        action_type = _top_key(bucket["action_types"])
        candidates.append(
            {
                "online_delta_id": online_delta_id,
                "sample_count": samples,
                "success_rate": success_rate,
                "success_rate_baseline": baseline_success_rate,
                "success_delta": success_rate - baseline_success_rate,
                "mean_cost": mean_cost,
                "mean_cost_baseline": baseline_cost,
                "cost_delta": (mean_cost - baseline_cost) if mean_cost is not None and baseline_cost is not None else None,
                "block_rate": block_rate,
                "forced_confirm_rate": forced_rate,
                "top_action_type": action_type,
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item.get("success_delta") or 0.0),
            -float(item.get("cost_delta") or 0.0),
            int(item.get("sample_count") or 0),
        ),
        reverse=True,
    )
    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return {
        "baseline": {
            "success_rate": baseline_success_rate,
            "mean_cost": baseline_cost,
            "sample_count_success": len(baseline_successes),
            "sample_count_cost": len(baseline_costs),
        },
        "candidates": candidates,
    }


def decide_promotions(
    summary: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> List[PromotionDecision]:
    min_samples = int(policy.get("min_samples", DEFAULT_POLICY["min_samples"]) or DEFAULT_POLICY["min_samples"])
    min_success_delta = float(policy.get("min_success_delta", DEFAULT_POLICY["min_success_delta"]) or DEFAULT_POLICY["min_success_delta"])
    min_cost_delta = float(policy.get("min_cost_delta", DEFAULT_POLICY["min_cost_delta"]) or DEFAULT_POLICY["min_cost_delta"])
    max_block_rate = float(policy.get("max_block_rate", DEFAULT_POLICY["max_block_rate"]) or DEFAULT_POLICY["max_block_rate"])
    max_forced = float(
        policy.get("max_forced_confirm_rate", DEFAULT_POLICY["max_forced_confirm_rate"])
        or DEFAULT_POLICY["max_forced_confirm_rate"]
    )

    out: List[PromotionDecision] = []
    for item in summary.get("candidates") or []:
        if not isinstance(item, Mapping):
            continue
        online_delta_id = str(item.get("online_delta_id") or "")
        if not online_delta_id:
            continue
        samples = int(_as_int(item.get("sample_count")) or 0)
        success_delta = float(_as_float(item.get("success_delta")) or 0.0)
        block_rate = float(_as_float(item.get("block_rate")) or 0.0)
        forced_rate = float(_as_float(item.get("forced_confirm_rate")) or 0.0)
        cost_delta_raw = _as_float(item.get("cost_delta"))
        improves_cost = isinstance(cost_delta_raw, float) and cost_delta_raw <= -abs(min_cost_delta)
        improves_success = success_delta >= min_success_delta
        reason_codes: List[str] = []
        status = "REJECTED"
        if samples < min_samples:
            reason_codes.append("PROMOTION_REJECT_MIN_SAMPLES")
        if block_rate > max_block_rate:
            reason_codes.append("PROMOTION_REJECT_BLOCK_RATE_HIGH")
        if forced_rate > max_forced:
            reason_codes.append("PROMOTION_REJECT_FORCED_CONFIRM_RATE_HIGH")
        if not improves_success and not improves_cost:
            reason_codes.append("PROMOTION_REJECT_NO_EFFECT")
        if not reason_codes:
            status = "PROMOTE"
            reason_codes.append("PROMOTION_ACCEPT_EFFECTIVE")
        out.append(
            PromotionDecision(
                online_delta_id=online_delta_id,
                status=status,
                reason_codes=reason_codes,
                sample_count=samples,
                success_rate=float(_as_float(item.get("success_rate")) or 0.0),
                success_rate_baseline=float(_as_float(item.get("success_rate_baseline")) or 0.0),
                success_delta=success_delta,
                mean_cost=_as_float(item.get("mean_cost")),
                mean_cost_baseline=_as_float(item.get("mean_cost_baseline")),
                cost_delta=cost_delta_raw,
                block_rate=block_rate,
                forced_confirm_rate=forced_rate,
                top_action_type=str(item.get("top_action_type") or ""),
            )
        )
    return out


def append_rule_delta_promotions(
    *,
    rule_delta_path: Path,
    decisions: Iterable[PromotionDecision],
    day_key: str,
    timestamp_ms: int,
) -> Dict[str, Any]:
    existing_keys = set(_existing_promotion_keys(rule_delta_path))
    appended: List[str] = []
    skipped: List[Dict[str, Any]] = []
    rule_delta_path.parent.mkdir(parents=True, exist_ok=True)
    with rule_delta_path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            if decision.status != "PROMOTE":
                skipped.append(
                    {
                        "online_delta_id": decision.online_delta_id,
                        "status": decision.status,
                        "reason_codes": list(decision.reason_codes),
                    }
                )
                continue
            promotion_key = f"online_delta:{decision.online_delta_id}"
            if promotion_key in existing_keys:
                skipped.append(
                    {
                        "online_delta_id": decision.online_delta_id,
                        "status": "SKIPPED_DUPLICATE",
                        "reason_codes": ["PROMOTION_SKIP_ALREADY_EXISTS"],
                    }
                )
                continue
            payload = {
                "schema_version": "rule_delta.v0",
                "timestamp_ms": int(timestamp_ms),
                "day_key": str(day_key),
                "promotion_key": promotion_key,
                "source": "online_delta_promotion_v0",
                "operation": "add",
                "target_online_delta_id": decision.online_delta_id,
                "action": _promoted_action_payload(decision.top_action_type),
                "evidence": decision.to_dict(),
                "reason_codes": list(decision.reason_codes),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            appended.append(decision.online_delta_id)
            existing_keys.add(promotion_key)
    return {"appended_online_delta_ids": appended, "decisions_skipped": skipped}


def _existing_promotion_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            key = row.get("promotion_key")
            if isinstance(key, str) and key:
                out.append(key)
    return out


def _promoted_action_payload(action_type: str) -> Dict[str, Any]:
    if action_type == "DISALLOW_TOOL":
        return {"type": "DISALLOW_TOOL"}
    if action_type == "FORCE_HUMAN_CONFIRM":
        return {"type": "FORCE_HUMAN_CONFIRM"}
    return {"type": "APPLY_CAUTIOUS_BUDGET", "payload": {"budget_profile": "cautious_budget_v1"}}


def _hub_policy_obs(row: Mapping[str, Any]) -> Dict[str, Any]:
    policy = row.get("policy")
    if not isinstance(policy, Mapping):
        return {}
    observations = policy.get("observations")
    if not isinstance(observations, Mapping):
        return {}
    hub = observations.get("hub")
    return dict(hub) if isinstance(hub, Mapping) else {}


def _get_path(row: Mapping[str, Any], path: str) -> Any:
    node: Any = row
    for key in str(path).split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _as_success(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"PASS", "HELPED", "SUCCESS", "ACCEPTED"}:
            return 1.0
        if v in {"FAIL", "HARMED", "REJECTED"}:
            return 0.0
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _is_tool_blocked(row: Mapping[str, Any]) -> bool:
    if str(row.get("event_type") or "") == "tool_call_blocked":
        return True
    reasons = row.get("reason_codes")
    if isinstance(reasons, list):
        return any(str(code) == "ONLINE_DELTA_TOOL_BLOCKED" for code in reasons)
    return False


def _is_forced_confirm(row: Mapping[str, Any]) -> bool:
    if str(row.get("event_type") or "") == "forced_gate_action":
        return str(row.get("forced_gate_action") or "") == "HUMAN_CONFIRM"
    policy_obs = _hub_policy_obs(row)
    return str(policy_obs.get("forced_gate_action") or "") == "HUMAN_CONFIRM"


def _top_key(counter_like: Mapping[str, Any]) -> str:
    best_key = ""
    best_value = -1
    for key, value in counter_like.items():
        count = _as_int(value) or 0
        if count > best_value:
            best_key = str(key)
            best_value = count
    return best_key

