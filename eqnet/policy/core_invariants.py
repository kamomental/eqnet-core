from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DEFAULT_INVARIANTS_PATH = Path(__file__).with_name("invariants.yaml")


def load_core_invariants(path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved = Path(path) if path else DEFAULT_INVARIANTS_PATH
    if not resolved.exists():
        return []
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if isinstance(data, Mapping):
        items = data.get("core_invariants")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, Mapping)]
    return []


def evaluate_core_invariants(
    trace_payload: Mapping[str, Any],
    invariants: Iterable[Mapping[str, Any]],
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for item in invariants:
        inv_id = str(item.get("id")) if item.get("id") else None
        if not inv_id:
            continue
        applies = item.get("applies_when") or {}
        if not _assert_all(trace_payload, applies):
            continue
        asserts = item.get("asserts") or {}
        results[inv_id] = _assert_all(trace_payload, asserts)
    return results


def _assert_all(trace_payload: Mapping[str, Any], expressions: Mapping[str, Any]) -> bool:
    for path, expr in expressions.items():
        value = _lookup(trace_payload, path)
        if not _evaluate_expr(value, expr):
            return False
    return True


def _lookup(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in str(dotted).split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _evaluate_expr(value: Any, expr: Any) -> bool:
    if isinstance(expr, str):
        text = expr.strip().lower()
        if text == "present":
            return value is not None
        if text == "absent":
            return value is None
        if text in {"true", "false"}:
            return bool(value) is (text == "true")
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if text.startswith(op):
                target = text[len(op) :].strip()
                return _compare(value, op, target)
    return value == expr


def _compare(value: Any, op: str, target: str) -> bool:
    try:
        expected = float(target)
        actual = float(value)
    except (TypeError, ValueError):
        actual = value
        expected = target
    if op == ">=":
        return actual >= expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == "<":
        return actual < expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    return False


__all__ = ["load_core_invariants", "evaluate_core_invariants"]
