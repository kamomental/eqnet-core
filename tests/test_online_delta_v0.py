from __future__ import annotations

import json
from pathlib import Path

from eqnet.runtime.online_delta_v0 import (
    ONLINE_DELTA_FILE,
    apply_online_deltas,
    load_online_deltas,
    select_online_deltas,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_online_delta_loader_fail_closed_on_missing_ttl(tmp_path: Path) -> None:
    delta_path = tmp_path / ONLINE_DELTA_FILE
    _write_jsonl(
        delta_path,
        [
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "delta_id": "od-1",
                "priority": 10,
                "condition": {"scenario_id": "commute"},
                "action": {"type": "FORCE_HUMAN_CONFIRM", "payload": {}},
            }
        ],
    )

    got = load_online_deltas(tmp_path, now_ms=1500)
    assert got == []


def test_online_delta_loader_discards_expired(tmp_path: Path) -> None:
    delta_path = tmp_path / ONLINE_DELTA_FILE
    _write_jsonl(
        delta_path,
        [
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 100,
                "delta_id": "od-1",
                "priority": 10,
                "condition": {"scenario_id": "commute"},
                "action": {"type": "FORCE_HUMAN_CONFIRM", "payload": {}},
            }
        ],
    )

    got = load_online_deltas(tmp_path, now_ms=1200)
    assert got == []


def test_online_delta_apply_only_when_condition_matches(tmp_path: Path) -> None:
    delta_path = tmp_path / ONLINE_DELTA_FILE
    _write_jsonl(
        delta_path,
        [
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 10000,
                "delta_id": "od-match",
                "priority": 10,
                "condition": {"scenario_id": "commute"},
                "action": {"type": "DISALLOW_TOOL", "payload": {"tools": ["web.fetch"]}},
            },
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 10000,
                "delta_id": "od-miss",
                "priority": 20,
                "condition": {"scenario_id": "shopping"},
                "action": {"type": "FORCE_HUMAN_CONFIRM", "payload": {}},
            },
        ],
    )
    loaded = load_online_deltas(tmp_path, now_ms=1500)
    selected = select_online_deltas(loaded, {"scenario_id": "commute"})

    out = apply_online_deltas({"gate_action": "EXECUTE"}, selected)
    assert out.get("disallow_tools") == ["web.fetch"]
    assert out.get("gate_action") == "EXECUTE"


def test_online_delta_conflict_prefers_safety(tmp_path: Path) -> None:
    delta_path = tmp_path / ONLINE_DELTA_FILE
    _write_jsonl(
        delta_path,
        [
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 10000,
                "delta_id": "od-force-confirm",
                "priority": 100,
                "condition": {},
                "action": {"type": "FORCE_HUMAN_CONFIRM", "payload": {}},
            },
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 10000,
                "delta_id": "od-block-tool",
                "priority": 90,
                "condition": {},
                "action": {"type": "DISALLOW_TOOL", "payload": {"tools": ["tool.b"]}},
            },
            {
                "schema_version": "online_delta_v0",
                "created_at_ms": 1000,
                "ttl_ms": 10000,
                "delta_id": "od-cautious-budget",
                "priority": 80,
                "condition": {},
                "action": {"type": "APPLY_CAUTIOUS_BUDGET", "payload": {"budget_profile": "cautious_budget_v1"}},
            },
        ],
    )

    loaded = load_online_deltas(tmp_path, now_ms=1500)
    selected = select_online_deltas(loaded, {})
    out = apply_online_deltas(
        {
            "gate_action": "EXECUTE",
            "allow_tools": ["tool.a", "tool.b"],
            "output_control_profile": "normal_v1",
            "budget_throttle_applied": False,
        },
        selected,
    )
    assert out["gate_action"] == "HUMAN_CONFIRM"
    assert out["budget_throttle_applied"] is True
    assert out["output_control_profile"] == "cautious_budget_v1"
    assert out["disallow_tools"] == ["tool.b"]
    assert out["allow_tools"] == ["tool.a"]
