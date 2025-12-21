#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _bit(value: Any, default: int = 0) -> int:
    try:
        return 1 if int(value) else 0
    except Exception:
        return default


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_receipt(event: Dict[str, Any]) -> Dict[str, Any]:
    meta = event.get("meta") or {}
    receipt = meta.get("receipt") or {}
    if isinstance(receipt, dict):
        return receipt
    return {}


def _narrative_flag(receipt: Dict[str, Any], qualia: Dict[str, Any]) -> Optional[int]:
    for key in ("conscious_episode", "episode", "conscious", "response"):
        block = receipt.get(key)
        if isinstance(block, dict) and "narrative" in block:
            narrative = block["narrative"]
            return 1 if isinstance(narrative, str) and narrative.strip() else 0

    gate_block = receipt.get("qualia_gate")
    if isinstance(gate_block, dict) and "suppress_narrative" in gate_block:
        return 0 if gate_block.get("suppress_narrative") else 1

    if "unconscious_success" in qualia and _bit(qualia.get("unconscious_success")):
        return 0

    return None


def summarize(events_path: Path) -> Dict[str, Any]:
    turns = 0
    u_sum = 0.0
    m_sum = 0.0
    load_sum = 0.0
    gate_open_sum = 0
    unconscious_sum = 0
    narr_known = 0
    narr_attached_sum = 0

    for event in _iter_jsonl(events_path):
        receipt = _get_receipt(event)
        qualia = receipt.get("qualia") or {}
        if not isinstance(qualia, dict):
            qualia = {}

        turns += 1
        u_sum += _safe_float(qualia.get("u_t"))
        m_sum += _safe_float(qualia.get("m_t"))
        load_sum += _safe_float(qualia.get("load") or qualia.get("load_t"))
        gate_open_sum += _bit(qualia.get("a_t"))
        unconscious_sum += _bit(qualia.get("unconscious_success"))

        narr_flag = _narrative_flag(receipt, qualia)
        if narr_flag is not None:
            narr_known += 1
            narr_attached_sum += narr_flag

    def mean(total: float) -> float:
        return total / turns if turns else 0.0

    stats: Dict[str, Any] = {
        "turns": turns,
        "qualia_u_mean": mean(u_sum),
        "qualia_m_mean": mean(m_sum),
        "qualia_load_mean": mean(load_sum),
        "qualia_gate_open_rate": mean(gate_open_sum),
        "qualia_unconscious_success_rate": mean(unconscious_sum),
    }
    if narr_known:
        stats["narrative_attach_pct"] = 100.0 * narr_attached_sum / narr_known
        stats["narrative_attach_known_turns"] = narr_known
    else:
        stats["narrative_attach_pct"] = None
        stats["narrative_attach_known_turns"] = 0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, help="Path to events jsonl file")
    args = parser.parse_args()

    stats = summarize(Path(args.events))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
