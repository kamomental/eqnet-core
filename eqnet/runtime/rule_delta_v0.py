from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

RULE_DELTA_FILE = "rule_delta.v0.jsonl"
SCHEMA_VERSION = "rule_delta.v0"


@dataclass(frozen=True)
class RuleDeltaV0:
    rule_id: str
    op: str
    raw: Dict[str, Any]


def load_rule_deltas(state_dir: Path) -> List[RuleDeltaV0]:
    path = Path(state_dir) / RULE_DELTA_FILE
    if not path.exists():
        return []

    loaded: List[RuleDeltaV0] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            item = _normalize_rule_delta(payload)
            if item is not None:
                loaded.append(item)
    return loaded


def _normalize_rule_delta(raw: Any) -> RuleDeltaV0 | None:
    if not isinstance(raw, Mapping):
        return None
    if str(raw.get("schema_version") or "") != SCHEMA_VERSION:
        return None

    rule_id = str(raw.get("rule_id") or "")
    op = str(raw.get("op") or "")
    if not rule_id or not op:
        return None

    return RuleDeltaV0(rule_id=rule_id, op=op, raw=dict(raw))
