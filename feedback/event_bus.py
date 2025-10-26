# -*- coding: utf-8 -*-
"""Simple event log for user feedback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def record_event(event: Dict[str, Any], path: str = "logs/feedback.jsonl") -> None:
    payload = dict(event)
    payload.setdefault("ts", time.time())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["record_event"]

