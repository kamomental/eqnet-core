# -*- coding: utf-8 -*-
"""Simple JSONL canary logger for assoc-kernel metrics."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


class CanaryLogger:
    """Append-only JSONL writer with UTC timestamps."""

    def __init__(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.path = path

    def write(self, record: Dict[str, Any]) -> None:
        payload = dict(record or {})
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["CanaryLogger"]
