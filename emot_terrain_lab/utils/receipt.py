# -*- coding: utf-8 -*-
"""Decision receipt logging helpers."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict


LOG_PATH = Path("logs/decisions.jsonl")


def build_receipt(
    *,
    mode: str,
    style: str,
    heartiness: float,
    controls_before: Dict[str, Any],
    controls_after: Dict[str, Any],
    mood: Dict[str, Any],
    replay: Dict[str, Any],
    green: Dict[str, Any],
    norms: Dict[str, Any],
    community: Dict[str, Any],
    value: Dict[str, Any],
    organism: Dict[str, Any],
    engine_trace: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a structured record of a hub decision."""
    return {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "mode": mode,
        "style": style,
        "heartiness": float(heartiness),
        "controls_before": controls_before,
        "controls_after": controls_after,
        "mood": mood,
        "replay": replay,
        "green": green,
        "norms": norms,
        "community": community,
        "value": value,
        "organism": organism,
        "engine_trace": engine_trace,
    }


def log_receipt(receipt: Dict[str, Any]) -> None:
    """Append decision receipts to disk."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False) + "\n")


__all__ = ["build_receipt", "log_receipt", "LOG_PATH"]
