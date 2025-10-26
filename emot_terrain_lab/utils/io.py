# -*- coding: utf-8 -*-
"""Utility helpers for lightweight file IO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append a JSON-serializable payload to the given JSONL file."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    except Exception:
        logger.exception("Failed to append JSONL payload to %s", path)


__all__ = ["append_jsonl"]
