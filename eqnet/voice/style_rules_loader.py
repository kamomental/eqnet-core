"""Loader for style rules (fillers/laughter/etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_style_rules(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("styles", {})
