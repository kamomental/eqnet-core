"""Loader for style presets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from eqnet.voice.style_state import UtteranceStyleState


def load_style_presets(path: str | Path) -> Dict[str, UtteranceStyleState]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    presets_raw = data.get("presets", {})
    presets: Dict[str, UtteranceStyleState] = {}
    for preset_id, vals in presets_raw.items():
        presets[preset_id] = UtteranceStyleState(**vals)
    return presets
