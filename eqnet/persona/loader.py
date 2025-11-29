"""Persona configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PersonaConfig:
    """Represents a persona profile loaded from YAML."""

    raw: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    visual: Dict[str, Any] = field(default_factory=dict)
    speech: Dict[str, Any] = field(default_factory=dict)
    qfs: Dict[str, Any] = field(default_factory=dict)
    diary_style: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)

    @property
    def persona_id(self) -> Optional[str]:
        return self.meta.get("id")

    @property
    def display_name(self) -> Optional[str]:
        return self.meta.get("display_name")


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_persona(path: Path) -> PersonaConfig:
    """Load a persona YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Persona file {path} does not contain a mapping")
    return PersonaConfig(
        raw=data,
        meta=_section(data, "meta"),
        visual=_section(data, "visual"),
        speech=_section(data, "speech"),
        qfs=_section(data, "qfs"),
        diary_style=_section(data, "diary_style"),
        safety=_section(data, "safety"),
    )


def load_persona_from_dir(root: Path, persona_id: str) -> Optional[PersonaConfig]:
    """Load persona YAML from a directory if present."""

    for ext in (".yaml", ".yml"):
        candidate = root / f"{persona_id}{ext}"
        if candidate.exists():
            return load_persona(candidate)
    return None
