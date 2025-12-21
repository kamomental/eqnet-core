# -*- coding: utf-8 -*-
"""Minimal PersonaDraft stubs for hub persona manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PersonaDraft:
    name: str = "default"
    preferences: Dict[str, Any] | None = None


def persona_from_text(text: str) -> PersonaDraft:
    name = (text or "default").strip() or "default"
    return PersonaDraft(name=name, preferences={})
