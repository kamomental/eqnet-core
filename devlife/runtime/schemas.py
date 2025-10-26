"""Episode schemas backed by pydantic."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EpisodeRecord(BaseModel):
    t: float
    stage: Optional[str] = None
    tokens: List[int] = Field(default_factory=list)
    link_strength: float = 0.0
    self_event: float = 0.0
    other_event: float = 0.0
    conflict: float = 0.0
    mood: List[float] = Field(default_factory=list)
    plan: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, float] = Field(default_factory=dict)
    taste: Dict[str, Any] = Field(default_factory=dict)
    story: Dict[str, Any] = Field(default_factory=dict)
    tom: Dict[str, Any] = Field(default_factory=dict)
    value: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
