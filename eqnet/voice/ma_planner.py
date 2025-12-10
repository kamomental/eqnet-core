from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .context import VoiceContext


@dataclass
class MaPlan:
    silence_ms: int
    insert_filler: bool
    filler_text: str | None


class MaPlanner(ABC):
    """Voice Field §2 / §7 timing planner interface."""

    @abstractmethod
    def plan(self, ctx: VoiceContext) -> MaPlan:
        ...
