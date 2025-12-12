from __future__ import annotations

from enum import Enum


class TalkMode(str, Enum):
    """Canonical talk-mode labels shared across runtime + policy."""

    WATCH = "watch"
    SOOTHE = "soothe"
    ASK = "ask"
    TALK = "talk"

    @classmethod
    def from_any(cls, value: str | "TalkMode" | None) -> "TalkMode":
        if isinstance(value, TalkMode):
            return value
        if not value:
            return TalkMode.WATCH
        normalized = str(value).strip().lower()
        for mode in TalkMode:
            if mode.value == normalized or mode.name.lower() == normalized:
                return mode
        raise ValueError(f"Unknown talk mode: {value}")

    def uplift(self, other: "TalkMode") -> "TalkMode":
        """Pick the more active mode (WATCH < SOOTHE < ASK < TALK)."""

        order = {TalkMode.WATCH: 0, TalkMode.SOOTHE: 1, TalkMode.ASK: 2, TalkMode.TALK: 3}
        return self if order[self] >= order[other] else other
