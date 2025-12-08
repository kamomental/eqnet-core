"""Voice profile definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class VoiceProfile:
    """Configuration describing how to render speech."""

    id: str
    engine: str
    speaker_id: Optional[int | str]
    base_speed: float = 1.0
    base_pitch: float = 0.0
    default_style: str = "neutral"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "engine": self.engine,
            "speaker_id": self.speaker_id,
            "base_speed": self.base_speed,
            "base_pitch": self.base_pitch,
            "default_style": self.default_style,
        }
