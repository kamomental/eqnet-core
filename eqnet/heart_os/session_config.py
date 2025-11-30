"""Session-level configuration helpers for Heart OS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

ConversationMode = Literal["stream", "meet"]
VoiceStyle = Literal["normal", "whisper"]


@dataclass(slots=True)
class HeartOSSessionConfig:
    """Lightweight configuration dataclass for a Heart OS session.  
    
    The runner expects conversation mode / voice style to be supplied at
    boot so downstream layers (memory, tone, prompts) can make consistent
    decisions.  ``metadata`` is intentionally loose to keep the type usable
    during early refactors.
    """

    session_id: str
    persona_id: str
    conversation_mode: ConversationMode = "stream"
    default_voice_style: VoiceStyle = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_mode(self, mode: ConversationMode) -> "HeartOSSessionConfig":
        """Return a copy updated with ``mode``."""

        return HeartOSSessionConfig(
            session_id=self.session_id,
            persona_id=self.persona_id,
            conversation_mode=mode,
            default_voice_style=self.default_voice_style,
            metadata=dict(self.metadata),
        )

    def with_voice_style(self, voice_style: VoiceStyle) -> "HeartOSSessionConfig":
        """Return a copy updated with ``voice_style`` as the default."""

        return HeartOSSessionConfig(
            session_id=self.session_id,
            persona_id=self.persona_id,
            conversation_mode=self.conversation_mode,
            default_voice_style=voice_style,
            metadata=dict(self.metadata),
        )
