"""Session state containers for Heart OS."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from eqnet.runtime.state import QualiaState

from .session_config import ConversationMode, VoiceStyle, HeartOSSessionConfig


@dataclass(slots=True)
class ViewerSessionState:
    """Per-viewer session context used during live runs."""

    viewer_id: str
    first_seen: datetime
    last_comment: datetime
    comment_count: int = 0
    history: List[str] = field(default_factory=list)
    last_emotion: Optional[str] = None
    last_addressed: Optional[datetime] = None

    def record_comment(self, text: str, *, ts: datetime, emotion: Optional[str]) -> None:
        self.comment_count += 1
        self.last_comment = ts
        if text:
            self.history.append(text)
        if emotion:
            self.last_emotion = emotion


@dataclass(slots=True)
class SessionState:
    """Runtime session state flowing through the Heart OS runner."""

    session_id: str
    persona_id: str
    conversation_mode: ConversationMode
    voice_style: VoiceStyle
    voice_style_ttl: int = 0
    viewer_sessions: Dict[str, ViewerSessionState] = field(default_factory=dict)
    qualia_state: Optional[QualiaState] = None
    qualia_history: List[QualiaState] = field(default_factory=list)
    latest_emotion_tag: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: HeartOSSessionConfig,
        *,
        voice_style_ttl: int = 0,
    ) -> "SessionState":
        return cls(
            session_id=config.session_id,
            persona_id=config.persona_id,
            conversation_mode=config.conversation_mode,
            voice_style=config.default_voice_style,
            voice_style_ttl=voice_style_ttl,
        )

    def set_voice_style(self, voice_style: VoiceStyle, ttl: int) -> None:
        """Apply a temporary voice style override."""

        self.voice_style = voice_style
        self.voice_style_ttl = max(0, int(ttl))

    def tick_voice_style(self) -> None:
        """Decrement TTL and reset to normal when it expires."""

        if self.voice_style_ttl <= 0:
            return
        self.voice_style_ttl -= 1
        if self.voice_style_ttl <= 0:
            self.voice_style = "normal"

    def update_qualia(self, state: QualiaState, *, emotion_tag: Optional[str]) -> None:
        self.qualia_state = state
        self.qualia_history.append(state)
        self.latest_emotion_tag = emotion_tag
