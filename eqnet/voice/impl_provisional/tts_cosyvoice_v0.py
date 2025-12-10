from __future__ import annotations
from ..voice_field import TTSClient
from ..context import ProsodyVec


class CosyVoiceTTSClient(TTSClient):
    """Skeleton TTS client for CosyVoice (to be implemented)."""

    def __init__(self, base_url: str = "http://localhost:8002") -> None:
        self.base_url = base_url

    def speak(self, text: str, prosody: ProsodyVec) -> bytes:
        raise NotImplementedError("CosyVoiceTTSClient.speak must be implemented.")
