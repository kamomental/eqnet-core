from __future__ import annotations
from ..voice_field import TTSClient
from ..context import ProsodyVec


class Qwen3TTSClient(TTSClient):
    """Skeleton TTS client for Qwen3-TTS (to be implemented)."""

    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self.base_url = base_url

    def speak(self, text: str, prosody: ProsodyVec) -> bytes:
        raise NotImplementedError("Qwen3TTSClient.speak must be implemented.")
