from __future__ import annotations
from ..voice_field import TTSClient
from ..context import ProsodyVec


class DummyTTSClient(TTSClient):
    """Provisional dummy TTS: just logs text and prosody."""

    def speak(self, text: str, prosody: ProsodyVec) -> bytes:
        print(f"[DummyTTS] text   = {text}")
        print(f"[DummyTTS] prosody= {prosody}")
        return b""
