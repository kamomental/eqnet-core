from __future__ import annotations
import pyttsx3

from ..voice_field import TTSClient
from ..context import ProsodyVec


class Pyttsx3TTSClient(TTSClient):
    """Simple pyttsx3-based TTS client for quick feedback."""

    def __init__(self, base_rate: int = 200) -> None:
        self.engine = pyttsx3.init()
        self.base_rate = base_rate

    def speak(self, text: str, prosody: ProsodyVec) -> bytes:
        tempo = max(0.5, min(1.5, prosody.tempo))
        rate = int(self.base_rate * tempo)
        volume = max(0.0, min(1.0, prosody.energy))

        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self.engine.say(text)
        self.engine.runAndWait()
        return b""
