# -*- coding: utf-8 -*-
"""
Audio backends for the perception bridge.

Provides lightweight RMS/ZCR extraction and extension points for more advanced
pipelines (WebRTC VAD, deep prosody models, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

try:
    import webrtcvad  # type: ignore
    HAS_WEBRTCVAD = True
except Exception:
    HAS_WEBRTCVAD = False


@dataclass
class AudioBackendConfig:
    vad_mode: int = 2
    sample_rate: int = 16_000


class BaseAudioBackend:
    def __init__(self, config: AudioBackendConfig):
        self.config = config

    def process(self, chunk: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


class LightweightAudioBackend(BaseAudioBackend):
    """Default RMS + ZCR extractor."""

    def process(self, chunk: np.ndarray) -> Dict[str, float]:
        if chunk.dtype != np.float32:
            max_val = np.max([np.abs(np.iinfo(chunk.dtype).min), np.abs(np.iinfo(chunk.dtype).max)])
            norm = max(max_val, 1)
            chunk = chunk.astype(np.float32) / norm
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk))))) * 0.5
        return {"rms": rms, "zcr": zcr}


class WebRTCVADBackend(BaseAudioBackend):
    """Voice activity detection via WebRTC VAD."""

    def __init__(self, config: AudioBackendConfig):
        if not HAS_WEBRTCVAD:
            raise RuntimeError("webrtcvad is not installed.")
        super().__init__(config)
        self._vad = webrtcvad.Vad(config.vad_mode)

    def process(self, chunk: np.ndarray) -> Dict[str, float]:
        if chunk.dtype != np.int16:
            chunk_int16 = (chunk * np.iinfo(np.int16).max).astype(np.int16)
        else:
            chunk_int16 = chunk
        active = self._vad.is_speech(chunk_int16.tobytes(), self.config.sample_rate)
        energy = float(np.sqrt(np.mean(chunk_int16.astype(np.float32) ** 2) + 1e-9) / np.iinfo(np.int16).max)
        return {"rms": energy, "zcr": 0.0, "voice_activity": 1.0 if active else 0.0}


def create_audio_backend(name: str, config: AudioBackendConfig) -> BaseAudioBackend:
    name = name.lower()
    if name == "webrtcvad":
        return WebRTCVADBackend(config)
    return LightweightAudioBackend(config)

