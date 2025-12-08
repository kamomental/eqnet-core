"""TTS routing helpers."""

from __future__ import annotations

from typing import Optional

from .style_state import UtteranceStyleState
from .voice_profile import VoiceProfile


def speak(text: str, style: UtteranceStyleState, voice_profile: VoiceProfile) -> Optional[bytes]:
    engine = voice_profile.engine.lower()
    speed = voice_profile.base_speed
    pitch = voice_profile.base_pitch

    if style.mood == "excited":
        speed *= 1.1
        pitch += 0.1
    elif style.mood == "tired":
        speed *= 0.9

    if engine == "voicevox":
        return _speak_voicevox(text, voice_profile, speed, pitch)
    if engine in {"aivis", "stylebertvits2", "sbvits2"}:
        return _speak_aivis(text, voice_profile, speed, pitch, style)
    if engine in {"openai_tts", "azure_tts", "google_tts", "elevenlabs", "cartesia"}:
        return _speak_cloud(text, voice_profile, speed, pitch, style)

    print(f"[tts_router] unsupported engine: {engine}")
    return None


def _speak_voicevox(text: str, vp: VoiceProfile, speed: float, pitch: float) -> Optional[bytes]:
    print(f"[VOICEVOX] text={text[:30]} speaker={vp.speaker_id} speed={speed:.2f} pitch={pitch:.2f}")
    return None


def _speak_aivis(
    text: str,
    vp: VoiceProfile,
    speed: float,
    pitch: float,
    style: UtteranceStyleState,
) -> Optional[bytes]:
    print(
        f"[AIVIS] text={text[:30]} speaker={vp.speaker_id} speed={speed:.2f} "
        f"pitch={pitch:.2f} mood={style.mood}"
    )
    return None


def _speak_cloud(
    text: str,
    vp: VoiceProfile,
    speed: float,
    pitch: float,
    style: UtteranceStyleState,
) -> Optional[bytes]:
    print(
        f"[CLOUD TTS] engine={vp.engine} text={text[:30]} speed={speed:.2f} "
        f"pitch={pitch:.2f} mood={style.mood}"
    )
    return None
