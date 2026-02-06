from dataclasses import dataclass


@dataclass
class AvatarState:
    blink: float
    mouth_open: float
    wavy: float
    fang_skin: bool
    bob: float
    hop: float


@dataclass
class AvatarInputs:
    voice_energy: float
    is_speaking: bool
