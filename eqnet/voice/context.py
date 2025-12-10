from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


class TalkMode(str, Enum):
    STREAM = "stream"
    INTIMATE = "intimate"
    PRESENCE = "presence"
    CAREFUL = "careful"


@dataclass
class ProsodyVec:
    tempo: float
    pitch: float
    energy: float


@dataclass
class VoiceContext:
    q_t: np.ndarray
    partner_id: str
    partner_profile: np.ndarray
    m_t: np.ndarray
    M_long: np.ndarray
    id_t: np.ndarray
    talk_mode: TalkMode
    extra: dict[str, Any] | None = None
