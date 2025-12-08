"""Loader for voice profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from eqnet.voice.voice_profile import VoiceProfile


def load_voice_profiles(path: str | Path) -> Dict[str, VoiceProfile]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    profiles_raw = data.get("profiles", {})
    profiles: Dict[str, VoiceProfile] = {}
    for profile_id, vals in profiles_raw.items():
        profiles[profile_id] = VoiceProfile(id=profile_id, **vals)
    return profiles
