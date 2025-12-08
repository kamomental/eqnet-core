"""Character configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from eqnet.voice.style_state import UtteranceStyleState
from eqnet.voice.voice_profile import VoiceProfile


@dataclass
class CharacterConfig:
    id: str
    display_name: str
    persona_prompt: str
    base_style: UtteranceStyleState
    voice_profile: VoiceProfile
    raw: Dict[str, Any]


def load_character(
    path: str | Path,
    style_presets: Dict[str, UtteranceStyleState],
    voice_profiles: Dict[str, VoiceProfile],
) -> CharacterConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    char_id = data["id"]
    display_name = data.get("display_name", char_id)
    persona_prompt = data.get("persona_prompt", "")

    style_cfg = data.get("style", {})
    base_preset_id = style_cfg.get("base_preset")
    if base_preset_id not in style_presets:
        raise ValueError(f"Unknown style preset: {base_preset_id}")
    base_preset = style_presets[base_preset_id]
    overrides = style_cfg.get("overrides", {})
    style_dict = base_preset.__dict__.copy()
    style_dict.update(overrides)
    base_style = UtteranceStyleState(**style_dict)

    voice_cfg = data.get("voice", {})
    profile_id = voice_cfg.get("profile_id")
    if profile_id not in voice_profiles:
        raise ValueError(f"Unknown voice profile: {profile_id}")
    voice_profile = voice_profiles[profile_id]

    return CharacterConfig(
        id=char_id,
        display_name=display_name,
        persona_prompt=persona_prompt,
        base_style=base_style,
        voice_profile=voice_profile,
        raw=data,
    )
