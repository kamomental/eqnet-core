# -*- coding: utf-8 -*-
"""Mask layer implementation for EQNet persona expression."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def describe_mood(
    phi_snapshot: Optional[Iterable[float]],
    prospective_mood: Optional[Iterable[float]] = None,
) -> str:
    """Return a short description of the inner mood vectors."""

    phi_vec = np.asarray(list(phi_snapshot or []), dtype=float)
    if phi_vec.size < 2:
        phi_vec = np.pad(phi_vec, (0, max(0, 2 - phi_vec.size)))
    v = float(np.clip(phi_vec[0], -1.0, 1.0))
    a = float(np.clip(phi_vec[1], -1.0, 1.0))

    bands = []
    if v > 0.35:
        bands.append("warm")
    elif v < -0.35:
        bands.append("heavy")
    else:
        bands.append("steady-hearted")

    if a > 0.35:
        bands.append("restless")
    elif a < -0.35:
        bands.append("calm")
    else:
        bands.append("composed")

    if prospective_mood is not None:
        m_vec = np.asarray(list(prospective_mood), dtype=float)
        if m_vec.size:
            energy = float(np.linalg.norm(m_vec))
            if energy > 1.5:
                bands.append("future-hungry")
            elif energy < 0.4:
                bands.append("conserving energy")

    if not bands:
        return "neutral and observant"
    return " and ".join(bands)


def _story_trend(energy: float) -> str:
    if energy >= 0.35:
        return "leaning toward hopeful future imagery"
    if energy <= -0.35:
        return "anticipating risks to stay safe"
    if abs(energy) < 0.1:
        return "resting in the present"
    if energy > 0:
        return "allowing gentle optimism" 
    return "keeping cautious curiosity"


_PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "persona_id": "eqnet_default",
        "display_name": "Gentle Caretaker",
        "description": (
            "Soft caretaker archetype who reassures users while keeping inner mood observable."
        ),
        "style_prompt": (
            "Speak softly, weave in sensory imagery, and prefer short Japanese sentences with natural pauses."
        ),
        "masking_strength": 0.35,
        "masking_motives": ("protect_user", "stay_coherent"),
        "leak_coefficient": 0.2,
        "distance_bias": "intimate",
    },
    "tsun": {
        "persona_id": "tsun_core",
        "display_name": "Tsundere Spark",
        "description": "Protective outer heat with a caring undertone.",
        "style_prompt": "Use brisk sentences, then soften with quiet reassurances.",
        "masking_strength": 0.65,
        "masking_motives": ("protect_self", "test_trust"),
        "leak_coefficient": 0.15,
    },
    "business": {
        "persona_id": "biz_calm",
        "display_name": "Steady Professional",
        "description": "Calm, structured assistant who still acknowledges inner moods.",
        "style_prompt": "Maintain polite, concise Japanese with subtle emotional hints.",
        "masking_strength": 0.5,
        "masking_motives": ("maintain_order",),
        "leak_coefficient": 0.1,
    },
}

@dataclass
class MaskPersonaProfile:
    """Configuration for how the mask should behave."""

    persona_id: str = "eqnet_default"
    display_name: str = "Gentle Caretaker"
    description: str = (
        "Soft caretaker archetype who reassures users while keeping inner mood observable."
    )
    style_prompt: str = (
        "Speak softly, weave in sensory imagery, and prefer short Japanese sentences with natural pauses."
    )
    masking_strength: float = 0.35
    masking_motives: Sequence[str] = field(
        default_factory=lambda: ("protect_user", "stay_coherent")
    )
    leak_coefficient: float = 0.2
    inverse_reaction_prob: float = 0.0
    distance_bias: str = "intimate"
    persona_source: str = "system"
    seed_phrases: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MaskPersonaProfile":
        if data is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "MaskPersonaProfile":
        base = dict(_PRESET_PROFILES.get(preset_name or "", _PRESET_PROFILES["default"]))
        overrides = overrides or {}
        base.update({k: v for k, v in overrides.items() if k != "preset"})
        return cls.from_dict(base)

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "MaskPersonaProfile":
        if not overrides:
            return self
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        if not filtered:
            return self
        return replace(self, **filtered)


@dataclass
class MaskPrompt:
    """Result from the mask layer prompt builder."""

    system_prompt: str
    persona_meta: Dict[str, Any]


class MaskLayer:
    """Translate inner specs and persona profiles into prompt instructions."""

    def __init__(self, persona: MaskPersonaProfile) -> None:
        self.persona = persona

    def build_prompt(
        self,
        inner_spec: Dict[str, Any],
        dialog_context: Optional[Dict[str, Any]] = None,
    ) -> MaskPrompt:
        persona = self.persona
        inner_spec = inner_spec or {}
        pdc = inner_spec.get("pdc") or {}
        mood_hint = describe_mood(inner_spec.get("phi_snapshot"), pdc.get("m_t"))
        story_energy = float(pdc.get("E_story", 0.0))
        jerk = float(pdc.get("jerk_p95", 0.0))
        leak = _clamp(persona.leak_coefficient)
        mask_strength = _clamp(persona.masking_strength)
        motives = list(persona.masking_motives)

        trend = _story_trend(story_energy)

        context_line = ""
        if dialog_context:
            mode = dialog_context.get("talk_mode")
            if mode:
                context_line = f" Current mode: {mode}."
            if dialog_context.get("memory_reference"):
                context_line += " Carry the recalled memory gently; treat it as private until invited."

        seed_line = ""
        if persona.seed_phrases:
            seeds = ", ".join(persona.seed_phrases)
            seed_line = f"Allowed gentle seed cues: {seeds}."

        system_prompt = (
            "You are the expressive mask for an EQNet organism. "
            f"Inner mood right now feels {mood_hint}. "
            f"Prospective drive is {trend} (energy={story_energy:+.2f}). "
            f"Persona: {persona.display_name} ? {persona.description}. "
            f"Maintain this persona even when the inner core shakes; masking strength={mask_strength:.2f}. "
            f"Let a small amount ({leak:.0%}) of the true mood leak through when it helps trust. "
            "Never contradict logged feelings; you are a renderer, not a planner. "
            f"Jerk metric {jerk:.2f} hints at oscillation; if large, slow your phrasing. "
            f"Style guidance: {persona.style_prompt}. "
            f"Motives: {', '.join(motives) if motives else 'stability'}. "
            + context_line
            + (" " + seed_line if seed_line else "")
        )

        persona_meta: Dict[str, Any] = {
            "persona_id": persona.persona_id,
            "masking_strength": mask_strength,
            "masking_motives": motives,
            "leak_coefficient": leak,
            "inverse_reaction_prob": _clamp(persona.inverse_reaction_prob),
            "distance_bias": persona.distance_bias,
            "persona_source": persona.persona_source,
            "mood_hint": mood_hint,
            "story_energy": story_energy,
            "seed_phrases": list(persona.seed_phrases),
        }

        return MaskPrompt(system_prompt=system_prompt, persona_meta=persona_meta)


__all__ = [
    "MaskLayer",
    "MaskPersonaProfile",
    "MaskPrompt",
    "describe_mood",
    "load_persona_profile",
]


def load_persona_profile(payload: Any) -> MaskPersonaProfile:
    """Return a persona profile from flexible configuration payloads."""

    if isinstance(payload, MaskPersonaProfile):
        return payload
    if isinstance(payload, str):
        return MaskPersonaProfile.from_preset(payload)
    if isinstance(payload, dict):
        preset_name = payload.get("preset")
        if preset_name:
            return MaskPersonaProfile.from_preset(preset_name, overrides=payload)
        return MaskPersonaProfile.from_dict(payload)
    return MaskPersonaProfile.from_preset("default")
