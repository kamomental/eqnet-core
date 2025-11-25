# -*- coding: utf-8 -*-
"""Minimal tests for the mask layer persona helpers."""

from __future__ import annotations

from eqnet.mask_layer import (
    MaskPersonaProfile,
    MaskLayer,
    MaskPrompt,
    describe_mood,
    load_persona_profile,
)


def test_load_persona_profile_from_preset() -> None:
    profile = load_persona_profile({"preset": "tsun"})
    assert isinstance(profile, MaskPersonaProfile)
    assert profile.persona_id == "tsun_core"
    assert 0.0 <= profile.masking_strength <= 1.0


def test_with_overrides_applies_changes() -> None:
    base = MaskPersonaProfile.from_preset("business")
    tweaked = base.with_overrides({"masking_strength": 0.2, "distance_bias": "formal"})
    assert tweaked.masking_strength == 0.2
    assert tweaked.distance_bias == "formal"
    assert tweaked.persona_id == base.persona_id


def test_describe_mood_returns_non_empty_text() -> None:
    mood_text = describe_mood([0.5, -0.2, 0.0])
    assert isinstance(mood_text, str)
    assert mood_text.strip()


def test_mask_layer_build_prompt_uses_persona() -> None:
    profile = load_persona_profile({"preset": "default"})
    layer = MaskLayer(profile)
    prompt = layer.build_prompt(inner_spec={"pdc": {"E_story": 0.2}}, dialog_context={})
    assert isinstance(prompt, MaskPrompt)
    assert profile.display_name in prompt.system_prompt
    assert prompt.persona_meta.get("persona_id") == profile.persona_id
