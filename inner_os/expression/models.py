from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class DialogueContext:
    user_text: str = ""
    history: list[str] = field(default_factory=list)
    expression_hints: dict[str, object] = field(default_factory=dict)


@dataclass
class ResponsePlan:
    speech_act: str
    content_brief: list[str] = field(default_factory=list)
    multimodal_cues: list[str] = field(default_factory=list)
    interaction_policy: MutableMapping[str, object] = field(default_factory=dict)
    action_posture: Mapping[str, object] = field(default_factory=dict)
    actuation_plan: Mapping[str, object] = field(default_factory=dict)
    reaction_contract: Mapping[str, object] = field(default_factory=dict)
    discourse_shape: dict[str, object] = field(default_factory=dict)
    surface_profile: dict[str, object] = field(default_factory=dict)
    boundary_transform: dict[str, object] = field(default_factory=dict)
    residual_reflection: dict[str, object] = field(default_factory=dict)
    llm_payload: dict[str, object] = field(default_factory=dict)
