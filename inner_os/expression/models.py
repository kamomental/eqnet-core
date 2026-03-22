from __future__ import annotations

from dataclasses import dataclass, field


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
    interaction_policy: dict[str, object] = field(default_factory=dict)
    action_posture: dict[str, object] = field(default_factory=dict)
    actuation_plan: dict[str, object] = field(default_factory=dict)
    surface_profile: dict[str, object] = field(default_factory=dict)
    llm_payload: dict[str, object] = field(default_factory=dict)
