# -*- coding: utf-8 -*-
"""EQNet emotional hub scaffolding."""

from .perception import PerceptionBridge, PerceptionConfig, AffectSample
from .policy import PolicyHead, PolicyConfig, AffectControls

# Optional imports (LLM backends may not be available in every environment)
try:
    from .llm_hub import LLMHub, LLMHubConfig, HubResponse  # type: ignore
except Exception:  # pragma: no cover - optional
    LLMHub = None  # type: ignore
    LLMHubConfig = None  # type: ignore
    HubResponse = None  # type: ignore

try:
    from .runtime import EmotionalHubRuntime, RuntimeConfig  # type: ignore
except Exception:  # pragma: no cover - optional
    EmotionalHubRuntime = None  # type: ignore
    RuntimeConfig = None  # type: ignore

try:
    from .hub import Hub  # type: ignore
except Exception:  # pragma: no cover - optional
    Hub = None  # type: ignore

__all__ = [
    "PerceptionBridge",
    "PerceptionConfig",
    "AffectSample",
    "PolicyHead",
    "PolicyConfig",
    "AffectControls",
    "LLMHub",
    "LLMHubConfig",
    "HubResponse",
    "EmotionalHubRuntime",
    "RuntimeConfig",
    "Hub",
]
