# -*- coding: utf-8 -*-
"""
Minimal LLM hub wrapper.

Routes requests to the existing ``terrain.llm`` helpers while tracking metadata
so that the outer runtime can treat this as a structured "mouth". The routing
logic is intentionally lightweight; production deployments can swap it out with
tool-enabled planners without changing the call-site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time
import os

from terrain import llm as terrain_llm
from emot_terrain_lab.tools.registry import SkillRegistry
from emot_terrain_lab.hub.akorn import AkornGate, AkornConfig
from emot_terrain_lab.rag.lazy_rag import LazyRAG, LazyRAGConfig


@dataclass
class LLMHubConfig:
    """Configuration for the hub interface."""

    default_system_prompt: str = (
        "You are an assistant voice for an EQNet-driven companion. "
        "Keep responses concise and compassionate."
    )
    default_intent: str = "chitchat"
    fallback_text: str = "ごめんね、今すぐ良い言葉が浮かばなかったよ。"


@dataclass
class HubResponse:
    text: str
    model: Optional[str]
    trace_id: str
    latency_ms: float
    controls_used: Dict[str, float]
    safety: Dict[str, str]


class LLMHub:
    """Very small routing facade over ``terrain.llm``."""

    def __init__(self, config: Optional[LLMHubConfig] = None) -> None:
        self.config = config or LLMHubConfig()
        self.registry = SkillRegistry()
        # Allow environment-based tuning of AKOrN small gains
        self.akorn = AkornGate(AkornConfig.from_env())
        self._lazy_rag: Optional[LazyRAG] = None

    def _get_lazy_rag(self) -> Optional[LazyRAG]:
        flag = (os.getenv("EQNET_LAZY_RAG", "0") or "0").lower()
        if flag in {"0", "false", "off"}:
            return None
        if self._lazy_rag is None:
            self._lazy_rag = LazyRAG(LazyRAGConfig.from_env())
        return self._lazy_rag

    def generate(
        self,
        user_text: str,
        context: Optional[str],
        controls: Dict[str, float],
        intent: Optional[str] = None,
        slos: Optional[Dict[str, float]] = None,
    ) -> HubResponse:
        """
        Generate text under EQNet control.

        Parameters
        ----------
        user_text:
            Raw user utterance.
        context:
            Optional additional context (e.g., retrieved snippets). For the
            scaffold we simply append it to the prompt.
        controls:
            Dict containing behaviour controls (pause_ms, temperature, warmth…)
            produced by :class:`hub.policy.PolicyHead`.
        intent:
            Intent label (qa/chitchat/code…). Currently only chitchat behaviour
            is defined but the parameter is kept for future routing policies.
        slos:
            Optional SLO hints (e.g., ``{"p95_ms": 180}``). The scaffold only
            records the value for metrics.
        """
        skill = self.registry.find_by_intent(intent or self.config.default_intent)
        system_prompt = self.config.default_system_prompt
        chosen_llm = None
        if skill:
            chosen_llm = skill.llm
            if skill.intent == "qa":
                system_prompt += " Use the provided context when helpful."
            if skill.context_sources and context:
                context = context.strip()
        if not context:
            rag = self._get_lazy_rag()
            if rag is not None:
                try:
                    context = rag.build_context(user_text)
                except Exception:
                    context = None
        prompt = user_text
        if context:
            prompt = context.strip() + "\n\n---\n\n" + user_text.strip()

        # Apply AKOrN gate (minimal): looks for R/rho/I/q in controls or nested under 'akorn'
        gated_controls, gate_log = self.akorn.apply(controls)
        temperature = float(gated_controls.get("temperature", controls.get("temperature", 0.6)))
        top_p = float(gated_controls.get("top_p", controls.get("top_p", 0.85)))
        pause_ms = float(gated_controls.get("pause_ms", controls.get("pause_ms", 360.0)))

        start = time.perf_counter()
        text = terrain_llm.chat_text(system_prompt, prompt, temperature=temperature, top_p=top_p)
        latency = (time.perf_counter() - start) * 1000.0

        if not text:
            text = self.config.fallback_text
        model = terrain_llm.get_llm().model or chosen_llm

        trace_id = f"hub-{int(time.time() * 1000)}"
        safety = {
            "rating": "G",
            "spoiler": controls.get("spoiler_mode", "warn"),
            "pii_block": str(True),
        }
        if slos and latency > slos.get("p95_ms", 1e9):
            safety["note"] = "latency_degraded"

        # Merge control usage for traceability
        used_controls = dict(gated_controls)
        used_controls.setdefault("temperature", temperature)
        used_controls.setdefault("top_p", top_p)
        used_controls.setdefault("pause_ms", pause_ms)

        # Attach AKOrN deltas into safety note
        if gate_log:
            gate_note = {k: float(v) for k, v in gate_log.items()}
        else:
            gate_note = {}

        return HubResponse(
            text=text,
            model=model,
            trace_id=trace_id,
            latency_ms=latency,
            controls_used=used_controls,
            safety={**safety, **({"akorn": "applied"} if gate_log else {}), **gate_note},
        )
