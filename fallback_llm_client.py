from __future__ import annotations
from typing import List, Dict
import openai

from heartos_mini import LLMClient, DummyLLMClient
from lmstudio_llm_client import LMStudioLLMClient


class FallbackLLMClient(LLMClient):
    """Try LM Studio first; fall back to Dummy if it fails."""

    def __init__(self) -> None:
        self.primary = LMStudioLLMClient()
        self.fallback = DummyLLMClient()
        self.use_fallback = False

    def chat(self, system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
        if self.use_fallback:
            return self.fallback.chat(system_prompt, history, user_text)
        try:
            return self.primary.chat(system_prompt, history, user_text)
        except openai.APIConnectionError as exc:
            print("[FallbackLLM] LM Studio unreachable. Switching to DummyLLM.")
            print(f"[FallbackLLM] error={exc}")
            self.use_fallback = True
            return self.fallback.chat(system_prompt, history, user_text)
