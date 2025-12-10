from __future__ import annotations
from typing import List, Dict
import os
from openai import OpenAI

from heartos_mini import LLMClient

LMSTUDIO_BASE_URL = os.getenv('LMSTUDIO_BASE_URL', 'http://127.0.0.1:1234/v1')
LMSTUDIO_API_KEY = os.getenv('LMSTUDIO_API_KEY', 'lm-studio')
LMSTUDIO_MODEL = os.getenv('LMSTUDIO_MODEL', 'lmstudio-community/gpt-oss-20b')


class LMStudioLLMClient(LLMClient):
    """LLM client that talks to LM Studio's OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or LMSTUDIO_BASE_URL
        self.api_key = api_key or LMSTUDIO_API_KEY
        self.model = model or LMSTUDIO_MODEL
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""
