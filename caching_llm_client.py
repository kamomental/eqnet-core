from __future__ import annotations
import json
import hashlib
from collections import OrderedDict
from typing import List, Dict

from heartos_mini import LLMClient


class CachingLLMClient(LLMClient):
    """Simple LRU cache wrapper for any LLMClient backend."""

    def __init__(self, backend: LLMClient, max_entries: int = 256) -> None:
        self.backend = backend
        self.max_entries = max_entries
        self._cache: OrderedDict[str, str] = OrderedDict()

    def chat(self, system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
        key = self._make_key(system_prompt, history, user_text)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        response = self.backend.chat(system_prompt, history, user_text)
        self._cache[key] = response
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return response

    def _make_key(self, system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
        payload = {
            "system": system_prompt,
            "history": history,
            "user": user_text,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
