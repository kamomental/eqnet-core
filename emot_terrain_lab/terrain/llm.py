# -*- coding: utf-8 -*-
import os
import json
from typing import Optional, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_BASE = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
PREFER = [s.strip() for s in os.getenv("LMSTUDIO_MODEL_PREFER", "Qwen,Llama,Phi,Nous,Deepseek").split(",") if s.strip()]

@dataclass
class LLMInfo:
    client: OpenAI
    base_url: str
    api_key: str
    model: Optional[str]
    available: bool

def _client(base: str = DEFAULT_BASE, key: str = DEFAULT_KEY) -> OpenAI:
    return OpenAI(base_url=base, api_key=key)

def list_models(base: str = DEFAULT_BASE, key: str = DEFAULT_KEY) -> List[str]:
    try:
        cli = _client(base, key)
        res = cli.models.list()
        return [m.id for m in res.data]
    except Exception:
        return []

def pick_model(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    for pref in PREFER:
        for m in candidates:
            if pref.lower() in m.lower():
                return m
    return candidates[0]

def get_llm() -> LLMInfo:
    base = DEFAULT_BASE
    key = DEFAULT_KEY
    models = list_models(base, key)
    model = pick_model(models)
    try:
        cli = _client(base, key)
        available = model is not None
    except Exception:
        cli = _client(base, key)
        available = False
    return LLMInfo(client=cli, base_url=base, api_key=key, model=model, available=available)

def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.2, top_p: Optional[float] = None) -> Optional[dict]:
    llm = get_llm()
    if not llm.available:
        return None
    try:
        params = dict(
            model=llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        if top_p is not None:
            params["top_p"] = float(top_p)
        resp = llm.client.chat.completions.create(**params)
        txt = resp.choices[0].message.content
        import json
        return json.loads(txt)
    except Exception:
        return None

def chat_text(system_prompt: str, user_prompt: str, temperature: float = 0.5, top_p: Optional[float] = None) -> Optional[str]:
    llm = get_llm()
    if not llm.available:
        return None
    try:
        params = dict(
            model=llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        if top_p is not None:
            params["top_p"] = float(top_p)
        resp = llm.client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
