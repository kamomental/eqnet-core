# -*- coding: utf-8 -*-
import os
import json
from typing import Optional, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Priority 1: user-provided OpenAI-compatible endpoint (self-hosted or cloud)
CUSTOM_BASE = os.getenv("OPENAI_BASE_URL") or None
CUSTOM_KEY = os.getenv("OPENAI_API_KEY") or None
CUSTOM_MODEL = os.getenv("OPENAI_MODEL") or None

# Priority 2: LM Studio local server
LM_BASE = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LM_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LM_MODEL = os.getenv("LMSTUDIO_MODEL") or None

PREFER = [s.strip() for s in os.getenv("LLM_MODEL_PREFER", os.getenv("LMSTUDIO_MODEL_PREFER", "Qwen,Llama,Phi,Nous,Deepseek")).split(",") if s.strip()]

@dataclass
class LLMInfo:
    client: OpenAI
    base_url: str
    api_key: str
    model: Optional[str]
    available: bool

def _client(base: str, key: str) -> OpenAI:
    return OpenAI(base_url=base, api_key=key)

def list_models(base: str, key: str) -> List[str]:
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

def _resolve_endpoint() -> Optional[LLMInfo]:
    candidates = []
    if CUSTOM_BASE and CUSTOM_KEY:
        candidates.append(
            {
                "base": CUSTOM_BASE,
                "key": CUSTOM_KEY,
                "forced_model": CUSTOM_MODEL,
            }
        )
    # Always fall back to LM Studio
    if LM_BASE and LM_KEY:
        candidates.append(
            {
                "base": LM_BASE,
                "key": LM_KEY,
                "forced_model": LM_MODEL,
            }
        )

    for entry in candidates:
        base = entry["base"]
        key = entry["key"]
        forced_model = entry.get("forced_model")
        models = list_models(base, key)
        model = forced_model or pick_model(models)
        try:
            cli = _client(base, key)
            available = model is not None
        except Exception:
            cli = _client(base, key)
            available = False
        if available:
            return LLMInfo(client=cli, base_url=base, api_key=key, model=model, available=available)
    # If all attempts failed, return last candidate (so user still gets a client even if unavailable)
    if candidates:
        entry = candidates[-1]
        cli = _client(entry["base"], entry["key"])
        model = entry.get("forced_model")
        return LLMInfo(client=cli, base_url=entry["base"], api_key=entry["key"], model=model, available=False)
    return None

def get_llm() -> LLMInfo:
    info = _resolve_endpoint()
    if info is None:
        raise RuntimeError("No LLM endpoint configured. Set OPENAI_BASE_URL/API_KEY or LMSTUDIO_BASE_URL/API_KEY.")
    return info

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
