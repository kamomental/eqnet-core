# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
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
LLM_MODEL_CACHE_PATH = os.getenv("LLM_MODEL_CACHE_PATH", "data/state/llm_model_cache.json")

@dataclass
class LLMInfo:
    client: OpenAI
    base_url: str
    api_key: str
    model: Optional[str]
    available: bool
    model_source: str = ""

def _client(base: str, key: str) -> OpenAI:
    return OpenAI(base_url=base, api_key=key)

def list_models(base: str, key: str) -> List[str]:
    try:
        cli = _client(base, key)
        res = cli.models.list()
        return [m.id for m in res.data]
    except Exception:
        return []

def pick_model(candidates: List[str], cached_selected: Optional[str] = None) -> Optional[str]:
    if not candidates:
        return None
    normalized_cached = str(cached_selected or "").strip()
    if normalized_cached:
        for model_name in candidates:
            if model_name == normalized_cached:
                return model_name
    for pref in PREFER:
        for m in candidates:
            if pref.lower() in m.lower():
                return m
    return candidates[0]


def _cache_path() -> Path:
    return Path(LLM_MODEL_CACHE_PATH)


def _normalize_base(base: str) -> str:
    return str(base or "").rstrip("/").strip()


def _load_model_cache() -> dict:
    path = _cache_path()
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_cached_model_entry(base: str) -> dict:
    cache = _load_model_cache()
    entry = cache.get(_normalize_base(base))
    return dict(entry) if isinstance(entry, dict) else {}


def _store_model_cache(base: str, models: List[str], selected_model: Optional[str]) -> None:
    normalized_base = _normalize_base(base)
    if not normalized_base or not models:
        return
    cache = _load_model_cache()
    cache[normalized_base] = {
        "models": [str(item) for item in models if str(item).strip()],
        "selected_model": str(selected_model or "").strip(),
        "updated_at": int(time.time()),
    }
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def prefer_cached_model(base: str, model: str, *, available_models: Optional[List[str]] = None) -> None:
    normalized_base = _normalize_base(base)
    selected_model = str(model or "").strip()
    if not normalized_base or not selected_model:
        return
    cache = _load_model_cache()
    existing_entry = cache.get(normalized_base) if isinstance(cache.get(normalized_base), dict) else {}
    existing_models = [str(item) for item in (existing_entry.get("models") or []) if str(item).strip()]
    merged_models = list(existing_models)
    for candidate in list(available_models or []):
        candidate_text = str(candidate).strip()
        if candidate_text and candidate_text not in merged_models:
            merged_models.append(candidate_text)
    if selected_model not in merged_models:
        merged_models.insert(0, selected_model)
    cache[normalized_base] = {
        "models": merged_models,
        "selected_model": selected_model,
        "updated_at": int(time.time()),
    }
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_cached_selected_model(base: str) -> Optional[str]:
    entry = _get_cached_model_entry(base)
    selected_model = str(entry.get("selected_model") or "").strip()
    return selected_model or None

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
        cached_entry = _get_cached_model_entry(base)
        cached_models = [str(item) for item in cached_entry.get("models") or [] if str(item).strip()]
        cached_selected = str(cached_entry.get("selected_model") or "").strip() or None
        models = list_models(base, key)
        model_source = ""
        if forced_model:
            model = forced_model
            model_source = "forced"
        elif models:
            model = pick_model(models, cached_selected=cached_selected)
            model_source = "live_list"
            _store_model_cache(base, models, model)
        elif cached_models:
            model = pick_model(cached_models, cached_selected=cached_selected)
            model_source = "cache"
        else:
            model = None
        try:
            cli = _client(base, key)
            available = model is not None
        except Exception:
            cli = _client(base, key)
            available = False
        if available:
            return LLMInfo(
                client=cli,
                base_url=base,
                api_key=key,
                model=model,
                available=available,
                model_source=model_source,
            )
    # If all attempts failed, return last candidate (so user still gets a client even if unavailable)
    if candidates:
        entry = candidates[-1]
        cli = _client(entry["base"], entry["key"])
        base = entry["base"]
        forced_model = entry.get("forced_model")
        cached_entry = _get_cached_model_entry(base)
        cached_models = [str(item) for item in cached_entry.get("models") or [] if str(item).strip()]
        cached_selected = str(cached_entry.get("selected_model") or "").strip() or None
        if forced_model:
            model = forced_model
            model_source = "forced"
        elif cached_models:
            model = pick_model(cached_models, cached_selected=cached_selected)
            model_source = "cache"
        else:
            model = None
            model_source = ""
        return LLMInfo(
            client=cli,
            base_url=base,
            api_key=entry["key"],
            model=model,
            available=False,
            model_source=model_source,
        )
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
