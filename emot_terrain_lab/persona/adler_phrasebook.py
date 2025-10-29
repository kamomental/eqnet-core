# -*- coding: utf-8 -*-
"""
Phrasebook for Adlerian (community-oriented) acknowledgement language.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

_PHRASEBOOK_PATH = Path("config/i18n/adler.yaml")


@lru_cache(maxsize=1)
def _load_phrasebook() -> Dict[str, Dict[str, dict]]:
    if not _PHRASEBOOK_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(_PHRASEBOOK_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    normalised: Dict[str, Dict[str, dict]] = {}
    for culture, payload in data.items():
        if not isinstance(payload, dict):
            continue
        normalised[culture.lower()] = payload
    return normalised


def _culture_keys(culture: str) -> Iterable[str]:
    norm = (culture or "").lower()
    if norm:
        yield norm
        if "-" in norm:
            yield norm.split("-")[0]
    yield "default"


def _deterministic_choice(options: Iterable[str], key: str) -> Optional[str]:
    opts = [opt.strip() for opt in options if isinstance(opt, str) and opt.strip()]
    if not opts:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(opts)
    return opts[idx]


def get_phrase(
    culture: str,
    tone: str,
    category: str,
    seed: Optional[str] = None,
) -> Optional[str]:
    data = _load_phrasebook()
    tone_key = (tone or "default").lower()
    cat = category.lower()
    lookup_key = f"{culture}|{tone_key}|{category}|{seed or ''}"
    for culture_key in _culture_keys(culture):
        culture_payload = data.get(culture_key)
        if not culture_payload:
            continue
        cat_payload = culture_payload.get(cat)
        if isinstance(cat_payload, dict):
            options = cat_payload.get(tone_key) or cat_payload.get("default") or []
        elif isinstance(cat_payload, list):
            options = cat_payload
        else:
            options = []
        phrase = _deterministic_choice(options, lookup_key)
        if phrase:
            return phrase
    return None


def get_avoid_phrases(culture: str) -> Iterable[str]:
    data = _load_phrasebook()
    phrases = []
    for culture_key in _culture_keys(culture):
        payload = data.get(culture_key)
        if not payload:
            continue
        avoid = payload.get("avoid_phrases") or []
        for item in avoid:
            if isinstance(item, str) and item:
                phrases.append(item)
    return phrases


def clean_message(culture: str, message: str) -> str:
    cleaned = message
    for phrase in get_avoid_phrases(culture):
        cleaned = cleaned.replace(phrase, "")
    return " ".join(cleaned.split())


__all__ = ["get_phrase", "get_avoid_phrases", "clean_message"]
