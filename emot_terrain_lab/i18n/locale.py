"""Simple locale text lookup backed by locales/*.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCALE_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_locale(locale: str) -> str:
    return (locale or "ja").strip().lower().replace("_", "-")


def locale_candidates(locale: str) -> List[str]:
    normalized = normalize_locale(locale)
    parts = [normalized]
    if "-" in normalized:
        parts.append(normalized.split("-", 1)[0])
    if "ja" not in parts:
        parts.append("ja")
    seen = set()
    ordered = []
    for item in parts:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def load_locale_table(locale: str) -> Dict[str, Any]:
    cache_key = normalize_locale(locale)
    if cache_key in _LOCALE_CACHE:
        return _LOCALE_CACHE[cache_key]
    payload: Dict[str, Any] = {}
    for candidate in locale_candidates(locale):
        path = Path("locales") / f"{candidate}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload:
            break
    _LOCALE_CACHE[cache_key] = payload
    return payload


def lookup_text(locale: str, key: str) -> Optional[str]:
    if not key:
        return None
    payload = load_locale_table(locale)
    cursor: Any = payload
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor if isinstance(cursor, str) else None


def truncate_text(text: str, limit: Optional[int]) -> str:
    if limit is None or limit <= 0:
        return text
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "..."

