from __future__ import annotations

from typing import Dict, Optional


def _extract_tag_value(context_tags: list[str] | None, prefix: str) -> str:
    if not context_tags:
        return ""
    prefix_lower = prefix.lower()
    value = ""
    for raw_tag in context_tags:
        tag = str(raw_tag or "").strip()
        if tag.lower().startswith(prefix_lower):
            value = tag.split(":", 1)[1].strip()
    return value


def extract_working_memory_seed_from_context_tags(
    context_tags: list[str] | None,
) -> Optional[Dict[str, str]]:
    focus = _extract_tag_value(context_tags, "wm_seed_focus:")
    anchor = _extract_tag_value(context_tags, "wm_seed_anchor:")
    if not focus and not anchor:
        return None
    payload: Dict[str, str] = {}
    if focus:
        payload["focus"] = focus
    if anchor:
        payload["anchor"] = anchor
    return payload


def extract_long_term_theme_from_context_tags(
    context_tags: list[str] | None,
) -> Optional[Dict[str, str]]:
    focus = _extract_tag_value(context_tags, "ltm_theme_focus:")
    anchor = _extract_tag_value(context_tags, "ltm_theme_anchor:")
    kind = _extract_tag_value(context_tags, "ltm_theme_kind:")
    summary = _extract_tag_value(context_tags, "ltm_theme_summary:")
    if not focus and not anchor and not kind and not summary:
        return None
    payload: Dict[str, str] = {}
    if focus:
        payload["focus"] = focus
    if anchor:
        payload["anchor"] = anchor
    if kind:
        payload["kind"] = kind
    if summary:
        payload["summary"] = summary
    return payload
