from __future__ import annotations

import re
from typing import Iterable


_SPACE_RE = re.compile(r"[\s\u3000]+")
_CLAUSE_SPLIT_RE = re.compile(r"[。！？…]|\.{2,}|!|\?")
_QUOTED_PATTERNS = (
    re.compile(r"「([^」\n]{1,64})」"),
    re.compile(r"『([^』\n]{1,64})』"),
    re.compile(r'"([^"\n]{1,64})"'),
    re.compile(r"'([^'\n]{1,64})'"),
)
_LEADING_CONTEXT_RE = re.compile(r"^(?:前に|前の|その)\s*")
_TRAILING_TOPIC_SUFFIXES = (
    "のところなら",
    "のところを",
    "のところ",
    "について",
    "の話を",
    "の話",
    "のことを",
    "のこと",
    "のあたりから",
    "のあたり",
)
_CONTINUATION_PATTERNS = (
    re.compile(r"^(?P<core>.+?)のところなら(?:、|,).*$"),
    re.compile(r"^(?P<core>.+?)のところを(?:、|,).*$"),
    re.compile(r"^(?P<core>.+?)の続き(?:を|は)?(?:、|,).*$"),
    re.compile(r"^(?P<core>.+?)を、?いま.+$"),
    re.compile(r"^(?P<core>.+?)が、?いま.+$"),
    re.compile(r"^(?P<core>.+?)(?:、|,)\s*いま話せる.+$"),
)


def normalize_anchor_hint(value: str, *, limit: int = 48) -> str:
    text = _compact_text(value)
    if not text:
        return ""

    quoted = _extract_quoted_anchor(text)
    if quoted:
        normalized = _normalize_anchor_core(quoted)
        if normalized:
            return _truncate(normalized, limit=limit)

    clause = _first_clause(text)
    if not clause:
        return ""

    normalized = _normalize_anchor_core(clause)
    if not normalized:
        normalized = _compact_text(clause)
    return _truncate(normalized, limit=limit)


def select_anchor_hint(candidates: Iterable[str], *, limit: int = 48) -> str:
    best_value = ""
    best_score = float("-inf")
    for raw_candidate in candidates:
        raw = _compact_text(raw_candidate)
        if not raw:
            continue
        normalized = normalize_anchor_hint(raw, limit=limit)
        if not normalized:
            continue
        score = _anchor_score(raw, normalized)
        if score > best_score:
            best_value = normalized
            best_score = score
    return best_value


def _extract_quoted_anchor(text: str) -> str:
    for pattern in _QUOTED_PATTERNS:
        match = pattern.search(text)
        if match:
            return _compact_text(match.group(1))
    return ""


def _first_clause(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(compact) if part.strip()]
    return parts[0] if parts else compact


def _normalize_anchor_core(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    compact = _LEADING_CONTEXT_RE.sub("", compact).strip()
    compact = _strip_trailing_topic_suffix(compact)
    compact = _strip_continuation_frame(compact)
    compact = _LEADING_CONTEXT_RE.sub("", compact).strip()
    return compact


def _strip_trailing_topic_suffix(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    for suffix in _TRAILING_TOPIC_SUFFIXES:
        if compact.endswith(suffix):
            return compact[: -len(suffix)].strip()
    return compact


def _strip_continuation_frame(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    for pattern in _CONTINUATION_PATTERNS:
        match = pattern.match(compact)
        if match:
            return _compact_text(match.group("core"))
    return compact


def _anchor_score(raw: str, normalized: str) -> float:
    score = 0.0
    if _extract_quoted_anchor(raw):
        score += 3.0
    length = len(normalized)
    if 2 <= length <= 16:
        score += 1.8
    elif length <= 28:
        score += 1.2
    else:
        score += max(0.2, 1.0 - min(length, 64) / 80.0)
    if " " not in normalized:
        score += 0.35
    if any(normalized.endswith(suffix) for suffix in _TRAILING_TOPIC_SUFFIXES):
        score -= 0.5
    return score


def _compact_text(text: object) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip())


def _truncate(text: str, *, limit: int) -> str:
    compact = _compact_text(text)
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return compact[:limit]
    return compact[: limit - 1].rstrip() + "…"
