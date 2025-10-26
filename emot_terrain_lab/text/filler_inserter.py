# -*- coding: utf-8 -*-
"""Utility helpers for inserting filler phrases."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

_SENTENCE_SPLIT = re.compile(r"([。．！？!?…
])")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_URL = re.compile(r"https?://\S+")
_LIST_LINE = re.compile(r"^\s*[-*]\s+.*$", re.MULTILINE)
_HEADING = re.compile(r"^#+\s.*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^>\s+.*$", re.MULTILINE)
_MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_MATH_INLINE = re.compile(r"\$[^$]+\$")


def scan_protected_regions(text: str) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    spans: List[Tuple[int, int]] = []
    stats: Dict[str, int] = {
        "code": 0,
        "inline_code": 0,
        "url": 0,
        "list": 0,
        "heading": 0,
        "table": 0,
        "blockquote": 0,
        "math_block": 0,
        "math_inline": 0,
    }
    for rx, key in (
        (_CODE_FENCE, "code"),
        (_INLINE_CODE, "inline_code"),
        (_URL, "url"),
        (_HEADING, "heading"),
        (_TABLE_ROW, "table"),
        (_BLOCKQUOTE, "blockquote"),
        (_MATH_BLOCK, "math_block"),
        (_MATH_INLINE, "math_inline"),
    ):
        for match in rx.finditer(text):
            spans.append((match.start(), match.end()))
            stats[key] += 1
    for match in _LIST_LINE.finditer(text):
        spans.append((match.start(), match.end()))
        stats["list"] += 1
    spans.sort()
    return spans, stats


def insert_fillers(
    text: str,
    fillers: Iterable[Tuple[str, str, str]],
) -> str:
    """Insert filler phrases into text without breaking punctuation."""
    fillers = list(fillers)
    if not fillers or not text:
        return text

    parts = _SENTENCE_SPLIT.split(text)
    if not parts:
        return text

    spans, _ = scan_protected_regions(text)
    entries = list(fillers)
    idx = 0
    offset = 0
    result_parts = list(parts)

    for i in range(0, len(parts), 2):
        orig_chunk = parts[i]
        chunk = orig_chunk
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        chunk_len = len(orig_chunk)
        chunk_range = (offset, offset + chunk_len)
        while idx < len(entries):
            kind, position, phrase = entries[idx]
            if not phrase:
                idx += 1
                continue
            if _range_overlaps(spans, chunk_range):
                idx += 1
                continue
            if position == "sentence_start":
                before = chunk
                chunk = _inject_sentence_start(chunk, phrase)
                if chunk != before:
                    idx += 1
                    break
                idx += 1
                continue
            elif position == "clause":
                new_chunk = _inject_clause(chunk, phrase)
                if new_chunk != chunk:
                    chunk = new_chunk
                    idx += 1
                    break
                idx += 1
            else:
                idx += 1
        result_parts[i] = chunk
        if punct:
            result_parts[i + 1] = punct
        offset += len(orig_chunk) + len(punct)
    return "".join(result_parts)


def to_placeholder(text: str, breaks_ms: List[int]) -> str:
    """Embed placeholder tokens ([[BRK:ms]]) for later TTS rendering."""
    if not breaks_ms or not text:
        return text
    result = text
    for ms in breaks_ms:
        token = f"[[BRK:{int(ms)}]]"
        replaced = False
        for needle in ("… ", "…　"):
            if needle in result:
                result = result.replace(needle, f"… {token} ", 1)
                replaced = True
                break
        if not replaced:
            result = f"{result} {token}"
    return result


def _inject_sentence_start(chunk: str, phrase: str) -> str:
    stripped = chunk.lstrip()
    prefix = phrase.strip() + "… "
    if stripped:
        if stripped.startswith(phrase.strip()):
            return chunk
        return prefix + stripped
    return prefix.rstrip()


def _inject_clause(chunk: str, phrase: str) -> str:
    replacement = f"、{phrase.strip()}… "
    new_chunk, count = re.subn(r"、", replacement, chunk, count=1)
    if count == 0:
        return _inject_sentence_start(chunk, phrase)
    return new_chunk


def _range_overlaps(spans: List[Tuple[int, int]], rng: Tuple[int, int]) -> bool:
    start, end = rng
    for span_start, span_end in spans:
        if span_start < end and start < span_end:
            return True
    return False


__all__ = ["insert_fillers", "to_placeholder", "scan_protected_regions"]
