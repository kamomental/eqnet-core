"""YAML-driven style filters."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from .style_state import UtteranceStyleState


def apply_style_with_rules(
    raw_text: str,
    style: UtteranceStyleState,
    style_rules: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Apply YAML-defined style rules to ``raw_text``.

    Returns (styled_text, meta). ``meta`` currently exposes ``pause_events``
    so TTS engines can insert pauses without the text containing "…" tokens.
    """

    text = raw_text
    meta: Dict[str, Any] = {"pause_events": []}

    pronoun_rules = style_rules.get("pronoun_rules", {}) or {}
    for rule in pronoun_rules.get("self", []):
        src = rule.get("from")
        dst = rule.get("to")
        if src and dst:
            text = text.replace(src, dst)
    for rule in pronoun_rules.get("other", []):
        src = rule.get("from")
        dst = rule.get("to")
        if src and dst:
            text = text.replace(src, dst)
    if not pronoun_rules.get("self"):
        text = text.replace("私", style.self_pronoun)
    if not pronoun_rules.get("other"):
        text = text.replace("あなた", style.other_pronoun)

    # Simple replacements from YAML
    for rep in style_rules.get("replacements", []):
        src = rep.get("from")
        dst = rep.get("to")
        if src and dst:
            text = text.replace(src, dst)

    sentences = [s for s in text.split("。") if s.strip()]
    new_sentences: List[str] = []
    for idx, sentence in enumerate(sentences):
        is_first = idx == 0
        is_last = idx == len(sentences) - 1

        if is_first:
            sentence = _maybe_insert_at_begin(sentence, style_rules, "fillers")
            sentence = _maybe_insert_at_begin(sentence, style_rules, "tics")
            sentence = _apply_laughter_at_begin(sentence, style_rules)
        else:
            sentence = _maybe_insert_inside(sentence, style_rules, "fillers")
            sentence = _maybe_insert_inside(sentence, style_rules, "tics")
            sentence = _apply_laughter_inside(sentence, style_rules)

        if is_last:
            sentence = _maybe_append_at_end(sentence, style_rules, "fillers")
            sentence = _maybe_append_at_end(sentence, style_rules, "tics")
            sentence = _apply_laughter_at_end(sentence, style_rules)

        new_sentences.append(sentence)

    if new_sentences:
        text = "。".join(new_sentences) + "。"

    pause_cfg = style_rules.get("pause", {})
    weight = float(pause_cfg.get("weight", 0.0) or 0.0)
    if weight > 0 and len(new_sentences) > 1:
        meta["pause_events"] = _estimate_pause_events(len(new_sentences), weight)

    return text, meta


def _maybe_insert_at_begin(sentence: str, rules: Dict[str, Any], key: str) -> str:
    cfg = (rules.get(key, {}) or {}).get("begin", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if cands and random.random() < prob:
        token = random.choice(cands)
        return f"{token}、{sentence}"
    return sentence


def _maybe_insert_inside(sentence: str, rules: Dict[str, Any], key: str) -> str:
    cfg = (rules.get(key, {}) or {}).get("middle", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if not cands or random.random() >= prob:
        return sentence
    token = random.choice(cands)
    if "、" in sentence:
        parts = sentence.split("、", 1)
        return f"{parts[0]}、{token}、{parts[1]}"
    return f"{sentence}、{token}"


def _maybe_append_at_end(sentence: str, rules: Dict[str, Any], key: str) -> str:
    cfg = (rules.get(key, {}) or {}).get("end", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if cands and random.random() < prob:
        token = random.choice(cands)
        return f"{sentence}{token}"
    return sentence


def _apply_laughter_at_begin(sentence: str, rules: Dict[str, Any]) -> str:
    cfg = (rules.get("laughter", {}) or {}).get("begin", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if cands and random.random() < prob:
        return f"{random.choice(cands)} {sentence}"
    return sentence


def _apply_laughter_inside(sentence: str, rules: Dict[str, Any]) -> str:
    cfg = (rules.get("laughter", {}) or {}).get("middle", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if cands and random.random() < prob:
        return f"{sentence} {random.choice(cands)}"
    return sentence


def _apply_laughter_at_end(sentence: str, rules: Dict[str, Any]) -> str:
    cfg = (rules.get("laughter", {}) or {}).get("end", {})
    prob = float(cfg.get("prob", 0.0) or 0.0)
    cands = cfg.get("candidates", []) or []
    if cands and random.random() < prob:
        return f"{sentence} {random.choice(cands)}"
    return sentence


def _estimate_pause_events(n_sentences: int, weight: float) -> List[Dict[str, int]]:
    events: List[Dict[str, int]] = []
    for idx in range(n_sentences - 1):
        if random.random() < weight:
            events.append({"between": idx, "pause_ms": 400})
    return events
