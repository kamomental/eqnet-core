# -*- coding: utf-8 -*-
"""
Heuristic text-based affect estimator with coarse categorical tagging.

This module provides a fast, dependency-free approximation that is aware of
different affective tags (joy, anger, surprise, etc.) so the UI can respond
with richer acknowledgements even when no AV sensors are present.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

# --------------------------------------------------------------------------- #
# Lexicons (minimum viable; extend as needed)
# --------------------------------------------------------------------------- #

POSITIVE_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["うれしい", "嬉しい", "楽しい", "最高", "大好き", "好き", "助かった", "よかった", "ワクワク", "楽しみ"],
    "en": ["happy", "glad", "awesome", "great", "love", "excited", "joy"],
}

JOY_HINT_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["やった", "やりました", "すごい", "最高", "ハッピー", "いぇい", "やっと"],
    "en": ["yay", "hooray", "finally", "awesome"],
}

NEGATIVE_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["つらい", "辛い", "嫌い", "最悪", "悲しい", "不快", "ムカつく", "むかつく", "腹が立つ", "許せない"],
    "en": ["hate", "awful", "sad", "pain", "bad", "upset", "annoyed"],
}

ANGER_HOT_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["ふざけんな", "怒", "キレ", "許せない", "ムカつく", "腹立つ", "ぶち切れ", "ふざけるな"],
    "en": ["angry", "furious", "rage", "pissed"],
}

ANGER_QUIET_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["正直不快", "控えめに言って", "静かに怒", "納得いかない", "釈然としない", "遺憾", "残念です"],
    "en": ["disappointed", "frustrated", "quietly angry", "not okay"],
}

SURPRISE_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["びっくり", "驚いた", "まさか", "信じられない", "えっ"],
    "en": ["surprised", "wow", "no way", "unbelievable"],
}

SOFTENER_TOKENS: Dict[str, Iterable[str]] = {
    "ja": ["やんわり", "穏やかに", "落ち着いて"],
    "en": ["calmly", "gently"],
}


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def quick_text_affect_v2(text: str, lang: str = "ja") -> Tuple[Dict[str, float], str]:
    """
    Return affect metrics and a coarse affect tag.

    Parameters
    ----------
    text:
        Input utterance.
    lang:
        ISO language prefix (ja, en, ...).
    """

    normalized = text.strip()
    if not normalized:
        return {"valence": 0.0, "arousal": 0.0, "confidence": 0.0}, "neutral"

    lang_key = lang if lang in POSITIVE_TOKENS else "en"
    lower = normalized.lower()
    exclamations = normalized.count("!") + normalized.count("！")
    question = "?" in normalized or "？" in normalized

    is_pos = _contains_any(normalized, POSITIVE_TOKENS.get(lang_key, ()))
    joy_hint = _contains_any(normalized, JOY_HINT_TOKENS.get(lang_key, ()))
    is_neg = _contains_any(normalized, NEGATIVE_TOKENS.get(lang_key, ()))
    anger_hot = _contains_any(normalized, ANGER_HOT_TOKENS.get(lang_key, ()))
    anger_quiet = _contains_any(normalized, ANGER_QUIET_TOKENS.get(lang_key, ()))
    surprise_hint = _contains_any(normalized, SURPRISE_TOKENS.get(lang_key, ()))
    softener = _contains_any(normalized, SOFTENER_TOKENS.get(lang_key, ()))

    tag = "neutral"
    valence = 0.0
    arousal = 0.1
    confidence = 0.4

    if (is_pos or joy_hint) and exclamations > 0:
        tag = "happy_excited"
        valence = 0.55
        arousal = 0.60
        confidence = 0.75
    elif is_pos or joy_hint:
        tag = "calm_positive"
        valence = 0.42
        arousal = 0.25
        confidence = 0.65
    elif anger_hot or (is_neg and exclamations > 0 and not softener):
        tag = "angry_hot"
        valence = -0.65
        arousal = 0.68
        confidence = 0.8
    elif anger_quiet or (is_neg and not is_pos):
        tag = "angry_quiet"
        valence = -0.55
        arousal = 0.30 if not softener else 0.22
        confidence = 0.7
    elif exclamations > 0 or surprise_hint:
        tag = "surprise"
        valence = 0.05 if not is_neg else -0.1
        arousal = 0.55
        confidence = 0.6
    elif question:
        tag = "curious"
        valence = 0.05
        arousal = 0.35
        confidence = 0.5

    return (
        {
            "valence": float(max(-1.0, min(1.0, valence))),
            "arousal": float(max(-1.0, min(1.0, arousal))),
            "confidence": float(max(0.0, min(1.0, confidence))),
        },
        tag,
    )


def quick_text_affect(text: str, lang: str = "ja") -> Dict[str, float]:
    """Backward-compatible wrapper returning only the affect metrics."""
    affect, _ = quick_text_affect_v2(text, lang=lang)
    return affect


__all__ = ["quick_text_affect", "quick_text_affect_v2"]
