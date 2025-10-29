# -*- coding: utf-8 -*-
"""
Utilities to derive persona preference profiles from free-form text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


DEFAULT_PROFILE: Dict[str, Any] = {
    "schema": "eqnet_persona.v1",
    "persona": {
        "name": "",
        "tone": "support",
        "culture": "ja-JP",
        "politeness": 0.6,
        "directness": 0.4,
        "humor": 0.1,
        "brevity": 0.6,
    },
    "preferences": {
        "advice_style": "one_point",
        "empathy_bias": 0.2,
        "taboo": [],
    },
    "constraints": {
        "always_hypothesis": True,
        "max_chars": 140,
    },
}

_TONE_KEYWORDS = {
    "casual": {"casual", "friendly", "ラフ", "くだけた", "フランク"},
    "polite": {"polite", "formal", "丁寧", "敬語", "礼儀"},
    "support": {"support", "supportive", "優しく", "そっと", "寄り添って"},
}

_ADVICE_STYLE_KEYWORDS = {
    "step_by_step": {"ステップ", "順番", "段階", "step-by-step", "step by step"},
    "compare": {"比較", "比べ", "options", "choose", "対比"},
}

_TABOO_KEYWORDS = [
    ("断定", "断定"),
    ("説教", "説教調"),
    ("命令", "命令口調"),
    ("押し付け", "押し付け"),
    ("judg", "judgment"),
    ("lectur", "lecturing"),
    ("pushy", "押し付け"),
]

_POSITIVE_POLITENESS = {"丁寧", "敬語", "polite", "courteous"}
_NEGATIVE_POLITENESS = {"砕け", "フランク", "くだけた", "casual"}

_POSITIVITY_DIRECTNESS = {"率直", "正直", "ストレート", "direct"}
_NEGATIVE_DIRECTNESS = {"婉曲", "オブラート", "やわらかく"}

_POSITIVE_HUMOR = {"ジョーク", "冗談", "ユーモア", "humor", "funny"}
_NEGATIVE_HUMOR = {"真面目", "serious", "堅く", "formal only"}

_POSITIVE_BREVITY = {"短く", "簡潔", "concise", "手短"}
_NEGATIVE_BREVITY = {"詳しく", "丁寧に説明", "long", "じっくり"}

_POSITIVE_EMPATHY = {"聴いて", "聞いて", "受け止め", "寄り添", "listen", "empath"}
_NEGATIVE_EMPATHY = {"効率", "direct answer", "すぐ答え", "即答"}

_HYPOTHESIS_EXPLICIT = {"仮説", "もし違ったら", "確認して", "check with me"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _looks_japanese(text: str) -> bool:
    for ch in text:
        if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _contains_any(text: str, keywords: List[str] | Tuple[str, ...] | set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


@dataclass
class PersonaDraft:
    profile: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.profile, allow_unicode=True, sort_keys=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")


def persona_from_text(text: str, lang_hint: Optional[str] = None) -> PersonaDraft:
    if not text:
        text = ""
    text_norm = text.lower()
    is_japanese = _looks_japanese(text)
    profile = yaml.safe_load(yaml.safe_dump(DEFAULT_PROFILE))
    notes: List[str] = []

    # Detect culture
    culture = (lang_hint or "").strip().lower()
    if not culture:
        culture = "ja-jp" if is_japanese else "en-us"
    profile["persona"]["culture"] = culture
    notes.append(f"culture inferred as {culture}")

    # Determine tone
    tone = profile["persona"]["tone"]
    for candidate, keywords in _TONE_KEYWORDS.items():
        if _contains_any(text_norm, keywords) or _contains_any(text, keywords):
            tone = candidate
    profile["persona"]["tone"] = tone
    notes.append(f"tone set to {tone}")

    # Adjust politeness
    politeness = profile["persona"]["politeness"]
    if _contains_any(text, _POSITIVE_POLITENESS) or _contains_any(text_norm, _POSITIVE_POLITENESS):
        politeness += 0.15
        notes.append("politeness increased due to polite keywords")
    if _contains_any(text, _NEGATIVE_POLITENESS) or _contains_any(text_norm, _NEGATIVE_POLITENESS):
        politeness -= 0.15
        notes.append("politeness decreased due to casual keywords")
    profile["persona"]["politeness"] = round(_clamp01(politeness), 2)

    # Adjust directness
    directness = profile["persona"]["directness"]
    if _contains_any(text, _POSITIVITY_DIRECTNESS) or _contains_any(text_norm, _POSITIVITY_DIRECTNESS):
        directness += 0.2
        notes.append("directness increased")
    if _contains_any(text, _NEGATIVE_DIRECTNESS) or _contains_any(text_norm, _NEGATIVE_DIRECTNESS):
        directness -= 0.2
        notes.append("directness decreased")
    profile["persona"]["directness"] = round(_clamp01(directness), 2)

    # Adjust humor
    humor = profile["persona"]["humor"]
    if _contains_any(text, _POSITIVE_HUMOR) or _contains_any(text_norm, _POSITIVE_HUMOR):
        humor += 0.25
        notes.append("humor increased")
    if _contains_any(text, _NEGATIVE_HUMOR) or _contains_any(text_norm, _NEGATIVE_HUMOR):
        humor -= 0.2
        notes.append("humor decreased")
    profile["persona"]["humor"] = round(_clamp01(humor), 2)

    # Adjust brevity
    brevity = profile["persona"]["brevity"]
    if _contains_any(text, _POSITIVE_BREVITY) or _contains_any(text_norm, _POSITIVE_BREVITY):
        brevity += 0.2
        notes.append("brevity increased")
    if _contains_any(text, _NEGATIVE_BREVITY) or _contains_any(text_norm, _NEGATIVE_BREVITY):
        brevity -= 0.2
        notes.append("brevity decreased")
    profile["persona"]["brevity"] = round(_clamp01(brevity), 2)

    # Advice style
    advice_style = profile["preferences"]["advice_style"]
    for style, keywords in _ADVICE_STYLE_KEYWORDS.items():
        if _contains_any(text, keywords) or _contains_any(text_norm, keywords):
            advice_style = style
            notes.append(f"advice_style set to {style}")
            break
    profile["preferences"]["advice_style"] = advice_style

    # Empathy bias
    empathy_bias = profile["preferences"]["empathy_bias"]
    if _contains_any(text, _POSITIVE_EMPATHY) or _contains_any(text_norm, _POSITIVE_EMPATHY):
        empathy_bias += 0.2
        notes.append("empathy_bias increased")
    if _contains_any(text, _NEGATIVE_EMPATHY) or _contains_any(text_norm, _NEGATIVE_EMPATHY):
        empathy_bias -= 0.15
        notes.append("empathy_bias decreased")
    profile["preferences"]["empathy_bias"] = round(_clamp01(empathy_bias), 2)

    # Taboo terms
    taboo_list = set(profile["preferences"].get("taboo", []))
    for keyword, label in _TABOO_KEYWORDS:
        if keyword in text or keyword in text_norm:
            taboo_list.add(label)
    profile["preferences"]["taboo"] = sorted(taboo_list)

    # Always hypothesis flag
    if _contains_any(text, _HYPOTHESIS_EXPLICIT) or _contains_any(text_norm, _HYPOTHESIS_EXPLICIT):
        profile["constraints"]["always_hypothesis"] = True
        notes.append("always_hypothesis confirmed by user text")

    # Max characters
    matches = list(re.finditer(r"(\d+)\s*(文字|chars?|characters?)", text, flags=re.IGNORECASE))
    if matches:
        max_chars = min(int(m.group(1)) for m in matches)
        profile["constraints"]["max_chars"] = max(max_chars, 40)
        notes.append(f"max_chars set to {profile['constraints']['max_chars']}")

    # Name detection (simple quoted string)
    m_name = re.search(r"name\s*[:：]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m_name:
        profile["persona"]["name"] = m_name.group(1).strip()
        notes.append(f"name captured as {profile['persona']['name']}")

    extras: Dict[str, Any] = {}
    return PersonaDraft(profile=profile, notes=notes, extras=extras)


__all__ = ["persona_from_text", "PersonaDraft", "DEFAULT_PROFILE"]
