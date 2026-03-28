from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from .anchor_normalization import normalize_anchor_hint


_SPACE_RE = re.compile(r"[\s\u3000]+")
_PUNCT_RE = re.compile(r"[\"'「」『』,.;:!?！？()\[\]{}<>/\\|+\-_=~^*&%$#@]+")
_QUOTED_ANCHOR_RE = re.compile(r"[「『\"]([^」』\"\n]{1,40})[」』\"]")
_DEFAULT_REVISIT_MARKERS = (
    "その続き",
    "前の話",
    "前に話した",
    "前に触れた",
    "続きを",
    "続き",
    "戻りたい",
    "戻したい",
    "again",
    "continue",
    "resume",
    "back",
)
_DEFAULT_UNRESOLVED_MARKERS = (
    "まだ",
    "引っかか",
    "言えていない",
    "言えない",
    "怖い",
    "わからない",
    "なんで",
    "なぜ",
    "どうして",
    "どう見られる",
    "why",
    "how",
    "what",
    "?",
    "？",
)
_DEFAULT_SETTLED_MARKERS = (
    "ひとまず",
    "今日はここまで",
    "ここまでで",
    "落ち着いた",
    "わかった",
    "大丈夫",
    "fine",
    "settled",
    "clear now",
    "okay now",
)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)


def _clean_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _char_ngrams(text: str, *, width: int = 2) -> set[str]:
    compact = text.replace(" ", "")
    if not compact:
        return set()
    if len(compact) <= width:
        return {compact}
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def _overlap_score(left: str, right: str) -> float:
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    shared = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    if union <= 0:
        return 0.0
    return shared / union


def _anchor_text(value: str, *, limit: int = 40) -> str:
    normalized = normalize_anchor_hint(value, limit=limit)
    if normalized:
        return normalized
    text = str(value or "").strip()
    quoted_match = _QUOTED_ANCHOR_RE.search(text)
    if quoted_match:
        text = str(quoted_match.group(1) or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _quoted_history_anchor(history: Sequence[str], *, limit: int = 40) -> str:
    for item in reversed(list(history)[-5:]):
        quoted_match = _QUOTED_ANCHOR_RE.search(str(item or "").strip())
        if quoted_match:
            quoted = str(quoted_match.group(1) or "").strip()
            normalized = normalize_anchor_hint(quoted, limit=limit)
            if normalized:
                return normalized
    return ""


@dataclass(frozen=True)
class DiscussionThreadState:
    state: str = "ambient"
    topic_anchor: str = ""
    unresolved_pressure: float = 0.0
    revisit_readiness: float = 0.0
    thread_visibility: float = 0.0
    dominant_inputs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "topic_anchor": self.topic_anchor,
            "unresolved_pressure": round(float(self.unresolved_pressure), 4),
            "revisit_readiness": round(float(self.revisit_readiness), 4),
            "thread_visibility": round(float(self.thread_visibility), 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_discussion_thread_state(
    current_text: str,
    history: Sequence[str] | None,
    *,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    interaction_policy: Mapping[str, Any] | None = None,
) -> DiscussionThreadState:
    current = str(current_text or "").strip()
    cleaned_current = _clean_text(current)
    raw_history = [
        str(item or "").strip()
        for item in (history or [])
        if str(item or "").strip()
    ]
    if not cleaned_current and not raw_history:
        return DiscussionThreadState()

    overlap = 0.0
    topic_anchor = ""
    for item in reversed(raw_history[-5:]):
        candidate_score = _overlap_score(cleaned_current, _clean_text(item))
        if candidate_score >= overlap:
            overlap = candidate_score
            topic_anchor = _anchor_text(item)

    lowered_current = current.lower()
    revisit_hits = sum(1 for marker in _DEFAULT_REVISIT_MARKERS if marker in lowered_current)
    unresolved_hits = sum(1 for marker in _DEFAULT_UNRESOLVED_MARKERS if marker in lowered_current)
    settled_hits = sum(1 for marker in _DEFAULT_SETTLED_MARKERS if marker in lowered_current)
    quoted_history_anchor = _quoted_history_anchor(raw_history)

    revisit_readiness = _clamp01(revisit_hits * 0.24 + overlap * 0.38)
    unresolved_pressure = _clamp01(unresolved_hits * 0.2 + overlap * 0.26 - settled_hits * 0.16)
    if quoted_history_anchor and revisit_hits > 0:
        topic_anchor = quoted_history_anchor
        revisit_readiness = _clamp01(max(revisit_readiness, 0.42))

    recent_state = dict(recent_dialogue_state or {})
    recent_kind = str(recent_state.get("state") or "").strip()
    recent_thread_carry = _clamp01(recent_state.get("thread_carry"))
    if recent_kind == "reopening_thread":
        revisit_readiness = _clamp01(max(revisit_readiness, recent_thread_carry * 0.92))
    elif recent_kind == "continuing_thread":
        revisit_readiness = _clamp01(max(revisit_readiness, recent_thread_carry * 0.74))

    packet = dict(interaction_policy or {})
    continuity_state = ""
    if isinstance(packet.get("relational_continuity_state"), Mapping):
        continuity_state = str(
            (packet.get("relational_continuity_state") or {}).get("state") or ""
        ).strip()
    continuity_bonus = 0.12 if continuity_state in {"holding_thread", "reopen_ready"} else 0.0
    thread_visibility = _clamp01(overlap * 0.64 + revisit_readiness * 0.42 + continuity_bonus)

    state = "ambient"
    dominant_inputs: list[str] = []
    if overlap >= 0.14:
        dominant_inputs.append("history_overlap")
    if revisit_hits > 0:
        dominant_inputs.append("revisit_marker")
    if unresolved_hits > 0:
        dominant_inputs.append("unresolved_marker")
    if settled_hits > 0:
        dominant_inputs.append("settled_marker")
    if continuity_bonus > 0.0:
        dominant_inputs.append("continuity_carry")
    if quoted_history_anchor and revisit_hits > 0:
        dominant_inputs.append("quoted_history_anchor")
    if recent_kind:
        dominant_inputs.append(f"recent_dialogue:{recent_kind}")

    if revisit_readiness >= 0.42 and unresolved_pressure >= 0.18:
        state = "revisit_issue"
    elif thread_visibility >= 0.34 and unresolved_pressure >= 0.16:
        state = "active_issue"
    elif settled_hits > 0 and unresolved_pressure <= 0.22:
        state = "settling_issue"
    elif unresolved_pressure >= 0.26:
        state = "fresh_issue"

    if not topic_anchor:
        topic_anchor = _anchor_text(current)

    return DiscussionThreadState(
        state=state,
        topic_anchor=topic_anchor,
        unresolved_pressure=unresolved_pressure,
        revisit_readiness=revisit_readiness,
        thread_visibility=thread_visibility,
        dominant_inputs=tuple(dominant_inputs),
    )
