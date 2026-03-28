from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from .anchor_normalization import normalize_anchor_hint


_SPACE_RE = re.compile(r"[\s\u3000]+")
_PUNCT_RE = re.compile(r"[\"'「」『』,.;:!?！？()\[\]{}<>/\\|+\-_=~^*&%$#@]+")
_QUOTED_ANCHOR_RE = re.compile(r"[「『\"]([^」』\"\n]{1,48})[」』\"]")
_DEFAULT_REOPEN_MARKERS = (
    "その続き",
    "前の話",
    "前に話した",
    "前に触れた",
    "続きを",
    "続き",
    "戻りたい",
    "戻したい",
    "また",
    "もう一度",
    "再開",
    "again",
    "continue",
    "resume",
    "back",
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


def _anchor_text(value: str, *, limit: int = 48) -> str:
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


def _quoted_history_anchor(history: Sequence[str], *, limit: int = 48) -> str:
    for item in reversed(list(history)[-5:]):
        quoted_match = _QUOTED_ANCHOR_RE.search(str(item or "").strip())
        if quoted_match:
            quoted = str(quoted_match.group(1) or "").strip()
            normalized = normalize_anchor_hint(quoted, limit=limit)
            if normalized:
                return normalized
    return ""


@dataclass(frozen=True)
class RecentDialogueState:
    state: str = "fresh_opening"
    overlap_score: float = 0.0
    reopen_pressure: float = 0.0
    thread_carry: float = 0.0
    recent_anchor: str = ""
    history_size: int = 0
    dominant_inputs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "overlap_score": round(float(self.overlap_score), 4),
            "reopen_pressure": round(float(self.reopen_pressure), 4),
            "thread_carry": round(float(self.thread_carry), 4),
            "recent_anchor": self.recent_anchor,
            "history_size": int(self.history_size),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_recent_dialogue_state(
    current_text: str,
    history: Sequence[str] | None,
    *,
    interaction_policy: Mapping[str, Any] | None = None,
) -> RecentDialogueState:
    current = _clean_text(current_text)
    raw_history = [
        str(item or "").strip()
        for item in (history or [])
        if str(item or "").strip()
    ]
    history_size = len(raw_history)
    if not current or history_size <= 0:
        return RecentDialogueState(history_size=history_size)

    overlap_score = 0.0
    recent_anchor = ""
    for item in reversed(raw_history[-5:]):
        candidate_score = _overlap_score(current, _clean_text(item))
        if candidate_score >= overlap_score:
            overlap_score = candidate_score
            recent_anchor = _anchor_text(item)

    lowered_current = str(current_text or "").lower()
    marker_hits = sum(1 for marker in _DEFAULT_REOPEN_MARKERS if marker in lowered_current)
    reopen_pressure = _clamp01(marker_hits * 0.24)
    quoted_history_anchor = _quoted_history_anchor(raw_history)
    if quoted_history_anchor and marker_hits > 0:
        recent_anchor = quoted_history_anchor
        reopen_pressure = _clamp01(max(reopen_pressure, 0.32))

    packet = dict(interaction_policy or {})
    continuity_state = ""
    if isinstance(packet.get("relational_continuity_state"), Mapping):
        continuity_state = str(
            (packet.get("relational_continuity_state") or {}).get("state") or ""
        ).strip()
    continuity_bias = 0.14 if continuity_state in {"holding_thread", "reopen_ready"} else 0.0
    thread_carry = _clamp01(overlap_score * 1.2 + reopen_pressure * 0.54 + continuity_bias)

    state = "fresh_opening"
    dominant_inputs: list[str] = []
    if overlap_score >= 0.18:
        dominant_inputs.append("history_overlap")
    if reopen_pressure >= 0.2:
        dominant_inputs.append("reopen_marker")
    if continuity_bias > 0.0:
        dominant_inputs.append("continuity_carry")
    if quoted_history_anchor and marker_hits > 0:
        dominant_inputs.append("quoted_history_anchor")
    if history_size > 0:
        dominant_inputs.append("history_available")

    if thread_carry >= 0.45 and reopen_pressure >= 0.22:
        state = "reopening_thread"
    elif thread_carry >= 0.28 and overlap_score >= 0.1:
        state = "continuing_thread"
    elif reopen_pressure >= 0.34:
        state = "reopening_thread"

    return RecentDialogueState(
        state=state,
        overlap_score=overlap_score,
        reopen_pressure=reopen_pressure,
        thread_carry=thread_carry,
        recent_anchor=recent_anchor,
        history_size=history_size,
        dominant_inputs=tuple(dominant_inputs),
    )
