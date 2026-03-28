from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence


_SPACE_RE = re.compile(r"[\s\u3000]+")
_PUNCT_RE = re.compile(r"[\"',.;:!?()\[\]{}<>/\\|+\-_=~^*&%$#@]+")
_DEFAULT_QUESTION_MARKERS = (
    "どう",
    "何",
    "なぜ",
    "どこ",
    "どれ",
    "どうして",
    "どうやって",
    "聞きたい",
    "教えて",
    "?",
    "？",
)
_DEFAULT_PAUSE_MARKERS = (
    "いったん",
    "ひとまず",
    "また",
    "あとで",
    "今はここまで",
    "ここまで",
    "置いて",
    "保留",
    "まだ整理できない",
)
_DEFAULT_RESOLUTION_MARKERS = (
    "わかった",
    "整理できた",
    "見えてきた",
    "大丈夫",
    "決まった",
    "落ち着いた",
    "平気",
    "ひとまず大丈夫",
    "fine",
    "got it",
    "clear now",
    "settled",
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
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class IssueState:
    state: str = "ambient"
    issue_anchor: str = ""
    question_pressure: float = 0.0
    pause_readiness: float = 0.0
    resolution_readiness: float = 0.0
    dominant_inputs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "issue_anchor": self.issue_anchor,
            "question_pressure": round(float(self.question_pressure), 4),
            "pause_readiness": round(float(self.pause_readiness), 4),
            "resolution_readiness": round(float(self.resolution_readiness), 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_issue_state(
    current_text: str,
    history: Sequence[str] | None,
    *,
    discussion_thread_state: Mapping[str, Any] | None = None,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    interaction_policy: Mapping[str, Any] | None = None,
) -> IssueState:
    current = str(current_text or "").strip()
    cleaned_current = _clean_text(current)
    raw_history = [
        str(item or "").strip()
        for item in (history or [])
        if str(item or "").strip()
    ]
    if not cleaned_current and not raw_history:
        return IssueState()

    overlap = 0.0
    history_anchor = ""
    for item in reversed(raw_history[-5:]):
        candidate_score = _overlap_score(cleaned_current, _clean_text(item))
        if candidate_score >= overlap:
            overlap = candidate_score
            history_anchor = _anchor_text(item)

    lowered = current.lower()
    question_hits = sum(
        1 for marker in _DEFAULT_QUESTION_MARKERS if marker in lowered
    )
    pause_hits = sum(
        1 for marker in _DEFAULT_PAUSE_MARKERS if marker in lowered
    )
    resolution_hits = sum(
        1 for marker in _DEFAULT_RESOLUTION_MARKERS if marker in lowered
    )

    discussion = dict(discussion_thread_state or {})
    discussion_state = str(discussion.get("state") or "").strip()
    discussion_unresolved = _clamp01(discussion.get("unresolved_pressure"))
    discussion_revisit = _clamp01(discussion.get("revisit_readiness"))

    recent = dict(recent_dialogue_state or {})
    recent_state = str(recent.get("state") or "").strip()
    recent_thread_carry = _clamp01(recent.get("thread_carry"))

    packet = dict(interaction_policy or {})
    continuity_state = ""
    if isinstance(packet.get("relational_continuity_state"), Mapping):
        continuity_state = str(
            (packet.get("relational_continuity_state") or {}).get("state") or ""
        ).strip()

    question_pressure = _clamp01(
        question_hits * 0.22
        + overlap * 0.18
        + (0.14 if discussion_state in {"active_issue", "fresh_issue", "revisit_issue"} else 0.0)
    )
    pause_readiness = _clamp01(
        pause_hits * 0.24
        + (0.18 if discussion_state in {"revisit_issue", "settling_issue"} else 0.0)
        + (0.16 if recent_state == "reopening_thread" else 0.0)
        + recent_thread_carry * 0.16
    )
    resolution_readiness = _clamp01(
        resolution_hits * 0.24
        + (0.18 if discussion_state == "settling_issue" else 0.0)
        + (0.08 if continuity_state in {"holding_thread", "reopen_ready"} else 0.0)
        + max(0.0, 0.18 - discussion_unresolved * 0.16)
    )

    state = "ambient"
    dominant_inputs: list[str] = []
    if overlap >= 0.14:
        dominant_inputs.append("history_overlap")
    if question_hits > 0:
        dominant_inputs.append("question_marker")
    if pause_hits > 0:
        dominant_inputs.append("pause_marker")
    if resolution_hits > 0:
        dominant_inputs.append("resolution_marker")
    if discussion_state:
        dominant_inputs.append(f"discussion:{discussion_state}")
    if recent_state:
        dominant_inputs.append(f"recent_dialogue:{recent_state}")

    if resolution_readiness >= 0.42 and question_pressure <= 0.3:
        state = "resolving_issue"
    elif pause_readiness >= 0.38 and discussion_state in {
        "revisit_issue",
        "active_issue",
        "fresh_issue",
        "settling_issue",
    }:
        state = "pausing_issue"
    elif question_pressure >= 0.4 and (
        discussion_state in {"revisit_issue", "active_issue", "fresh_issue"}
        or discussion_unresolved >= 0.28
        or discussion_revisit >= 0.34
    ):
        state = "exploring_issue"
    elif question_pressure >= 0.28:
        state = "naming_issue"

    issue_anchor = _anchor_text(
        str(discussion.get("topic_anchor") or history_anchor or current)
    )

    return IssueState(
        state=state,
        issue_anchor=issue_anchor,
        question_pressure=question_pressure,
        pause_readiness=pause_readiness,
        resolution_readiness=resolution_readiness,
        dominant_inputs=tuple(dominant_inputs),
    )
