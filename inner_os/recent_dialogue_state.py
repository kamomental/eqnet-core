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
_DEFAULT_CONTINUATION_MARKERS = (
    "さっきの続き",
    "さっきの",
    "続きなんだけど",
    "続きだけど",
    "続きで",
    "続き",
    "そのあと",
    "あのあと",
    "このあと",
    "その後",
    "あの後",
    "この後",
    "さっき",
    "引き続き",
    "続報",
    "follow up",
)
_SMALL_SHARED_MOMENT_KINDS = {
    "laugh",
    "relief",
    "pleasant_surprise",
    "tiny_win",
}
_SMALL_SHARED_CUES = (
    "笑え",
    "笑っ",
    "ふふ",
    "ほっと",
    "和ん",
    "安心",
)
_SMALL_SHARED_OFFERS = {
    "brief_shared_smile",
    "small_shared_relief",
    "tiny_shared_win",
}


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
    lowered_current = str(current_text or "").lower()
    raw_history = [
        str(item or "").strip()
        for item in (history or [])
        if str(item or "").strip()
    ]
    history_size = len(raw_history)
    if not current:
        return RecentDialogueState(history_size=history_size)

    marker_hits = sum(1 for marker in _DEFAULT_REOPEN_MARKERS if marker in lowered_current)
    continuation_hits = sum(
        1 for marker in _DEFAULT_CONTINUATION_MARKERS if marker in lowered_current
    )
    reopen_pressure = _clamp01(marker_hits * 0.24)
    continuation_pressure = _clamp01(continuation_hits * 0.2)

    overlap_score = 0.0
    recent_anchor = ""
    for item in reversed(raw_history[-5:]):
        candidate_score = _overlap_score(current, _clean_text(item))
        if candidate_score >= overlap_score:
            overlap_score = candidate_score
            recent_anchor = _anchor_text(item)

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
    live_state = ""
    if isinstance(packet.get("live_engagement_state"), Mapping):
        live_state = str((packet.get("live_engagement_state") or {}).get("state") or "").strip()
    lightness_state = ""
    if isinstance(packet.get("lightness_budget_state"), Mapping):
        lightness_state = str((packet.get("lightness_budget_state") or {}).get("state") or "").strip()
    shared_moment_state = (
        packet.get("shared_moment_state")
        if isinstance(packet.get("shared_moment_state"), Mapping)
        else {}
    )
    shared_moment_kind = str(shared_moment_state.get("moment_kind") or "").strip()
    try:
        shared_moment_score = float(shared_moment_state.get("score") or 0.0)
    except (TypeError, ValueError):
        shared_moment_score = 0.0
    try:
        shared_moment_jointness = float(shared_moment_state.get("jointness") or 0.0)
    except (TypeError, ValueError):
        shared_moment_jointness = 0.0
    try:
        shared_moment_afterglow = float(shared_moment_state.get("afterglow") or 0.0)
    except (TypeError, ValueError):
        shared_moment_afterglow = 0.0
    utterance_reason_packet = (
        packet.get("utterance_reason_packet")
        if isinstance(packet.get("utterance_reason_packet"), Mapping)
        else {}
    )
    utterance_reason_offer = str(utterance_reason_packet.get("offer") or "").strip()
    utterance_reason_question_policy = str(
        utterance_reason_packet.get("question_policy") or ""
    ).strip()
    organism_state = (
        packet.get("organism_state")
        if isinstance(packet.get("organism_state"), Mapping)
        else {}
    )
    organism_posture = str(organism_state.get("dominant_posture") or "").strip()
    try:
        organism_play_window = float(organism_state.get("play_window") or 0.0)
    except (TypeError, ValueError):
        organism_play_window = 0.0
    try:
        organism_expressive_readiness = float(
            organism_state.get("expressive_readiness") or 0.0
        )
    except (TypeError, ValueError):
        organism_expressive_readiness = 0.0
    interaction_bright_bias = (
        0.12
        if live_state in {"pickup_comment", "riff_with_comment", "seed_topic"}
        or lightness_state in {"open_play", "warm_only", "light_ok"}
        else 0.0
    )
    shared_moment_cue_hit = any(cue in lowered_current for cue in _SMALL_SHARED_CUES)
    shared_moment_room = _clamp01(
        shared_moment_score * 0.56
        + shared_moment_jointness * 0.24
        + shared_moment_afterglow * 0.2
    )
    shared_moment_reentry = (
        (
            (
                shared_moment_kind in _SMALL_SHARED_MOMENT_KINDS
                and shared_moment_room >= 0.3
            )
            or (
                shared_moment_cue_hit
                and (
                    live_state in {"pickup_comment", "riff_with_comment", "seed_topic"}
                    or lightness_state in {"open_play", "warm_only", "light_ok"}
                )
            )
        )
        and (
            live_state in {"pickup_comment", "riff_with_comment", "seed_topic"}
            or lightness_state in {"open_play", "warm_only", "light_ok"}
            or organism_posture in {"play", "open", "attune"}
        )
        and (
            shared_moment_cue_hit
            or utterance_reason_offer in _SMALL_SHARED_OFFERS
            or utterance_reason_question_policy in {"", "none"}
        )
        and (
            organism_play_window >= 0.2
            or organism_expressive_readiness >= 0.28
            or organism_posture in {"play", "open", "attune"}
        )
    )
    shared_moment_reentry_bias = 0.18 if shared_moment_reentry else 0.0
    shared_moment_reopen_softening = 0.55 if shared_moment_reentry else 1.0
    reopen_pressure = _clamp01(reopen_pressure * shared_moment_reopen_softening)
    continuation_pressure = _clamp01(
        max(continuation_pressure, 0.24) if shared_moment_reentry else continuation_pressure
    )
    thread_carry = _clamp01(
        overlap_score * 1.2
        + reopen_pressure * 0.42
        + continuation_pressure * 0.58
        + continuity_bias
        + interaction_bright_bias
        + shared_moment_reentry_bias
    )

    state = "fresh_opening"
    dominant_inputs: list[str] = []
    if overlap_score >= 0.18:
        dominant_inputs.append("history_overlap")
    if reopen_pressure >= 0.2:
        dominant_inputs.append("reopen_marker")
    if continuation_pressure >= 0.2:
        dominant_inputs.append("continuation_marker")
    if continuity_bias > 0.0:
        dominant_inputs.append("continuity_carry")
    if interaction_bright_bias > 0.0:
        dominant_inputs.append("interaction_bright_bias")
    if shared_moment_reentry:
        dominant_inputs.append("shared_moment_reentry")
    if shared_moment_cue_hit:
        dominant_inputs.append("shared_moment_cue")
    if quoted_history_anchor and marker_hits > 0:
        dominant_inputs.append("quoted_history_anchor")
    if history_size > 0:
        dominant_inputs.append("history_available")

    has_reopen_target = bool(history_size > 0 or recent_anchor)
    if shared_moment_reentry and has_reopen_target and thread_carry >= 0.3:
        state = "continuing_thread"
    elif thread_carry >= 0.45 and reopen_pressure >= 0.22 and has_reopen_target:
        state = "reopening_thread"
    elif thread_carry >= 0.28 and overlap_score >= 0.1:
        state = "continuing_thread"
    elif thread_carry >= 0.3 and continuation_pressure >= 0.22:
        state = "continuing_thread"
    elif history_size > 0 and recent_anchor and reopen_pressure >= 0.22:
        state = "continuing_thread"
    elif reopen_pressure >= 0.34 and has_reopen_target:
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
