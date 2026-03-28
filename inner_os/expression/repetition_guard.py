from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RepetitionGuard:
    recent_text_signatures: tuple[str, ...] = ()
    suppress_exact_text_reuse: bool = True
    max_history_items: int = 3
    reasons: tuple[str, ...] = ()

    def blocks_text(self, text: str) -> bool:
        normalized = _normalize_text(text)
        if not normalized or not self.suppress_exact_text_reuse:
            return False
        return normalized in self.recent_text_signatures

    def to_dict(self) -> dict[str, object]:
        return {
            "recent_text_signatures": list(self.recent_text_signatures),
            "recent_text_count": len(self.recent_text_signatures),
            "suppress_exact_text_reuse": self.suppress_exact_text_reuse,
            "max_history_items": self.max_history_items,
            "reasons": list(self.reasons),
        }


def derive_repetition_guard(
    history: Sequence[str] | None,
    *,
    max_history_items: int = 3,
) -> RepetitionGuard:
    normalized_history = [
        normalized
        for entry in list(history or [])[-max_history_items:]
        for normalized in [_normalize_text(entry)]
        if normalized
    ]
    reasons: list[str] = []
    if normalized_history:
        reasons.append("recent_response_history")
    return RepetitionGuard(
        recent_text_signatures=tuple(normalized_history),
        suppress_exact_text_reuse=True,
        max_history_items=max_history_items,
        reasons=tuple(reasons),
    )


def coerce_repetition_guard(payload: Mapping[str, object] | None) -> RepetitionGuard:
    if payload is None:
        return RepetitionGuard()
    signatures = tuple(
        _normalize_text(item)
        for item in payload.get("recent_text_signatures") or []
        if _normalize_text(str(item or ""))
    )
    reasons = tuple(
        str(item).strip()
        for item in payload.get("reasons") or []
        if str(item).strip()
    )
    max_history_items = int(payload.get("max_history_items") or 3)
    return RepetitionGuard(
        recent_text_signatures=signatures,
        suppress_exact_text_reuse=bool(payload.get("suppress_exact_text_reuse", True)),
        max_history_items=max_history_items,
        reasons=reasons,
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())
