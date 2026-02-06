from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional


_QUOTE_PATTERN = re.compile(r"[\"'`]")
_SPACE_PATTERN = re.compile(r"\s+")


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def render_recall_cue(
    result: Dict[str, Any],
    *,
    culture: str,
    max_reply_chars: int,
    cue_label_chars: int,
) -> Dict[str, Any]:
    out = dict(result)
    meta = dict(out.get("meta") or {})
    candidate = dict(out.get("candidate") or {})
    label = str(candidate.get("label") or "").strip()
    if not label:
        label = str(meta.get("summary") or "memory").strip()
    label = _QUOTE_PATTERN.sub("", label)
    label = _SPACE_PATTERN.sub(" ", label).strip()
    cue_label = _clip(label, max(4, cue_label_chars))

    if culture.lower().startswith("ja"):
        cue_text = f"そのときの空気をそっと思い出しています。合図: {cue_label}"
    else:
        cue_text = f"I'm gently recalling the feeling. Cue: {cue_label}"
    cue_text = _clip(cue_text, max(12, max_reply_chars))

    out["reply"] = cue_text
    meta["recall_render_mode"] = "cue_v1"
    meta["cue_label"] = cue_label
    out["meta"] = meta
    return out


@dataclass
class RarityBudgetState:
    day_key: str = ""
    week_key: str = ""
    day_used: int = 0
    week_used: int = 0


def apply_rarity_budget(
    result: Dict[str, Any],
    *,
    state: RarityBudgetState,
    now_utc: Optional[datetime] = None,
    daily_limit: int = 3,
    weekly_limit: int = 12,
) -> Dict[str, Any]:
    out = dict(result)
    meta = dict(out.get("meta") or {})
    now = now_utc or datetime.now(timezone.utc)
    day_key = now.strftime("%Y-%m-%d")
    iso = now.isocalendar()
    week_key = f"{iso.year}-W{iso.week:02d}"

    if state.day_key != day_key:
        state.day_key = day_key
        state.day_used = 0
    if state.week_key != week_key:
        state.week_key = week_key
        state.week_used = 0

    suppressed = False
    reason = "ok"
    reply = out.get("reply")
    has_recall = isinstance(reply, str) and bool(reply.strip())
    if has_recall:
        if state.day_used >= max(0, int(daily_limit)):
            suppressed = True
            reason = "RARITY_BUDGET_DAY_EXCEEDED"
        elif state.week_used >= max(0, int(weekly_limit)):
            suppressed = True
            reason = "RARITY_BUDGET_WEEK_EXCEEDED"
        if suppressed:
            out["reply"] = None
        else:
            state.day_used += 1
            state.week_used += 1

    meta["rarity_budget"] = {
        "day_key": state.day_key,
        "week_key": state.week_key,
        "daily_limit": int(max(0, daily_limit)),
        "weekly_limit": int(max(0, weekly_limit)),
        "day_used": int(state.day_used),
        "week_used": int(state.week_used),
        "suppressed": bool(suppressed),
        "reason": reason,
    }
    out["meta"] = meta
    return out

