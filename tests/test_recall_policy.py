from __future__ import annotations

from datetime import datetime, timezone

from emot_terrain_lab.memory.recall_policy import (
    RarityBudgetState,
    apply_rarity_budget,
    render_recall_cue,
)


def test_render_recall_cue_avoids_verbatim_quotes() -> None:
    result = {
        "reply": "raw",
        "candidate": {"label": '2019 Kyoto trip "let us rest today"'},
        "meta": {"mode": "recall"},
    }
    rendered = render_recall_cue(
        result,
        culture="ja-JP",
        max_reply_chars=120,
        cue_label_chars=20,
    )
    reply = rendered.get("reply") or ""
    assert isinstance(reply, str)
    assert '"' not in reply
    assert "合図:" in reply
    meta = rendered.get("meta") or {}
    assert meta.get("recall_render_mode") == "cue_v1"
    assert isinstance(meta.get("cue_label"), str)


def test_apply_rarity_budget_suppresses_when_daily_limit_exceeded() -> None:
    state = RarityBudgetState()
    base = {"reply": "cue", "meta": {}}
    now = datetime(2026, 2, 6, 10, 0, tzinfo=timezone.utc)

    first = apply_rarity_budget(
        base,
        state=state,
        now_utc=now,
        daily_limit=1,
        weekly_limit=10,
    )
    assert first.get("reply") == "cue"
    first_meta = (first.get("meta") or {}).get("rarity_budget") or {}
    assert first_meta.get("suppressed") is False

    second = apply_rarity_budget(
        base,
        state=state,
        now_utc=now,
        daily_limit=1,
        weekly_limit=10,
    )
    assert second.get("reply") is None
    second_meta = (second.get("meta") or {}).get("rarity_budget") or {}
    assert second_meta.get("suppressed") is True
    assert second_meta.get("reason") == "RARITY_BUDGET_DAY_EXCEEDED"

