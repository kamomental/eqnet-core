from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    haystack = _text(text)
    if not haystack:
        return False
    return any(token and token in haystack for token in tokens)


@dataclass(frozen=True)
class SharedMomentState:
    state: str
    moment_kind: str
    score: float
    jointness: float
    afterglow: float
    fragility: float
    cue_text: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "moment_kind": self.moment_kind,
            "score": round(self.score, 4),
            "jointness": round(self.jointness, 4),
            "afterglow": round(self.afterglow, 4),
            "fragility": round(self.fragility, 4),
            "cue_text": self.cue_text,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_shared_moment_state(
    *,
    current_focus: str,
    current_risks: Sequence[str],
    self_state: Mapping[str, Any],
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    lightness_budget_state: Mapping[str, Any] | None = None,
) -> SharedMomentState:
    state = dict(self_state or {})
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})
    lightness = dict(lightness_budget_state or {})

    surface_user_text = _text(state.get("surface_user_text"))
    recent_history = [
        _text(item)
        for item in list(state.get("recent_dialogue_history") or [])
        if _text(item)
    ]
    haystack = "\n".join(
        text
        for text in [surface_user_text, *recent_history[-3:]]
        if text
    )

    cue_groups: list[tuple[str, tuple[str, ...], str]] = [
        (
            "laugh",
            (
                "\u7b11",
                "\u7b11\u3048",
                "\u7b11\u3063",
                "\u3075\u3075",
                "\u304f\u3059",
                "\u548c\u3093",
            ),
            "\u3061\u3087\u3063\u3068\u7b11\u3048\u305f\u3053\u3068",
        ),
        (
            "relief",
            (
                "\u307b\u3063\u3068",
                "\u6c17\u304c\u697d",
                "\u6c17\u6301\u3061\u304c\u8efd",
                "\u529b\u304c\u629c\u3051",
                "\u80a9\u306e\u529b",
                "\u843d\u3061\u7740\u3044",
            ),
            "\u5c11\u3057\u697d\u306b\u306a\u3063\u305f\u3053\u3068",
        ),
        (
            "pleasant_surprise",
            (
                "\u3044\u3044\u3053\u3068",
                "\u3088\u304b\u3063\u305f",
                "\u3046\u308c\u3057",
                "\u52a9\u304b\u3063\u305f",
                "\u3042\u308a\u304c\u305f",
                "\u3073\u3063\u304f\u308a",
            ),
            "\u3061\u3087\u3063\u3068\u3046\u308c\u3057\u3044\u3053\u3068",
        ),
    ]

    moment_kind = ""
    cue_text = ""
    dominant_inputs: list[str] = []
    for kind, tokens, cue_label in cue_groups:
        if _contains_any(haystack, tokens):
            moment_kind = kind
            cue_text = cue_label
            dominant_inputs.append(f"cue:{kind}")
            break

    lightness_state = _text(lightness.get("state"))
    lightness_room = _float01(lightness.get("banter_room"))
    recent_state_name = _text(recent.get("state"))
    continuing_thread = recent_state_name in {
        "continuing_thread",
        "bright_continuity",
        "reopening_thread",
    }
    bright_issue = _text(issue.get("state")) in {"light_tension", "bright_issue"}
    thread_visible = _text(discussion.get("state")) in {
        "revisit_issue",
        "continuing_thread",
        "open_thread",
    }
    comment_focus = _text(current_focus).startswith("comment:")
    danger_pressure = any(_text(item) == "danger" for item in list(current_risks or []))

    base_score = 0.0
    if moment_kind:
        base_score += 0.48
    if continuing_thread:
        base_score += 0.12
        dominant_inputs.append("thread:continuing")
    if thread_visible:
        base_score += 0.08
        dominant_inputs.append("thread:visible")
    if comment_focus:
        base_score += 0.08
        dominant_inputs.append("focus:comment")
    if bright_issue:
        base_score += 0.06
        dominant_inputs.append("issue:light")
    if lightness_state in {"open_play", "warm_only", "light_ok"}:
        base_score += 0.1
        dominant_inputs.append(f"lightness:{lightness_state}")
    base_score += lightness_room * 0.12
    if danger_pressure:
        base_score -= 0.18
        dominant_inputs.append("risk:danger")

    jointness = _float01(
        (0.54 if continuing_thread else 0.3)
        + (0.12 if comment_focus else 0.0)
        + (0.1 if thread_visible else 0.0)
        - (0.12 if danger_pressure else 0.0)
    )
    afterglow = _float01(
        base_score * 0.52
        + lightness_room * 0.28
        + (
            0.12
            if moment_kind == "laugh"
            else 0.1
            if moment_kind == "relief"
            else 0.08
            if moment_kind
            else 0.0
        )
    )
    fragility = _float01(
        0.16
        + (0.2 if bright_issue else 0.0)
        + (0.16 if not continuing_thread else 0.0)
        + (0.18 if danger_pressure else 0.0)
    )
    score = _float01(base_score)
    state_name = "shared_moment" if moment_kind and score >= 0.42 else "none"
    if state_name == "none":
        moment_kind = ""
        cue_text = ""
        dominant_inputs = []

    return SharedMomentState(
        state=state_name,
        moment_kind=moment_kind,
        score=score,
        jointness=jointness,
        afterglow=afterglow,
        fragility=fragility,
        cue_text=cue_text,
        dominant_inputs=dominant_inputs,
    )
