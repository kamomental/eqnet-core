from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _tuple_text(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(_text(item) for item in values if _text(item)))


@dataclass(frozen=True)
class ResidualReflection:
    mode: str = "none"
    focus: str = ""
    withheld_acts: tuple[str, ...] = ()
    softened_acts: tuple[str, ...] = ()
    deferred_topics: tuple[str, ...] = ()
    reason_tokens: tuple[str, ...] = ()
    strength: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_residual_reflection(
    *,
    boundary_transform: Mapping[str, Any] | None,
    conversation_contract: Mapping[str, Any] | None = None,
) -> ResidualReflection:
    transform = dict(boundary_transform or {})
    contract = dict(conversation_contract or {})
    withheld_acts = _tuple_text(transform.get("withheld_acts"))
    softened_acts = _tuple_text(transform.get("softened_acts"))
    deferred_topics = _tuple_text(transform.get("deferred_topics"))
    if not deferred_topics:
        deferred_topics = _tuple_text(contract.get("leave_closed_for_now"))

    reason_tokens = []
    if withheld_acts:
        reason_tokens.append("withheld_candidate")
    if softened_acts:
        reason_tokens.append("softened_candidate")
    if deferred_topics:
        reason_tokens.append("deferred_topic")
    if _clamp01(transform.get("residual_pressure")) >= 0.42:
        reason_tokens.append("boundary_pressure_residue")

    mode = "none"
    if withheld_acts:
        mode = "withheld"
    elif softened_acts:
        mode = "softened"
    elif deferred_topics:
        mode = "held_open"
    elif _clamp01(transform.get("residual_pressure")) >= 0.34:
        mode = "boundary_tension"

    focus = ""
    if deferred_topics:
        focus = deferred_topics[0]
    elif withheld_acts:
        focus = withheld_acts[0]
    elif softened_acts:
        focus = softened_acts[0]

    strength = _clamp01(
        max(
            _clamp01(transform.get("residual_pressure")),
            len(withheld_acts) * 0.28,
            len(softened_acts) * 0.14,
            len(deferred_topics) * 0.18,
        )
    )

    cues = [f"residual_{mode}"]
    if focus:
        cues.append("residual_focus_present")

    return ResidualReflection(
        mode=mode,
        focus=focus,
        withheld_acts=withheld_acts,
        softened_acts=softened_acts,
        deferred_topics=deferred_topics,
        reason_tokens=tuple(reason_tokens),
        strength=round(strength, 4),
        cues=tuple(dict.fromkeys(cues)),
    )
