from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SurfacePolicy:
    response_channel: str = "speak"
    max_sentences: int = 2
    question_budget: int = 0
    interpretation_budget: str = "low"
    advice_budget: int = 0
    brightness_budget: int = 0
    allowed_acts: tuple[str, ...] = field(default_factory=tuple)
    prohibited_acts: tuple[str, ...] = field(default_factory=tuple)
    fallback_shape_id: str = "minimal_ack"

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_channel": self.response_channel,
            "max_sentences": self.max_sentences,
            "question_budget": self.question_budget,
            "interpretation_budget": self.interpretation_budget,
            "advice_budget": self.advice_budget,
            "brightness_budget": self.brightness_budget,
            "allowed_acts": list(self.allowed_acts),
            "prohibited_acts": list(self.prohibited_acts),
            "fallback_shape_id": self.fallback_shape_id,
        }


def compile_surface_policy(
    reaction_contract: Mapping[str, Any] | None,
) -> SurfacePolicy:
    contract = dict(reaction_contract or {})
    response_channel = _text(contract.get("response_channel")) or "speak"
    scale = _text(contract.get("scale")) or "small"
    question_budget = max(0, int(contract.get("question_budget") or 0))
    interpretation_budget = _text(contract.get("interpretation_budget")) or "low"
    initiative = _text(contract.get("initiative"))
    distance_mode = _text(contract.get("distance_mode"))
    timing_mode = _text(contract.get("timing_mode"))
    closure_mode = _text(contract.get("closure_mode"))
    shape_id = _text(contract.get("shape_id"))

    max_sentences = _max_sentences_for_scale(scale)
    if response_channel == "hold":
        max_sentences = 0

    advice_budget = 1
    if (
        response_channel in {"hold", "backchannel"}
        or scale in {"micro", "small"}
        or initiative == "yield"
        or distance_mode in {"guarded", "steady"}
    ):
        advice_budget = 0

    brightness_budget = 1
    if (
        response_channel in {"hold", "backchannel"}
        or interpretation_budget in {"none", "low"}
        or timing_mode == "held_open"
        or closure_mode == "leave_open"
    ):
        brightness_budget = 0

    allowed: list[str] = []
    prohibited: list[str] = []
    if response_channel == "hold":
        allowed.extend(["nonverbal_presence", "wait"])
        prohibited.extend(["generate_text", "ask_question", "offer_advice"])
    else:
        allowed.append("natural_surface_text")
        if max_sentences <= 1:
            allowed.append("minimal_acknowledgement")
        if initiative in {"receive", "yield"}:
            allowed.append("user_led_response")

    if question_budget <= 0:
        prohibited.append("ask_question")
    if interpretation_budget in {"none", "low"}:
        prohibited.extend(["infer_hidden_feeling", "explain_meaning"])
    if advice_budget <= 0:
        prohibited.append("offer_advice")
    if brightness_budget <= 0:
        prohibited.append("bright_reframe")
    if initiative == "yield":
        prohibited.append("continue_conversation")
    if distance_mode == "guarded":
        prohibited.append("assistant_attractor")
    if timing_mode == "held_open":
        prohibited.append("close_or_conclude")

    return SurfacePolicy(
        response_channel=response_channel,
        max_sentences=max_sentences,
        question_budget=question_budget,
        interpretation_budget=interpretation_budget,
        advice_budget=advice_budget,
        brightness_budget=brightness_budget,
        allowed_acts=tuple(_dedupe(allowed)),
        prohibited_acts=tuple(_dedupe(prohibited)),
        fallback_shape_id=_fallback_shape_id(
            response_channel=response_channel,
            shape_id=shape_id,
            scale=scale,
        ),
    )


def _max_sentences_for_scale(scale: str) -> int:
    if scale == "micro":
        return 1
    if scale == "small":
        return 2
    if scale == "medium":
        return 3
    return 2


def _fallback_shape_id(*, response_channel: str, shape_id: str, scale: str) -> str:
    if response_channel == "hold":
        return "presence_hold"
    if shape_id:
        return f"{shape_id}_minimal"
    if scale == "micro":
        return "micro_ack"
    return "minimal_ack"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered
