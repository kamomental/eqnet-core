from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

DEFAULT_SURFACE_FALLBACK_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "eval" / "surface_fallbacks.json"
)


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
    surface_style: str = "plain"
    surface_style_reason: str = ""

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
            "surface_style": self.surface_style,
            "surface_style_reason": self.surface_style_reason,
        }


def load_surface_fallback_texts(path: str | Path | None = None) -> dict[str, Any]:
    fallback_path = Path(path) if path is not None else DEFAULT_SURFACE_FALLBACK_PATH
    payload = json.loads(fallback_path.read_text(encoding="utf-8"))
    if not isinstance(payload, MappingABC):
        raise ValueError("surface fallback config must be a JSON object")
    return {str(key): value for key, value in payload.items() if str(key).strip()}


def render_surface_fallback(
    surface_policy: Mapping[str, Any] | None,
    *,
    fallback_texts: Mapping[str, Any] | None = None,
) -> str:
    policy = dict(surface_policy or {})
    shape_id = _text(policy.get("fallback_shape_id"))
    style_id = _text(policy.get("surface_style") or policy.get("style_id")) or "plain"
    seed = _text(policy.get("fallback_variant_seed"))
    texts = dict(fallback_texts) if fallback_texts is not None else load_surface_fallback_texts()
    candidates = _fallback_candidates(texts, style_id=style_id, shape_id=shape_id)
    if not candidates and style_id != "plain":
        candidates = _fallback_candidates(texts, style_id="plain", shape_id=shape_id)
    if not candidates and shape_id != "minimal_ack":
        candidates = _fallback_candidates(texts, style_id=style_id, shape_id="minimal_ack")
    if not candidates:
        candidates = _fallback_candidates(texts, style_id="plain", shape_id="minimal_ack")
    if not candidates:
        return ""
    if seed:
        index = random.Random(seed).randrange(len(candidates))
        return candidates[index]
    return random.choice(candidates)


def compile_surface_policy(
    reaction_contract: Mapping[str, Any] | None,
    *,
    surface_style: str = "plain",
    surface_style_reason: str = "",
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

    if _has_forward_brightness_contradiction(
        interpretation_budget=interpretation_budget,
        shape_id=shape_id,
        strategy=_text(contract.get("strategy")),
        initiative=initiative,
        execution_mode=_text(contract.get("execution_mode")),
    ):
        max_sentences = min(max_sentences, 1)
        allowed = ["minimal_acknowledgement", "surface_mirror"]
        prohibited.extend(
            [
                "positive_spin",
                "attribute_motive",
                "summarize_as_truth",
                "normalize_or_advise",
            ]
        )
        fallback_shape_id = "low_inference_ack"
    else:
        fallback_shape_id = _fallback_shape_id(
            response_channel=response_channel,
            shape_id=shape_id,
            scale=scale,
        )

    return SurfacePolicy(
        response_channel=response_channel,
        max_sentences=max_sentences,
        question_budget=question_budget,
        interpretation_budget=interpretation_budget,
        advice_budget=advice_budget,
        brightness_budget=brightness_budget,
        allowed_acts=tuple(_dedupe(allowed)),
        prohibited_acts=tuple(_dedupe(prohibited)),
        fallback_shape_id=fallback_shape_id,
        surface_style=_text(surface_style) or "plain",
        surface_style_reason=_text(surface_style_reason),
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


def _has_forward_brightness_contradiction(
    *,
    interpretation_budget: str,
    shape_id: str,
    strategy: str,
    initiative: str,
    execution_mode: str,
) -> bool:
    if interpretation_budget != "none":
        return False
    return (
        shape_id == "bright_bounce"
        or strategy == "shared_world_next_step"
        or initiative == "co_move"
        or execution_mode == "shared_progression"
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _fallback_candidates(
    texts: Mapping[str, Any],
    *,
    style_id: str,
    shape_id: str,
) -> list[str]:
    if not shape_id:
        return []
    styles = texts.get("styles")
    if isinstance(styles, MappingABC):
        style_payload = styles.get(style_id)
        if isinstance(style_payload, MappingABC):
            candidates = _coerce_text_candidates(style_payload.get(shape_id))
            if candidates:
                return candidates
    fallbacks = texts.get("fallbacks")
    if isinstance(fallbacks, MappingABC):
        candidates = _coerce_text_candidates(fallbacks.get(shape_id))
        if candidates:
            return candidates
    return _coerce_text_candidates(texts.get(shape_id))


def _coerce_text_candidates(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [candidate for item in value if (candidate := _text(item))]
    return []


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered
