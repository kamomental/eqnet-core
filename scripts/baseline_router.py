from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_PROBLEM_WORDS: tuple[str, ...] = (
    "疲れ",
    "しんど",
    "つら",
    "苦し",
    "不安",
    "困",
    "迷",
    "悩",
    "問題",
    "解決",
    "答え",
    "助言",
)

DEFAULT_EMOTIONAL_WORDS: tuple[str, ...] = (
    "気持ち",
    "本音",
    "意味",
    "寂し",
    "嬉し",
    "悲し",
    "怖",
    "心配",
    "不安",
    "引っか",
    "分からない",
)

DEFAULT_QUESTION_MARKERS: tuple[str, ...] = ("?", "？", "どう", "なぜ", "なんで", "かな")
DEFAULT_SHORT_TEXT_LIMIT = 34


@dataclass(frozen=True)
class BaselineRouterDecision:
    mode: str
    prompt: str
    rule_name: str


def load_router_config(path: str | Path) -> dict[str, Any]:
    router_path = Path(path)
    with router_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"router config must be a mapping: {router_path}")
    return payload


def route_baseline_prompt(
    text: str,
    config: Mapping[str, Any],
) -> BaselineRouterDecision:
    mode = _select_mode(text=text, config=config)
    prompts = config.get("prompts")
    if not isinstance(prompts, Mapping):
        raise ValueError("router config requires prompts mapping")
    prompt = prompts.get(mode)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"router mode has no prompt: {mode}")
    return BaselineRouterDecision(
        mode=mode,
        prompt=_render_router_prompt(mode=mode, instruction=prompt),
        rule_name=_select_rule_name(text=text, config=config) or "default",
    )


def _select_mode(text: str, config: Mapping[str, Any]) -> str:
    detectors = _detector_config(config)
    rules = config.get("rules", [])
    if not isinstance(rules, Sequence):
        raise ValueError("router config rules must be a sequence")
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        condition = rule.get("when", {})
        if isinstance(condition, Mapping) and _matches_condition(
            condition,
            text,
            detectors=detectors,
        ):
            mode = rule.get("mode")
            if isinstance(mode, str) and mode.strip():
                return mode
    default = config.get("default", {})
    if isinstance(default, Mapping):
        mode = default.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode
    raise ValueError("router config requires default.mode")


def _select_rule_name(text: str, config: Mapping[str, Any]) -> str:
    detectors = _detector_config(config)
    rules = config.get("rules", [])
    if not isinstance(rules, Sequence):
        return ""
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        condition = rule.get("when", {})
        if isinstance(condition, Mapping) and _matches_condition(
            condition,
            text,
            detectors=detectors,
        ):
            name = rule.get("name")
            return str(name) if name else ""
    return ""


def _matches_condition(
    condition: Mapping[str, Any],
    text: str,
    *,
    detectors: Mapping[str, Any],
) -> bool:
    normalized = text.strip()
    for key, expected in condition.items():
        if key == "any_contains":
            if not _any_contains(normalized, expected):
                return False
            continue
        if key == "ends_with":
            if not _ends_with(normalized, expected):
                return False
            continue
        if key == "contains_problem_words":
            if bool(expected) != _contains_any(
                normalized,
                _detector_tokens(detectors, "problem_words", DEFAULT_PROBLEM_WORDS),
            ):
                return False
            continue
        if key == "not_contains_question":
            if bool(expected) == _contains_question(
                normalized,
                markers=_detector_tokens(
                    detectors,
                    "question_markers",
                    DEFAULT_QUESTION_MARKERS,
                ),
            ):
                return False
            continue
        if key == "emotional_words":
            if bool(expected) != _contains_any(
                normalized,
                _detector_tokens(detectors, "emotional_words", DEFAULT_EMOTIONAL_WORDS),
            ):
                return False
            continue
        if key == "short_text":
            if bool(expected) != (len(normalized) <= _short_text_limit(detectors)):
                return False
            continue
        return False
    return True


def _any_contains(text: str, expected: Any) -> bool:
    if not isinstance(expected, Sequence) or isinstance(expected, str):
        return False
    return any(isinstance(token, str) and token in text for token in expected)


def _ends_with(text: str, expected: Any) -> bool:
    if not isinstance(expected, Sequence) or isinstance(expected, str):
        return False
    return any(isinstance(token, str) and text.endswith(token) for token in expected)


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def _contains_question(text: str, *, markers: Sequence[str]) -> bool:
    return _contains_any(text, markers)


def _detector_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    detectors = config.get("detectors", {})
    return detectors if isinstance(detectors, Mapping) else {}


def _detector_tokens(
    detectors: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    raw_tokens = detectors.get(key, fallback)
    if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, str):
        return tuple(fallback)
    tokens = tuple(str(token) for token in raw_tokens if str(token))
    return tokens or tuple(fallback)


def _short_text_limit(detectors: Mapping[str, Any]) -> int:
    raw_limit = detectors.get("short_text_limit", DEFAULT_SHORT_TEXT_LIMIT)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return DEFAULT_SHORT_TEXT_LIMIT
    return limit if limit > 0 else DEFAULT_SHORT_TEXT_LIMIT


def _render_router_prompt(*, mode: str, instruction: str) -> str:
    return (
        "Respond in natural Japanese. Follow the YAML router mode exactly.\n"
        f"router_mode: {mode}\n"
        "The router is a stateless baseline, not EQNet state.\n"
        "Instruction:\n"
        f"{instruction.strip()}"
    )
