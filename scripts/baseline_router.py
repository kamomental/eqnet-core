from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_PROBLEM_WORDS: tuple[str, ...] = ()
DEFAULT_EMOTIONAL_WORDS: tuple[str, ...] = ()
DEFAULT_QUESTION_MARKERS: tuple[str, ...] = ("?",)
DEFAULT_SHORT_TEXT_LIMIT = 34


@dataclass(frozen=True)
class BaselineRouterDecision:
    mode: str
    prompt: str
    rule_name: str
    should_call_llm: bool
    final_action_type: str
    fixed_text: str
    constraints: dict[str, Any]


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
    rule_name = _select_rule_name(text=text, config=config) or "default"
    instruction = _mode_instruction(mode=mode, config=config)
    constraints = _mode_constraints(mode=mode, config=config)
    return BaselineRouterDecision(
        mode=mode,
        prompt=_render_router_prompt(
            mode=mode,
            rule_name=rule_name,
            instruction=instruction,
            constraints=constraints,
        ),
        rule_name=rule_name,
        should_call_llm=bool(constraints.get("call_llm", True)),
        final_action_type=str(constraints.get("final_action_type") or "speak"),
        fixed_text=str(constraints.get("fixed_text") or ""),
        constraints=dict(constraints),
    )


def _mode_instruction(*, mode: str, config: Mapping[str, Any]) -> str:
    prompts = config.get("prompts")
    if not isinstance(prompts, Mapping):
        raise ValueError("router config requires prompts mapping")
    prompt = prompts.get(mode)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"router mode has no prompt: {mode}")
    return prompt


def _mode_constraints(*, mode: str, config: Mapping[str, Any]) -> dict[str, Any]:
    controls = config.get("mode_controls", {})
    if not isinstance(controls, Mapping):
        return {}
    raw = controls.get(mode, {})
    return dict(raw) if isinstance(raw, Mapping) else {}


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


def _render_router_prompt(
    *,
    mode: str,
    rule_name: str,
    instruction: str,
    constraints: Mapping[str, Any],
) -> str:
    constraint_lines = _render_constraint_lines(constraints)
    return (
        "Respond in natural Japanese. Follow the YAML router mode exactly.\n"
        f"router_mode: {mode}\n"
        f"router_rule: {rule_name}\n"
        "The router is a stateless baseline, not EQNet state.\n"
        f"{constraint_lines}"
        "Instruction:\n"
        f"{instruction.strip()}"
    )


def _render_constraint_lines(constraints: Mapping[str, Any]) -> str:
    if not constraints:
        return ""
    lines = ["Constraints:"]
    for key, value in sorted(constraints.items()):
        if key in {"call_llm", "final_action_type", "fixed_text"}:
            continue
        if isinstance(value, Sequence) and not isinstance(value, str):
            joined = ", ".join(str(item) for item in value)
            lines.append(f"- {key}: {joined}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
