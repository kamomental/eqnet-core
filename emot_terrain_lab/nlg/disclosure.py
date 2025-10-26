# -*- coding: utf-8 -*-
"""Disclosure operator for Five-Sense-First candor loop."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from .metaphorizer import pick_bridges


def decide_disclosure(
    delta: float,
    top_dims: Sequence[Tuple[str, float]],
    thresholds: Dict[str, float] | None,
) -> Dict[str, object]:
    """Map scalar delta to warn/must/ask tiers."""

    thresholds = thresholds or {}
    warn = float(thresholds.get("warn", 0.25))
    must = float(thresholds.get("must", 0.45))
    ask = float(thresholds.get("ask", 0.60))
    if delta >= ask:
        level = "ask"
    elif delta >= must:
        level = "must"
    elif delta >= warn:
        level = "warn"
    else:
        level = "none"
    return {"level": level, "targets": list(top_dims)}


def craft_payload(
    level: str,
    targets: Sequence[Tuple[str, float]],
    *,
    locale: str,
    persona: str,
    templates: Dict[str, object],
    metaphors: Dict[str, object],
    ask_budget: int = 1,
) -> Dict[str, object]:
    """Return disclosure text, metaphor bridges, and optional micro-question."""

    disclosures = templates.get(locale.lower(), {}) if templates else {}
    target_names = [t[0] for t in targets if t]
    target_map = templates.get("targets_map", {}) if templates else {}
    mapped = [target_map.get(name, name) for name in target_names]
    target_label = "・".join(mapped)
    persona_block = {}
    persona_overrides = disclosures.get("persona_overrides", {}) if isinstance(disclosures, dict) else {}
    if isinstance(persona_overrides, dict):
        persona_block = persona_overrides.get(persona, {})

    message = ""
    ask_text = ""
    if level == "warn":
        template = _select_template("warn", persona_block, disclosures)
        message = template.format(targets=target_label)
    elif level == "must":
        template = _select_template("must", persona_block, disclosures)
        message = template.format(targets=target_label)
    elif level == "ask":
        template = _select_template("ask", persona_block, disclosures)
        ask_text = _pick_question(targets, templates, ask_budget)
        message = template.format(targets=target_label, ask=ask_text or "")

    bridges = pick_bridges(target_names, metaphors, max_items=2)
    payload = {"disclosure": message.strip(), "bridges": bridges, "asks": []}
    if level == "ask" and ask_text:
        payload["asks"] = [ask_text]
    return payload


def _pick_question(
    targets: Sequence[Tuple[str, float]],
    templates: Dict[str, object],
    ask_budget: int,
) -> str:
    asks = templates.get("asks", {}) if templates else {}
    if not isinstance(asks, dict) or ask_budget <= 0:
        return ""
    for name, _ in targets:
        questions = asks.get(name)
        if isinstance(questions, list) and questions:
            return str(questions[0])
    return ""


def _select_template(
    level: str,
    persona_block: Mapping[str, object],
    disclosures: Mapping[str, object],
) -> str:
    if isinstance(persona_block, dict) and level in persona_block:
        return str(persona_block.get(level, ""))
    return str(disclosures.get(level, ""))


__all__ = ["decide_disclosure", "craft_payload"]
