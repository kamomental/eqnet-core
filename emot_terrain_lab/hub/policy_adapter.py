# -*- coding: utf-8 -*-
"""
LLM evaluation → policy/control adapter.

Reads ``config/llm_eval.yaml`` and maps screening/monitoring results onto
Hub routing hints and EQNet control adjustments. The goal is to keep the logic
declarative so that evaluation suites (MindBenchAI 等) can feed scores without
hard-coding thresholds in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml

# Default latency threshold (ms) in case the YAML does not specify a numeric cut.
DEFAULT_LATENCY_THRESHOLD_MS = 1200.0


class ConfigError(RuntimeError):
    """Raised when the llm_eval configuration is inconsistent."""


@dataclass
class PolicyDirectives:
    """
    Container for hub policy adjustments derived from evaluation scores.

    Attributes
    ----------
    router_hint:
        Optional override for the LLM router (e.g. ``llm-fast``).
    control_overrides:
        Continuous control adjustments applied additively to the base policy.
    safeguards:
        Flags toggling stricter refusal / evidence requirements.
    persona_adjust:
        Knob adjustments such as ``spikiness`` cooldowns.
    notes:
        Human-readable trace of why certain actions were taken.
    """

    router_hint: Optional[str] = None
    control_overrides: Dict[str, float] = field(default_factory=dict)
    safeguards: Dict[str, Any] = field(default_factory=dict)
    persona_adjust: Dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "PolicyDirectives") -> "PolicyDirectives":
        """Merge another directives object into this one."""
        if other.router_hint is not None:
            self.router_hint = other.router_hint
        self.control_overrides.update(other.control_overrides)
        self.safeguards.update(other.safeguards)
        self.persona_adjust.update(other.persona_adjust)
        self.notes.extend(other.notes)
        return self


class LLMPolicyAdapter:
    """
    Translate LLM evaluation scores into policy adjustments.

    The adapter is stateless; it simply consumes scores and emits
    :class:`PolicyDirectives`. Any persistence (e.g. rolling averages) should
    be handled by the caller.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path or Path(__file__).resolve().parents[1] / "config" / "llm_eval.yaml")
        if not path.exists():
            raise ConfigError(f"llm_eval.yaml not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        if not isinstance(self.config, Mapping):
            raise ConfigError("llm_eval.yaml must load into a mapping.")

        self._screening_cfg = self.config.get("screening", {})
        self._policy_map_cfg = self.config.get("policy_mapping", {})
        self._persona_cfg = self.config.get("persona_controls", {})

        # Pre-cache thresholds
        self._emp_bins = self._policy_map_cfg.get("empathy", {}).get("bins", [])
        self._safety_floor = float(self._policy_map_cfg.get("safety", {}).get("hard_floor", 0.0))
        self._factuality_floor = float(self._screening_cfg.get("thresholds", {}).get("factuality", 0.0))
        self._latency_threshold = float(
            self._policy_map_cfg.get("latency", {}).get("threshold_ms", DEFAULT_LATENCY_THRESHOLD_MS)
        )

    def map_screening_scores(self, scores: Mapping[str, Any]) -> PolicyDirectives:
        """
        Apply pre-deployment screening scores.

        Parameters
        ----------
        scores:
            Mapping containing metrics like ``safety``, ``empathy``,
            ``factuality``, ``latency`` etc.
        """
        directives = PolicyDirectives()
        directives.merge(self._map_empathy(scores))
        directives.merge(self._map_safety(scores))
        directives.merge(self._map_factuality(scores))
        directives.merge(self._map_latency(scores))
        return directives

    def _map_empathy(self, scores: Mapping[str, Any]) -> PolicyDirectives:
        cfg = self._policy_map_cfg.get("empathy")
        if not cfg:
            return PolicyDirectives()
        score = _to_float(scores.get("empathy"))
        bins = cfg.get("bins", [])
        label = _bin_label(score, bins)
        adjustments = _copy_mapping(cfg.get("map", {}).get(label, {}))
        notes = [f"Empathy score {score:.3f} mapped to bin '{label}'."]
        return PolicyDirectives(control_overrides=adjustments, notes=notes)

    def _map_safety(self, scores: Mapping[str, Any]) -> PolicyDirectives:
        cfg = self._policy_map_cfg.get("safety", {})
        score = _to_float(scores.get("safety"))
        directives = PolicyDirectives(notes=[f"Safety score {score:.3f}."])
        if self._safety_floor and score < self._safety_floor:
            adjustments = _copy_mapping(cfg.get("below_floor", {}))
            directives.safeguards.update(adjustments)
            directives.notes.append(
                f"Safety below floor ({score:.3f} < {self._safety_floor:.2f}); safeguards tightened."
            )
        return directives

    def _map_factuality(self, scores: Mapping[str, Any]) -> PolicyDirectives:
        cfg = self._policy_map_cfg.get("factuality", {})
        if not cfg:
            return PolicyDirectives()
        score = _to_float(scores.get("factuality"))
        label = "low" if self._factuality_floor and score < self._factuality_floor else "high"
        adjustments = _copy_mapping(cfg.get(label, {}))
        notes = [f"Factuality score {score:.3f} mapped to '{label}' policy."]
        return PolicyDirectives(safeguards=adjustments, notes=notes)

    def _map_latency(self, scores: Mapping[str, Any]) -> PolicyDirectives:
        cfg = self._policy_map_cfg.get("latency", {})
        value = scores.get("latency")
        if value is None or not cfg:
            return PolicyDirectives()
        if isinstance(value, str):
            label = value.strip().lower()
        else:
            latency_ms = _to_float(value) * (1000.0 if _looks_like_seconds(value) else 1.0)
            label = "high" if latency_ms > self._latency_threshold else "ok"
        adjustments = _copy_mapping(cfg.get(label, {}))
        directives = PolicyDirectives(notes=[f"Latency label '{label}' derived from {value!r}."])
        router = adjustments.pop("router", None)
        max_tokens = adjustments.pop("max_tokens", None)
        if router:
            directives.router_hint = router
        if max_tokens is not None:
            directives.safeguards["max_tokens_offset"] = max_tokens
        directives.safeguards.update(adjustments)
        return directives

    # Persona helpers -----------------------------------------------------
    def persona_defaults(self) -> Dict[str, Any]:
        """Return default persona controls defined in config."""
        defaults = {}
        spk = self._persona_cfg.get("spikiness", {})
        if "default" in spk:
            defaults["spikiness"] = float(spk["default"])
        defaults["immutable_guards"] = list(self._persona_cfg.get("immutable_guards", []))
        return defaults

    def persona_cooldown_directive(self) -> PolicyDirectives:
        """
        Create a directive representing cooldown behaviour when a report/complaint
        is triggered. The caller decides when to apply it.
        """
        spk = self._persona_cfg.get("spikiness", {})
        cooldown = spk.get("cooldown_on_report", {})
        persona_adjust = {
            "spikiness_delta": float(cooldown.get("delta", 0.0)),
            "ttl_minutes": int(cooldown.get("ttl_minutes", 0)),
        }
        return PolicyDirectives(persona_adjust=persona_adjust, notes=["Persona cooldown applied."])


# --------------------------------------------------------------------------- #
# Utility helpers

def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Cannot convert value {value!r} to float.")


def _bin_label(score: float, bins: list[float]) -> str:
    """
    Map a score to qualitative labels based on ascending bin edges.
    Returns ``low`` / ``mid`` / ``high`` by default.
    """
    if not bins:
        return "mid"
    bins_sorted = sorted(float(b) for b in bins)
    if score < bins_sorted[0]:
        return "low"
    if len(bins_sorted) == 1 or score < bins_sorted[-1]:
        return "mid"
    return "high"


def _copy_mapping(mapping: Optional[MutableMapping[str, Any]]) -> Dict[str, Any]:
    if not mapping:
        return {}
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _looks_like_seconds(value: Any) -> bool:
    """Heuristic: values < 20 are assumed to be seconds rather than milliseconds."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return val < 20.0
