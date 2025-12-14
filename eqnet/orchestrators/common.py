from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping

from eqnet.contracts import (
    PerceptPacket,
    SocialContext,
    SomaticSignals,
    WorldSummary,
)

MAPPER_VERSION = "mapper_v2"

TRACE_SECTION_ATTRS = {
    "boundary": "boundary",
    "prospection": "prospection",
    "policy": "policy",
    "qualia": "qualia",
    "invariants": "invariants",
    "self": "self_state",
    "self_state": "self_state",
}


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_float(value: Any, default: float = 0.0, *, clamp01: bool = False) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not math.isfinite(result):
            return default
        if clamp01:
            return max(0.0, min(1.0, result))
        return result
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _merge_dict(*candidates: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for cand in candidates:
        if isinstance(cand, Mapping):
            merged.update(cand)
    return merged


def _extract_somatic(payload: Mapping[str, Any]) -> SomaticSignals:
    primary = _merge_dict(
        payload.get("somatic"),
        payload.get("somatic_overrides"),
        payload.get("sensor"),
        payload.get("sensor_metrics"),
        payload.get("streaming_sensor"),
    )
    metrics = primary.get("metrics")
    if isinstance(metrics, Mapping):
        primary = {**primary, **metrics}

    extras_keys = {
        "arousal_hint",
        "stress_hint",
        "body_stress_index",
        "fatigue_hint",
        "fatigue",
        "sleep_debt",
        "jitter",
        "motion_score",
        "proximity",
        "distance",
        "range",
        "metrics",
    }
    extras: dict[str, Any] = {
        k: v for k, v in primary.items() if k not in extras_keys
    }

    stress_val = _first_not_none(
        primary.get("stress_hint"),
        primary.get("body_stress_index"),
        primary.get("stress"),
    )
    fatigue_val = _first_not_none(
        primary.get("fatigue_hint"),
        primary.get("fatigue"),
        primary.get("sleep_debt"),
    )
    jitter_val = _first_not_none(primary.get("jitter"), primary.get("motion_score"))

    proximity_val = primary.get("proximity")
    distance_val = _first_not_none(primary.get("distance"), primary.get("range"))
    proximity = _coerce_float(proximity_val, 1.0, clamp01=True)
    if proximity_val is None and distance_val is not None:
        distance_num = _coerce_float(distance_val, 0.0)
        extras["distance_raw"] = distance_val
        proximity = 1.0 / (1.0 + max(distance_num, 0.0))

    return SomaticSignals(
        arousal_hint=_coerce_float(primary.get("arousal_hint"), 0.0, clamp01=True),
        stress_hint=_coerce_float(stress_val, 0.0, clamp01=True),
        fatigue_hint=_coerce_float(fatigue_val, 0.0, clamp01=True),
        jitter=_coerce_float(jitter_val, 0.0, clamp01=True),
        proximity=_coerce_float(proximity, 1.0, clamp01=True),
        extras=extras,
    )


def _extract_context(payload: Mapping[str, Any]) -> SocialContext:
    context = _merge_dict(
        payload.get("context"),
        payload.get("social"),
        payload.get("conversation"),
    )
    mode = context.get("mode") or context.get("style") or "casual"
    offer_source = _first_not_none(context.get("offer_requested"), context.get("request"))
    offer_requested = _coerce_bool(offer_source)
    if isinstance(offer_source, str) and offer_source.lower() in {"advice", "help"}:
        offer_requested = True
    disclosure_source = _first_not_none(context.get("disclosure_budget"), context.get("intimacy"))
    extras = {
        k: v
        for k, v in context.items()
        if k not in {"mode", "style", "cultural_pressure", "offer_requested", "request", "disclosure_budget", "intimacy"}
    }
    if disclosure_source is not None and "intimacy" in context:
        extras["intimacy_raw"] = context.get("intimacy")
    return SocialContext(
        mode=str(mode),
        cultural_pressure=_coerce_float(context.get("cultural_pressure"), 0.0, clamp01=True),
        offer_requested=offer_requested,
        disclosure_budget=_coerce_float(disclosure_source, 1.0, clamp01=True),
        extras=extras,
    )


def _extract_world(payload: Mapping[str, Any]) -> WorldSummary:
    world = _merge_dict(
        payload.get("world"),
        payload.get("world_state"),
        payload.get("scene"),
        payload.get("environment"),
    )
    extras_keys = {
        "hazard_level",
        "danger",
        "threat",
        "ambiguity",
        "clarity",
        "npc_affect",
        "npc_valence",
        "social_pressure",
        "crowd_pressure",
    }
    extras = {k: v for k, v in world.items() if k not in extras_keys}

    hazard = _first_not_none(world.get("hazard_level"), world.get("danger"), world.get("threat"))
    clarity = world.get("clarity")
    ambiguity = _first_not_none(world.get("ambiguity"), world.get("uncertainty"))
    if ambiguity is None and clarity is not None:
        clarity_val = _coerce_float(clarity, 0.0, clamp01=True)
        extras["clarity_raw"] = clarity
        ambiguity = 1.0 - clarity_val
    elif clarity is not None:
        extras["clarity_raw"] = clarity
    social_pressure = _first_not_none(world.get("social_pressure"), world.get("crowd_pressure"))
    npc_affect = _first_not_none(world.get("npc_affect"), world.get("npc_valence"))

    return WorldSummary(
        hazard_level=_coerce_float(hazard, 0.0, clamp01=True),
        ambiguity=_coerce_float(ambiguity, 0.0, clamp01=True),
        npc_affect=_coerce_float(npc_affect, 0.0),
        social_pressure=_coerce_float(social_pressure, 0.0, clamp01=True),
        extras=extras,
    )


def extract_trace_observations(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    candidates = [payload.get("trace_observations"), payload.get("observations"), payload.get("runtime_observations")]
    merged: dict[str, Mapping[str, Any]] = {}
    for cand in candidates:
        if isinstance(cand, Mapping):
            for key, value in cand.items():
                if isinstance(value, Mapping):
                    section = TRACE_SECTION_ATTRS.get(key, key)
                    merged[section] = value
    return merged


def apply_trace_observations(trace: Any, observations: Mapping[str, Mapping[str, Any]], *, source: str | None = None) -> None:
    for section, payload in observations.items():
        attr_name = TRACE_SECTION_ATTRS.get(section)
        if not attr_name or not hasattr(trace, attr_name):
            continue
        target = getattr(trace, attr_name)
        if not isinstance(target, MutableMapping):
            continue
        obs_key = "observations"
        if source:
            obs = target.setdefault(obs_key, {}).setdefault(source, {})
        else:
            obs = target.setdefault(obs_key, {})
        if isinstance(obs, MutableMapping):
            obs.update(dict(payload))


def record_mapper_metadata(trace: Any) -> None:
    meta = {
        "version": MAPPER_VERSION,
        "distance_norm": {"mode": "reciprocal", "formula": "1/(1+d)"},
        "clarity_inverted": True,
        "clamp_hints": True,
        "nan_policy": "reset_to_default",
    }
    targets = []
    for section in (trace.policy, trace.qualia):
        if isinstance(section, MutableMapping):
            obs = section.setdefault("observations", {}).setdefault("mapper", {})
            if isinstance(obs, MutableMapping):
                targets.append(obs)
    for target in targets:
        target.update(meta)


def build_percept_from_payload(payload: Mapping[str, Any]) -> PerceptPacket:
    somatic = _extract_somatic(payload)
    context = _extract_context(payload)
    world = _extract_world(payload)
    return PerceptPacket(
        turn_id=str(payload.get("turn_id", "turn-0")),
        timestamp_ms=int(payload.get("timestamp_ms", 0)),
        user_text=payload.get("user_text"),
        somatic=somatic,
        context=context,
        world=world,
        seed=payload.get("seed"),
        scenario_id=payload.get("scenario_id"),
        tags=list(payload.get("tags", [])),
    )


__all__ = [
    "build_percept_from_payload",
    "extract_trace_observations",
    "apply_trace_observations",
    "record_mapper_metadata",
    "MAPPER_VERSION",
]
