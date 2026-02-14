#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert trace_v1 jsonl into replay payload for the 2D RPG-style viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN guard
        return default
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _decision_from_trace(record: Mapping[str, Any]) -> str:
    if record.get("event_type") == "world_transition":
        return "HOLD"
    accepted = (record.get("prospection") or {}).get("accepted")
    if accepted is True:
        return "PASS"
    if accepted is False:
        return "VETO"
    return "UNKNOWN"


def _collect_trace_files(trace_dir: Path) -> List[Path]:
    if trace_dir.is_file():
        return [trace_dir]
    if not trace_dir.exists():
        raise FileNotFoundError(f"trace_dir not found: {trace_dir}")
    return sorted(trace_dir.glob("*.jsonl"))


def _read_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"rules file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    if not isinstance(rules, dict):
        raise ValueError("rules file must be a JSON object")
    return rules


def _digest_payload(value: Any) -> str | None:
    if value is None:
        return None
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pick_expression_profile(
    decision: str,
    risk: float,
    uncertainty: float,
    rules: Mapping[str, Any],
) -> str:
    thresholds = rules.get("thresholds") or {}
    high_risk = _coerce_float(thresholds.get("high_risk"), 0.6)
    high_uncertainty = _coerce_float(thresholds.get("high_uncertainty"), 0.55)
    low_risk = _coerce_float(thresholds.get("low_risk"), 0.25)
    if decision == "VETO" or risk >= high_risk:
        return "tense"
    if decision == "HOLD" or uncertainty >= high_uncertainty:
        return "curious"
    if decision == "PASS" and risk <= low_risk:
        return "recover"
    return "calm"


def _update_growth(
    growth_values: MutableMapping[str, float],
    decision: str,
    risk: float,
    uncertainty: float,
    drive: float,
    growth_rules: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    axes_rules = growth_rules.get("axes") or {}
    decision_rules = growth_rules.get("decision_delta") or {}
    coupling = growth_rules.get("coupling") or {}
    max_step = _coerce_float(growth_rules.get("max_step"), 0.03)
    min_value = _coerce_float(growth_rules.get("min_value"), 0.0)
    max_value = _coerce_float(growth_rules.get("max_value"), 1.0)
    deltas: Dict[str, Dict[str, float]] = {}
    for axis_name, axis_cfg in axes_rules.items():
        if axis_name not in growth_values:
            growth_values[axis_name] = _coerce_float(axis_cfg.get("initial"), 0.5)
        base = _coerce_float(axis_cfg.get("base"), 0.0)
        ddec = _coerce_float((decision_rules.get(decision) or {}).get(axis_name), 0.0)
        drisk = _coerce_float((coupling.get("risk") or {}).get(axis_name), 0.0) * (0.5 - risk)
        dunc = _coerce_float((coupling.get("uncertainty") or {}).get(axis_name), 0.0) * (0.5 - uncertainty)
        ddrive = _coerce_float((coupling.get("drive") or {}).get(axis_name), 0.0) * (drive - 0.5)
        delta = _clamp(base + ddec + drisk + dunc + ddrive, -max_step, max_step)
        next_value = _clamp(growth_values[axis_name] + delta, min_value, max_value)
        growth_values[axis_name] = next_value
        deltas[axis_name] = {"value": round(next_value, 4), "delta": round(delta, 4)}
    return deltas


def _update_culture(
    culture_values: MutableMapping[str, float],
    *,
    decision: str,
    risk: float,
    uncertainty: float,
    drive: float,
    world_type: str,
    culture_rules: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    axes_rules = culture_rules.get("axes") or {}
    decision_rules = culture_rules.get("decision_delta") or {}
    coupling = culture_rules.get("coupling") or {}
    world_coupling = culture_rules.get("world_coupling") or {}
    max_step = _coerce_float(culture_rules.get("max_step"), 0.04)
    min_value = _coerce_float(culture_rules.get("min_value"), 0.0)
    max_value = _coerce_float(culture_rules.get("max_value"), 1.0)
    world_bias = world_coupling.get(world_type) or {}
    deltas: Dict[str, Dict[str, float]] = {}
    for axis_name, axis_cfg in axes_rules.items():
        if axis_name not in culture_values:
            culture_values[axis_name] = _coerce_float(axis_cfg.get("initial"), 0.5)
        base = _coerce_float(axis_cfg.get("base"), 0.0)
        ddec = _coerce_float((decision_rules.get(decision) or {}).get(axis_name), 0.0)
        drisk = _coerce_float((coupling.get("risk") or {}).get(axis_name), 0.0) * (0.5 - risk)
        dunc = _coerce_float((coupling.get("uncertainty") or {}).get(axis_name), 0.0) * (0.5 - uncertainty)
        ddrive = _coerce_float((coupling.get("drive") or {}).get(axis_name), 0.0) * (drive - 0.5)
        dworld = _coerce_float(world_bias.get(axis_name), 0.0)
        delta = _clamp(base + ddec + drisk + dunc + ddrive + dworld, -max_step, max_step)
        next_value = _clamp(culture_values[axis_name] + delta, min_value, max_value)
        culture_values[axis_name] = next_value
        deltas[axis_name] = {"value": round(next_value, 4), "delta": round(delta, 4)}
    return deltas


def _derive_society_metrics(
    culture_axes: Mapping[str, Dict[str, float]],
    *,
    risk: float,
    uncertainty: float,
    drive: float,
) -> Dict[str, float]:
    trust = _coerce_float((culture_axes.get("trust") or {}).get("value"), 0.5)
    reciprocity = _coerce_float((culture_axes.get("reciprocity") or {}).get("value"), 0.5)
    rigidity = _coerce_float((culture_axes.get("norm_rigidity") or {}).get("value"), 0.5)
    openness = _coerce_float((culture_axes.get("openness") or {}).get("value"), 0.5)
    cohesion = _clamp((trust + reciprocity) * 0.5, 0.0, 1.0)
    friction = _clamp(0.5 * risk + 0.3 * uncertainty + 0.3 * rigidity - 0.2 * trust, 0.0, 1.0)
    innovation = _clamp(0.55 * openness + 0.35 * drive - 0.25 * rigidity, 0.0, 1.0)
    return {
        "cohesion": round(cohesion, 4),
        "friction": round(friction, 4),
        "innovation": round(innovation, 4),
    }


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True, help="Path to trace_v1/YYYY-MM-DD or a single jsonl")
    ap.add_argument("--out", dest="out_file", default="replay.json")
    ap.add_argument(
        "--rules",
        dest="rules_file",
        default=str(root_dir / "replay_rules.json"),
        help="Replay derivation rules JSON",
    )
    args = ap.parse_args()

    rules = _read_rules(Path(args.rules_file))
    growth_rules = rules.get("growth") or {}
    expression_profiles = rules.get("expression_profiles") or {}
    emotion_profiles = rules.get("emotion_profiles") or {}
    reaction_profiles = rules.get("reaction_profiles") or {}
    stability_rules = rules.get("stability") or {}
    culture_rules = rules.get("culture") or {}

    growth_values: Dict[str, float] = {}
    culture_values: Dict[str, float] = {}
    tension_ema = _coerce_float((stability_rules.get("initial") if isinstance(stability_rules, dict) else 0.5), 0.5)
    ema_alpha = _coerce_float((stability_rules.get("ema_alpha") if isinstance(stability_rules, dict) else 0.2), 0.2)

    trace_dir = Path(args.trace_dir)
    events: List[Dict[str, Any]] = []
    for fp in _collect_trace_files(trace_dir):
        with fp.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                decision = _decision_from_trace(record)
                boundary = record.get("boundary") or {}
                reasons = boundary.get("reasons") or {}
                transition = record.get("transition") or {}
                risk = _coerce_float(reasons.get("risk"), 0.5)
                uncertainty = _coerce_float(
                    record.get("transition_uncertainty_factor", transition.get("uncertainty_factor", reasons.get("uncertainty"))),
                    0.5,
                )
                drive = _coerce_float(reasons.get("drive_norm", reasons.get("drive")), 0.5)

                profile_key = _pick_expression_profile(decision=decision, risk=risk, uncertainty=uncertainty, rules=rules)
                expression_diff = expression_profiles.get(profile_key) or expression_profiles.get("calm") or {}
                emotion_view = emotion_profiles.get(profile_key) or emotion_profiles.get("calm") or {}
                reaction_line = reaction_profiles.get(profile_key) or reaction_profiles.get("calm") or {}
                growth_axes = _update_growth(
                    growth_values=growth_values,
                    decision=decision,
                    risk=risk,
                    uncertainty=uncertainty,
                    drive=drive,
                    growth_rules=growth_rules,
                )
                culture_axes = _update_culture(
                    culture_values=culture_values,
                    decision=decision,
                    risk=risk,
                    uncertainty=uncertainty,
                    drive=drive,
                    world_type=str(record.get("world_type") or record.get("scenario_id") or "community"),
                    culture_rules=culture_rules,
                )
                society = _derive_society_metrics(
                    culture_axes,
                    risk=risk,
                    uncertainty=uncertainty,
                    drive=drive,
                )

                tension_ema = _clamp((1.0 - ema_alpha) * tension_ema + ema_alpha * risk, 0.0, 1.0)
                stability = round(_clamp(1.0 - tension_ema, 0.0, 1.0), 4)

                events.append(
                    {
                        "ts": record.get("timestamp_ms"),
                        "seed": record.get("seed"),
                        "world_type": record.get("world_type") or record.get("scenario_id"),
                        "decision": decision,
                        "risk_pre": risk,
                        "risk_post": risk,
                        "post_risk_scale": record.get("post_risk_scale"),
                        "uncertainty_factor": uncertainty,
                        "drive": drive,
                        "decision_reason": None,
                        "decision_reason_digest": _digest_payload(record.get("decision_reason")),
                        "world_snapshot_digest": _digest_payload(record.get("world_snapshot")),
                        "override_context_digest": _digest_payload(record.get("override_context")),
                        "postmortem_note_digest": _digest_payload(record.get("postmortem_note")),
                        "growth_state": {
                            "v": int((growth_rules.get("version") or 0)),
                            "axes": growth_axes,
                        },
                        "expression_diff": {
                            "v": int((expression_diff.get("version") or 0)),
                            "face": expression_diff.get("face") or {"id": "neutral", "intensity": 0.0},
                            "pose": expression_diff.get("pose") or {"id": "still", "intensity": 0.0},
                            "voice": expression_diff.get("voice") or {"id": "normal", "intensity": 0.0},
                        },
                        "emotion_view": {
                            "v": int((emotion_view.get("version") or 0)),
                            "mode": emotion_view.get("mode", "basic_emotion_plus"),
                            "primary": emotion_view.get("primary", "normal"),
                            "secondary": emotion_view.get("secondary", "flat"),
                            "stability": stability,
                            "source": "estimated_from_trace_v1",
                            "evidence": {
                                "profile": profile_key,
                                "decision": decision,
                            },
                        },
                        "reaction_line": {
                            "v": int((reaction_line.get("version") or 0)),
                            "tokens": reaction_line.get("tokens") or [],
                        },
                        "culture_state": {
                            "v": int((culture_rules.get("version") or 0)),
                            "tag": str(record.get("world_type") or record.get("scenario_id") or "default"),
                            "axes": culture_axes,
                        },
                        "agent_society": {
                            "v": 0,
                            "metrics": society,
                        },
                        "_file": fp.name,
                        "_line": line_no,
                        "_event_type": record.get("event_type"),
                    }
                )

    def sort_key(ev: Dict[str, Any]) -> tuple[bool, Any]:
        ts = ev.get("ts")
        return (ts is None, ts)

    events.sort(key=sort_key)
    out_path = Path(args.out_file)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "replay_payload.v1",
                "events": events,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_path} with {len(events)} events")


if __name__ == "__main__":
    main()
