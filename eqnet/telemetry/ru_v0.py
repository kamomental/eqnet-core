# -*- coding: utf-8 -*-
"""Risk/uncertainty v0 calculator and gate."""

from __future__ import annotations

from typing import Dict, List, Tuple

POLICY_VERSION = "ru-v0.1"

R_HI = 0.70
U_HI = 0.65

UNCERT_WEIGHTS = {
    "missingness_ratio": 0.35,
    "novelty_score": 0.25,
    "model_confidence_inv": 0.25,
    "conflict_score": 0.15,
}

RISK_WEIGHTS = {
    "severity": 0.40,
    "exposure": 0.20,
    "irreversibility": 0.25,
    "compliance": 0.15,
}

SIGNALS_REQUIRED = (
    "missingness_ratio",
    "staleness_sec",
    "conflict_score",
    "model_confidence",
    "novelty_score",
    "severity_safety",
    "severity_quality",
    "severity_cost",
    "severity_trust",
    "exposure_scope",
    "exposure_freq",
    "irreversibility",
    "compliance_flag",
)

CONTEXT_REQUIRED = (
    "proposal_id",
    "proposal_type",
    "timestamp",
    "boundary_status",
    "boundary_reasons",
    "policy_version",
)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def validate_required(signals: Dict[str, float], context: Dict[str, object]) -> List[str]:
    missing: List[str] = []
    for key in SIGNALS_REQUIRED:
        if key not in signals:
            missing.append(f"signals.{key}")
    for key in CONTEXT_REQUIRED:
        if key not in context:
            missing.append(f"context.{key}")
    return missing


def calc_uncert_v0(signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    missingness = float(signals.get("missingness_ratio", 0.0))
    novelty = float(signals.get("novelty_score", 0.0))
    model_conf = float(signals.get("model_confidence", 0.5))
    conflict = float(signals.get("conflict_score", 0.0))
    model_conf_inv = 1.0 - model_conf

    u = (
        UNCERT_WEIGHTS["missingness_ratio"] * missingness
        + UNCERT_WEIGHTS["novelty_score"] * novelty
        + UNCERT_WEIGHTS["model_confidence_inv"] * model_conf_inv
        + UNCERT_WEIGHTS["conflict_score"] * conflict
    )
    components = {
        "missingness_ratio": _clamp01(missingness),
        "novelty_score": _clamp01(novelty),
        "model_confidence_inv": _clamp01(model_conf_inv),
        "conflict_score": _clamp01(conflict),
        "weights": dict(UNCERT_WEIGHTS),
    }
    return _clamp01(u), components


def calc_risk_v0(signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    severity = max(
        float(signals.get("severity_safety", 0.0)),
        float(signals.get("severity_quality", 0.0)),
        float(signals.get("severity_cost", 0.0)),
        float(signals.get("severity_trust", 0.0)),
    )
    exposure = max(
        float(signals.get("exposure_scope", 0.0)),
        float(signals.get("exposure_freq", 0.0)),
    )
    irreversibility = float(signals.get("irreversibility", 0.0))
    compliance = float(signals.get("compliance_flag", 0.0))

    r = (
        RISK_WEIGHTS["severity"] * severity
        + RISK_WEIGHTS["exposure"] * exposure
        + RISK_WEIGHTS["irreversibility"] * irreversibility
        + RISK_WEIGHTS["compliance"] * compliance
    )
    components = {
        "severity": _clamp01(severity),
        "exposure": _clamp01(exposure),
        "irreversibility": _clamp01(irreversibility),
        "compliance": _clamp01(compliance),
        "weights": dict(RISK_WEIGHTS),
    }
    return _clamp01(r), components


def gate_action_v0(
    *,
    risk: float,
    uncert: float,
    boundary_status: str,
) -> str:
    if boundary_status == "HARD_STOP":
        return "HOLD"
    if risk >= R_HI and uncert >= U_HI:
        return "HOLD"
    if risk < R_HI and uncert >= U_HI:
        return "EXPLORE"
    if risk >= R_HI and uncert < U_HI:
        return "HUMAN_CONFIRM"
    return "EXECUTE"


def build_ru_v0(
    *,
    signals: Dict[str, float],
    context: Dict[str, object],
    boundary_status: str,
) -> Dict[str, object]:
    missing = validate_required(signals, context)
    risk, risk_components = calc_risk_v0(signals)
    uncert, uncert_components = calc_uncert_v0(signals)
    action = "HOLD" if missing else gate_action_v0(risk=risk, uncert=uncert, boundary_status=boundary_status)
    return {
        "policy_version": POLICY_VERSION,
        "risk": risk,
        "uncert": uncert,
        "gate_action": action,
        "components": {
            "risk": risk_components,
            "uncert": uncert_components,
        },
        "missing_required_fields": missing,
    }
