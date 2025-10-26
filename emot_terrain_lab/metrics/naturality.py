# -*- coding: utf-8 -*-
"""Naturality residual metrics between prediction and observation."""

from __future__ import annotations

from typing import Dict, Mapping


def naturality_residual(
    pred: Mapping[str, float],
    obs: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Compute weighted residual capturing deviation from expected naturality."""
    w_u = float(weights.get("U", 1.0))
    w_coh = float(weights.get("coh", 0.5))
    w_mis = float(weights.get("mis", 0.7))
    u_resid = w_u * (float(pred.get("U_top", 0.0)) - float(obs.get("U_real", 0.0))) ** 2
    coh_resid = w_coh * (float(pred.get("coh_pred", 0.0)) - float(obs.get("coh_delta", 0.0))) ** 2
    misfire = float(obs.get("misfire", 0.0))
    mis_resid = w_mis * (misfire ** 2)
    return (u_resid + coh_resid + mis_resid) ** 0.5


def residual_components(
    pred: Mapping[str, float],
    obs: Mapping[str, float],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Return component-wise residual contributions."""
    w_u = float(weights.get("U", 1.0))
    w_coh = float(weights.get("coh", 0.5))
    w_mis = float(weights.get("mis", 0.7))
    u = (float(pred.get("U_top", 0.0)) - float(obs.get("U_real", 0.0))) ** 2
    coh = (float(pred.get("coh_pred", 0.0)) - float(obs.get("coh_delta", 0.0))) ** 2
    mis = float(obs.get("misfire", 0.0)) ** 2
    return {
        "U": w_u * u,
        "coh": w_coh * coh,
        "mis": w_mis * mis,
    }


__all__ = ["naturality_residual", "residual_components"]
