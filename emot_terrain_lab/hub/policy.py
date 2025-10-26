# -*- coding: utf-8 -*-
"""
Policy head scaffold.

Maps affect samples + EQNet metrics into continuous behaviour controls that the
LLM/TTS/Live2D stack can consume. The implementation here is intentionally
simple; the monotonic structure mirrors the spec so it can be replaced with a
learnt MLP later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np
from devlife.mind.mood import AffectState, gate_from_m


@dataclass
class PolicyConfig:
    """Configurable monotonic mapping parameters."""

    pause_base_ms: float = 360.0
    pause_arousal_gain: float = -180.0
    pause_sync_gain: float = 240.0  # reacts to R (synchrony)
    pause_bounds: tuple[float, float] = (180.0, 1200.0)

    temp_base: float = 0.65
    temp_valence_gain: float = -0.15
    temp_entropy_gain: float = -0.20
    temp_bounds: tuple[float, float] = (0.2, 0.95)

    top_p_base: float = 0.85
    top_p_entropy_gain: float = 0.10
    top_p_bounds: tuple[float, float] = (0.3, 0.98)

    prosody_energy_base: float = 0.0
    prosody_energy_arousal_gain: float = 0.6
    prosody_energy_sync_gain: float = -0.4
    prosody_bounds: tuple[float, float] = (-1.0, 1.0)

    prosody_f0_base: float = 0.0
    prosody_f0_valence_gain: float = 0.3
    prosody_f0_bounds: tuple[float, float] = (-1.0, 1.0)

    directness_base: float = -0.05
    directness_entropy_gain: float = -0.15
    directness_bounds: tuple[float, float] = (-0.6, 0.6)

    warmth_base: float = 0.1
    warmth_valence_gain: float = 0.35
    warmth_bounds: tuple[float, float] = (-0.2, 0.8)


@dataclass
class AffectControls:
    """Continuous controls that downstream subsystems can interpret."""

    pause_ms: float
    temperature: float
    top_p: float
    directness: float
    warmth: float
    prosody_f0_shift: float
    prosody_energy: float
    gaze_mode: Literal["engage", "yield"]
    gesture_amplitude: float


class PolicyHead:
    """Translate affect + EQNet metrics into behaviour controls."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def affect_to_controls(self, affect, metrics: Dict[str, float]) -> AffectControls:
        """
        Convert affect + metrics to monotonic control outputs.

        Parameters
        ----------
        affect:
            An object with ``valence`` / ``arousal`` attributes (e.g.,
            :class:`hub.perception.AffectSample`).
        metrics:
            Dict containing EQNet metrics (H, R, kappa, etc.). Missing keys are
            treated as neutral 0.5.
        """
        valence = float(np.clip(getattr(affect, "valence", 0.0), -1.0, 1.0))
        arousal = float(np.clip(getattr(affect, "arousal", 0.0), -1.0, 1.0))
        H = float(np.clip(metrics.get("H", 0.5), 0.0, 1.0))
        R = float(np.clip(metrics.get("R", 0.5), 0.0, 1.0))
        kappa = float(metrics.get("kappa", 0.0))

        cfg = self.config

        pause = cfg.pause_base_ms
        pause += cfg.pause_arousal_gain * arousal
        pause += cfg.pause_sync_gain * (R - 0.5)
        pause = float(np.clip(pause, *cfg.pause_bounds))

        temperature = cfg.temp_base
        temperature += cfg.temp_valence_gain * valence
        temperature += cfg.temp_entropy_gain * (0.5 - H)
        temperature = float(np.clip(temperature, *cfg.temp_bounds))

        top_p = cfg.top_p_base + cfg.top_p_entropy_gain * (0.5 - H)
        top_p = float(np.clip(top_p, *cfg.top_p_bounds))

        directness = cfg.directness_base + cfg.directness_entropy_gain * (0.5 - H)
        directness = float(np.clip(directness, *cfg.directness_bounds))

        warmth = cfg.warmth_base + cfg.warmth_valence_gain * max(valence, 0.0)
        warmth = float(np.clip(warmth, *cfg.warmth_bounds))

        prosody_energy = (
            cfg.prosody_energy_base
            + cfg.prosody_energy_arousal_gain * arousal
            + cfg.prosody_energy_sync_gain * (R - 0.5)
        )
        prosody_energy = float(np.clip(prosody_energy, *cfg.prosody_bounds))

        prosody_f0_shift = cfg.prosody_f0_base + cfg.prosody_f0_valence_gain * valence
        prosody_f0_shift = float(np.clip(prosody_f0_shift, *cfg.prosody_f0_bounds))

        # Gesture amplitude reacts to arousal but damped by |kappa| (absorbing boundary).
        gesture_amp = float(np.clip(0.5 + 0.5 * arousal - 0.3 * abs(kappa), 0.0, 1.0))

        gaze_mode: Literal["engage", "yield"] = "engage" if R < 0.7 else "yield"

        ctrls = AffectControls(
            pause_ms=pause,
            temperature=temperature,
            top_p=top_p,
            directness=directness,
            warmth=warmth,
            prosody_f0_shift=prosody_f0_shift,
            prosody_energy=prosody_energy,
            gaze_mode=gaze_mode,
            gesture_amplitude=gesture_amp,
        )

        # Optional mood gating (small-gain, metrics-sourced)
        mv = float(metrics.get("mood_v", 0.0))
        ma = float(metrics.get("mood_a", 0.0))
        me = float(metrics.get("mood_effort", 0.0))
        mu = float(metrics.get("mood_uncertainty", 0.0))
        if any(abs(x) > 1e-6 for x in (mv, ma, me, mu)):
            m = AffectState(valence=mv, arousal=ma, effort=me, uncertainty=mu)
            gT, gP, gPause, gDepth = gate_from_m(m)
            # Apply with clipping inside config bounds
            ctrls.temperature = float(np.clip(ctrls.temperature * gT, *cfg.temp_bounds))
            ctrls.top_p = float(np.clip(ctrls.top_p * gP, *cfg.top_p_bounds))
            ctrls.pause_ms = int(np.clip(ctrls.pause_ms + gPause, *cfg.pause_bounds))
            # depth is forwarded via downstream controller; here we bias directness slightly
            ctrls.directness = float(np.clip(ctrls.directness + 0.05 * (gDepth - 1), *cfg.directness_bounds))

        return ctrls
