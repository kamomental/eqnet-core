# -*- coding: utf-8 -*-
"""Inner replay controller with lightweight simulate→evaluate→veto flow."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration knobs that mirror Σ→Ψ の内的リプレイ挙動."""

    theta_prep: float = 0.65
    tau_conscious_s: float = 0.32
    steps: int = 16
    w_reward: float = 1.0
    w_risk: float = 0.6
    w_daff: float = 0.4
    w_uncert: float = 0.3
    w_tom_cost: float = 0.25
    tau_execute: float = 0.0
    beta_veto: float = 1.0
    seed: int = 1337
    keep_s_trace: bool = True
    trace_keep_prob: float = 1.0


@dataclass(frozen=True)
class ReplayInputs:
    """Fast-path で集約済みの Σ 指標."""

    chaos_sens: float
    tom_cost: float
    delta_aff_abs: float
    risk: float
    uncertainty: float
    reward_estimate: float
    mood_valence: float = 0.0
    mood_arousal: float = 0.0


@dataclass(frozen=True)
class ReplayOutcome:
    """結果ログを receipt に貼り付けやすいよう整理."""

    decision: str
    felt_intent_time: float
    u_hat: float
    veto_score: float
    prep_features: Dict[str, float]
    plan_features: Dict[str, float]
    trace: Sequence[float] = field(default_factory=tuple)


class InnerReplayController:
    """Deterministic inner replay runner (ランダム性は seed に依存)."""

    def __init__(self, config: ReplayConfig | None = None, **overrides) -> None:
        if config is None:
            if overrides:
                config = ReplayConfig(**overrides)
            else:
                config = ReplayConfig()
        elif overrides:
            raise ValueError("pass overrides via ReplayConfig, not kwargs")
        self.config = config
        self._rng = random.Random(config.seed)

    def run_cycle(
        self,
        inputs: ReplayInputs,
        *,
        mono_now: float | None = None,
        wall_now: float | None = None,
    ) -> ReplayOutcome:
        cfg = self.config
        mono_now = time.perf_counter() if mono_now is None else mono_now
        wall_now = time.time() if wall_now is None else wall_now
        safe_inputs = ReplayInputs(
            chaos_sens=_clip_nonneg(inputs.chaos_sens),
            tom_cost=_clip_nonneg(inputs.tom_cost),
            delta_aff_abs=_clip_nonneg(inputs.delta_aff_abs),
            risk=_clip_nonneg(inputs.risk),
            uncertainty=_clip_nonneg(inputs.uncertainty),
            reward_estimate=_clip_nonneg(inputs.reward_estimate),
            mood_valence=float(inputs.mood_valence or 0.0),
            mood_arousal=float(inputs.mood_arousal or 0.0),
        )
        steps = max(1, int(cfg.steps))
        dt = cfg.tau_conscious_s / steps if steps else cfg.tau_conscious_s
        state = 0.0
        trace = []
        t_prep = None
        capture_trace = bool(cfg.keep_s_trace)
        prob = max(0.0, min(1.0, float(cfg.trace_keep_prob)))
        if capture_trace and prob < 1.0:
            capture_trace = self._rng.random() < prob
        for idx in range(steps):
            signal = self._signal(safe_inputs)
            noise = 0.04 * self._rng.uniform(-1.0, 1.0)
            state = max(0.0, state * 0.82 + (signal + noise) * dt)
            if capture_trace:
                trace.append(state)
            if state >= cfg.theta_prep and t_prep is None:
                t_prep = (idx + 1) * dt
        prep_offset = t_prep if t_prep is not None else cfg.tau_conscious_s
        mono_intent_time = mono_now + prep_offset
        felt_intent_time = wall_now + prep_offset
        s_max = max(trace) if trace else 0.0
        prep_features = {"t_prep": prep_offset, "s_max": s_max, "mono_at": mono_intent_time}
        prep_trace = tuple(trace) if capture_trace else ()
        u_hat = self._utility(safe_inputs)
        plan_features = {
            "reward": safe_inputs.reward_estimate,
            "risk": safe_inputs.risk,
            "delta_aff": safe_inputs.delta_aff_abs,
            "uncertainty": safe_inputs.uncertainty,
            "tom_cost": safe_inputs.tom_cost,
        }
        veto_score = self._veto_score(safe_inputs)
        eps = 1e-3
        score = u_hat - cfg.beta_veto * veto_score
        if score > cfg.tau_execute + eps:
            decision = "execute"
        elif score < cfg.tau_execute - eps:
            decision = "cancel"
        else:
            decision = "cancel"
        return ReplayOutcome(
            decision=decision,
            felt_intent_time=felt_intent_time,
            u_hat=u_hat,
            veto_score=veto_score,
            prep_features=prep_features,
            plan_features=plan_features,
            trace=prep_trace,
        )

    def _signal(self, inputs: ReplayInputs) -> float:
        return (
            0.45 * inputs.chaos_sens
            + 0.25 * max(0.0, 1.0 - inputs.tom_cost)
            + 0.15 * max(0.0, 1.0 - inputs.delta_aff_abs)
            + 0.15 * max(0.0, inputs.mood_valence + inputs.mood_arousal)
        )

    def _utility(self, inputs: ReplayInputs) -> float:
        cfg = self.config
        return (
            cfg.w_reward * inputs.reward_estimate
            - cfg.w_risk * inputs.risk
            - cfg.w_daff * inputs.delta_aff_abs
            - cfg.w_uncert * inputs.uncertainty
            - cfg.w_tom_cost * inputs.tom_cost
        )

    def _veto_score(self, inputs: ReplayInputs) -> float:
        cfg = self.config
        mood_bonus = 0.2 * (inputs.mood_valence + inputs.mood_arousal)
        penalty = (
            cfg.w_risk * inputs.risk
            + cfg.w_daff * inputs.delta_aff_abs
            + cfg.w_uncert * inputs.uncertainty
            + 0.5 * cfg.w_tom_cost * inputs.tom_cost
            - mood_bonus
        )
        return max(0.0, penalty)


def _clip_nonneg(value: float) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, value)


__all__ = [
    "InnerReplayController",
    "ReplayConfig",
    "ReplayInputs",
    "ReplayOutcome",
]
