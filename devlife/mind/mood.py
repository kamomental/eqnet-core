from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AffectState:
    valence: float = 0.0
    arousal: float = 0.0
    effort: float = 0.0
    uncertainty: float = 0.0


@dataclass
class AffectUpdateIn:
    task_reward: float = 0.0
    regret: float = 0.0
    aesthetic: float = 0.0
    resonance: float = 0.0


def update_m(m: AffectState, x: AffectUpdateIn) -> AffectState:
    """Minimal affect update (bounded and simple).

    This is a placeholder suitable for initial experiments; replace with a
    learned update rule as data accrues.
    """
    v = np.clip(
        m.valence + 0.6 * x.task_reward + 0.3 * x.aesthetic + 0.2 * x.resonance - 0.4 * x.regret,
        -1.0,
        1.0,
    )
    a = np.clip(m.arousal + 0.5 * abs(x.task_reward) + 0.3 * x.resonance, -1.0, 1.0)
    e = np.clip(m.effort + 0.4 * x.regret - 0.2 * x.task_reward, -1.0, 1.0)
    u = np.clip(m.uncertainty + 0.5 * max(0.0, -x.task_reward) + 0.5 * x.regret, -1.0, 1.0)
    return AffectState(float(v), float(a), float(e), float(u))


def gate_from_m(m: AffectState) -> tuple[float, float, int, int]:
    """Return (T, top_p, pause_ms, depth) small-gain gates from affect.

    Caller should rate-limit and clip against its own bounds.
    """
    T = float(np.interp(m.arousal, [-1.0, 1.0], [0.3, 1.5]))
    top_p = float(np.interp(1.0 - m.uncertainty, [-1.0, 1.0], [0.7, 0.95]))
    pause_ms = int(np.interp(m.effort, [-1.0, 1.0], [0.0, 250.0]))
    depth = int(np.interp(m.valence, [-1.0, 1.0], [1.0, 3.0]))
    return T, top_p, pause_ms, depth


__all__ = [
    "AffectState",
    "AffectUpdateIn",
    "update_m",
    "gate_from_m",
]

