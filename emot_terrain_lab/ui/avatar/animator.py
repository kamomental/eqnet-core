from __future__ import annotations

import math
import random

from .cfg import AvatarConfig
from .contracts import AvatarInputs, AvatarState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class AvatarAnimator:
    def __init__(self, cfg: AvatarConfig) -> None:
        self._cfg = cfg
        self._time_sec = 0.0
        self._next_blink_at = 0.0
        self._blink_t = 0.0
        self._blinking = False
        self._last_speaking = False
        self._hop_t = 0.0
        self._hop_active = False
        self._schedule_next_blink()

    def _schedule_next_blink(self) -> None:
        lo, hi = self._cfg.motion.blink.base_interval_s
        self._next_blink_at = self._time_sec + random.uniform(lo, hi)

    def _update_blink(self, dt: float) -> float:
        cfg = self._cfg.motion.blink
        if (not self._blinking) and self._time_sec >= self._next_blink_at:
            self._blinking = True
            self._blink_t = 0.0
        if not self._blinking:
            return 0.0

        self._blink_t += dt
        if self._blink_t <= cfg.close_s:
            return _clamp01(self._blink_t / cfg.close_s)
        open_t = self._blink_t - cfg.close_s
        if open_t <= cfg.open_s:
            return _clamp01(1.0 - (open_t / cfg.open_s))

        self._blinking = False
        self._schedule_next_blink()
        return 0.0

    def _update_hop(self, dt: float, is_speaking: bool) -> float:
        if is_speaking and not self._last_speaking:
            self._hop_active = True
            self._hop_t = 0.0
        self._last_speaking = is_speaking
        if not self._hop_active:
            return 0.0
        self._hop_t += dt
        hop_duration = max(0.01, self._cfg.motion.hop_s)
        if self._hop_t >= hop_duration:
            self._hop_active = False
            return 0.0
        phase = self._hop_t / hop_duration
        return _clamp01(1.0 - (1.0 - phase) ** 2)

    def step(self, dt: float, inputs: AvatarInputs) -> AvatarState:
        self._time_sec += max(0.0, float(dt))
        blink = self._update_blink(dt)
        hop = self._update_hop(dt, inputs.is_speaking)
        mouth_open = _clamp01(inputs.voice_energy)
        wavy = _clamp01(0.2 + 0.8 * inputs.voice_energy)

        bob_period = max(0.05, self._cfg.motion.bob_period_s)
        bob = 0.5 + 0.5 * math.sin((self._time_sec / bob_period) * 2.0 * math.pi)

        return AvatarState(
            blink=blink,
            mouth_open=mouth_open,
            wavy=wavy,
            fang_skin=self._cfg.mouth_cfg.fang_enable,
            bob=_clamp01(bob),
            hop=_clamp01(hop),
        )

    @property
    def time_sec(self) -> float:
        return self._time_sec
