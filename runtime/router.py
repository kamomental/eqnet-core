"""Runtime router handling autonomy and skill adapter selection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, Optional, Tuple

from emot_terrain_lab.eqcore.state import CoreState, Stance


class AutonomyLevel(IntEnum):
    """Operational autonomy levels."""

    L0 = 0  # Observe only
    L1 = 1  # Assisted
    L2 = 2  # Semi-autonomous
    L3 = 3  # Autonomous


@dataclass
class SkillSlot:
    """Registered adapter slot per skill tag."""

    name: str
    current: Optional[str] = None
    previous: Optional[str] = None
    base: Optional[str] = None
    last_update: float = field(default_factory=time.time)

    def bind(self) -> Tuple[Optional[str], Tuple[str, ...]]:
        chain = tuple(path for path in (self.current, self.previous, self.base) if path)
        adapter = chain[0] if chain else None
        return adapter, chain


@dataclass
class RouterConfig:
    """Configuration controlling autonomy transitions and LLM triggers."""

    degrade_rho: float = 1.8
    degrade_synchrony: float = 0.78
    misfire_threshold: int = 3
    upgrade_hold_seconds: float = 180.0
    degrade_cooldown_seconds: float = 480.0
    llm_events: Tuple[str, ...] = ("guide", "repair", "escalation", "long_turn")
    safe_rho: float = 1.2
    safe_synchrony: float = 0.55
    # ToM thresholds and timing for flapping control
    cooldown_s: float = 3.0
    hysteresis: float = 0.08
    trust_low: float = 0.30
    trust_high: float = 0.45
    min_hold_s: float = 6.0


@dataclass
class RouterMetrics:
    rho: float
    synchrony: float
    misfires: int = 0
    incidents: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SkillBinding:
    skill: str
    adapter_path: Optional[str]
    fallback_chain: Tuple[str, ...]
    autonomy: AutonomyLevel


class RuntimeRouter:
    """Compute autonomy level and decide which adapter/LLM to call."""

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        self.config = config or RouterConfig()
        self._level: AutonomyLevel = AutonomyLevel.L0
        self._last_level_change: float = time.time()
        self._last_degrade: float = 0.0
        self._last_external_down: float = 0.0
        self._skill_slots: Dict[str, SkillSlot] = {}

    @property
    def level(self) -> AutonomyLevel:
        return self._level

    def register_skill(self, skill: str, *, current: Optional[str] = None, previous: Optional[str] = None, base: Optional[str] = None) -> None:
        self._skill_slots[skill] = SkillSlot(name=skill, current=current, previous=previous, base=base)

    def update_skill(self, skill: str, *, current: Optional[str] = None, previous: Optional[str] = None) -> None:
        slot = self._skill_slots.setdefault(skill, SkillSlot(name=skill))
        if current is not None:
            slot.previous = slot.current
            slot.current = current
            slot.last_update = time.time()
        if previous is not None:
            slot.previous = previous

    # ------------------------------------------------------------------ autonomy
    def observe(self, core_state: CoreState, stance: Stance, metrics: RouterMetrics) -> AutonomyLevel:
        now = metrics.timestamp
        degrade_reason = self._should_degrade(metrics)
        if degrade_reason:
            self._downgrade(degrade_reason, now)
        elif self._can_upgrade(metrics, now):
            self._upgrade()
        return self._level

    def _should_degrade(self, metrics: RouterMetrics) -> Optional[str]:
        if metrics.rho > (self.config.degrade_rho + self.config.hysteresis):
            return "rho"
        if metrics.synchrony > (self.config.degrade_synchrony + self.config.hysteresis):
            return "R"
        if metrics.misfires >= self.config.misfire_threshold:
            return "misfire"
        if metrics.incidents > 0:
            return "incident"
        return None

    def _downgrade(self, reason: str, timestamp: float) -> None:
        if timestamp - self._last_degrade < self.config.degrade_cooldown_seconds:
            return
        if self._level == AutonomyLevel.L0:
            return
        self._level = AutonomyLevel(max(self._level - 1, AutonomyLevel.L0))
        self._last_level_change = timestamp
        self._last_degrade = timestamp

    def _can_upgrade(self, metrics: RouterMetrics, now: float) -> bool:
        if self._level == AutonomyLevel.L3:
            return False
        if now - self._last_level_change < self.config.upgrade_hold_seconds:
            return False
        if metrics.rho > self.config.safe_rho or metrics.synchrony > self.config.safe_synchrony:
            return False
        if metrics.misfires or metrics.incidents:
            return False
        return True

    def _upgrade(self) -> None:
        self._level = AutonomyLevel(min(self._level + 1, AutonomyLevel.L3))
        self._last_level_change = time.time()

    def force_level(self, level: AutonomyLevel) -> None:
        self._level = level
        self._last_level_change = time.time()

    # Public helpers for external control bridges
    def downshift(self, reason: str = "external", target: Optional[AutonomyLevel] = None) -> None:
        now = time.time()
        if reason == "external" and (now - self._last_external_down) < self.config.cooldown_s:
            return
        if target is not None:
            new_level = target
        else:
            new_level = AutonomyLevel(max(self._level - 1, AutonomyLevel.L0))
        if new_level == self._level:
            return
        self._level = new_level
        self._last_level_change = now
        if reason == "external":
            self._last_external_down = now

    def maybe_upshift(self, trust: float) -> None:
        """Upshift by one level when trust recovered with hysteresis and hold time."""
        now = time.time()
        # Respect minimum hold time since last downshift
        if (now - self._last_degrade) < self.config.min_hold_s and (now - self._last_external_down) < self.config.min_hold_s:
            return
        if trust >= self.config.trust_high:
            if self._level < AutonomyLevel.L3:
                self._level = AutonomyLevel(self._level + 1)
                self._last_level_change = now

    # ------------------------------------------------------------------ LLM routing
    def should_call_llm(self, event: str, stance: Stance) -> bool:
        """Decide whether to invoke the heavyweight LLM path."""
        event = event.lower()
        if self._level == AutonomyLevel.L0:
            return False
        if self._level == AutonomyLevel.L1:
            return event in ("guide", "repair")
        if self._level == AutonomyLevel.L2:
            if event in self.config.llm_events:
                return True
            return stance.mode == "guide"
        # L3
        return True

    def resolve_skill(self, skill: str) -> SkillBinding:
        slot = self._skill_slots.get(skill)
        if slot is None:
            return SkillBinding(skill=skill, adapter_path=None, fallback_chain=tuple(), autonomy=self._level)
        adapter, chain = slot.bind()
        return SkillBinding(skill=skill, adapter_path=adapter, fallback_chain=chain, autonomy=self._level)

    def list_skills(self) -> Iterable[SkillSlot]:
        return self._skill_slots.values()
