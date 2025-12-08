# -*- coding: utf-8 -*-
"""Simplified MCP controller for the run_quick_loop lab environment."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

from devlife.metrics import kpi as kpi_mod
from devlife.mind.self_model import SelfReporter

try:  # pragma: no cover - optional in tests
    from emot_terrain_lab.utils.fastpath_config import load_fastpath_cfg
except Exception:  # pragma: no cover
    def load_fastpath_cfg(*_, **__):  # type: ignore
        return {}


@dataclass
class MCPConfig:
    """Thresholds mirrored from docs/eqnet_gap_analysis Section 5."""

    window: int = 64
    cooldown_steps: int = 8
    body_R_warn: float = 0.35
    body_rho_warn: float = 0.80
    love_low: float = 0.45
    love_high: float = 0.82
    trust_low: float = 0.30
    safety_floor: float = 0.05


class FastpathReleaseTracker:
    """Tracks which fast-path profiles may override the style layer."""

    def __init__(
        self,
        *,
        eligible_profiles: Optional[Iterable[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        log_path: Path = Path("logs/fastpath_state.jsonl"),
        experiment_tag: Optional[str] = None,
    ) -> None:
        cfg = dict(config or load_fastpath_cfg())
        release_cfg = cfg.get("profile_releases") or {}
        if eligible_profiles is None:
            eligible_profiles = [
                name for name, meta in release_cfg.items() if meta.get("release_flag")
            ]
        self.eligible_profiles: List[str] = [str(p) for p in eligible_profiles or []]
        self.mode = str(cfg.get("enforce_actions", "record_only")).lower()
        self._events: Deque[int] = deque(maxlen=100)
        self.log_path = log_path
        self.experiment_tag = experiment_tag

    @property
    def eligible(self) -> bool:
        return bool(self.eligible_profiles)

    def pick_profile(self) -> Optional[str]:
        return self.eligible_profiles[0] if self.eligible_profiles else None

    def register_event(self, episode: Dict[str, Any], override: bool) -> None:
        self._events.append(1 if override else 0)
        payload = {
            "type": "fastpath.event",
            "episode_id": episode.get("episode_id"),
            "stage": episode.get("stage"),
            "step": episode.get("step"),
            "override": override,
            "override_rate": self.override_rate,
        }
        if self.experiment_tag:
            payload["tag"] = self.experiment_tag
        self._write(payload)

    @property
    def override_rate(self) -> float:
        if not self._events:
            return 0.0
        return float(sum(self._events) / len(self._events))

    def _write(self, payload: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MCPController:
    """Bridges episode logs -> KPI rollup -> MCP actions -> learner hooks."""

    def __init__(
        self,
        *,
        config: Optional[MCPConfig] = None,
        learner_hooks: Optional[Any] = None,
        self_reporter: Optional[SelfReporter] = None,
        router: Optional[Any] = None,
        eligible_fastpath_profiles: Optional[Iterable[str]] = None,
        kpi_log_path: Path = Path("logs/kpi_rollup.jsonl"),
        action_log_path: Path = Path("logs/mcp_actions.jsonl"),
        experiment_tag: Optional[str] = None,
    ) -> None:
        self.config = config or MCPConfig()
        self.learner_hooks = learner_hooks
        self.self_reporter = self_reporter
        self.router = router
        self._episodes: Deque[Dict[str, Any]] = deque(maxlen=self.config.window)
        self.kpi_log_path = kpi_log_path
        self.action_log_path = action_log_path
        self._last_action_step: Dict[str, int] = {}
        self.experiment_tag = experiment_tag
        self.fastpath = FastpathReleaseTracker(
            eligible_profiles=eligible_fastpath_profiles,
            experiment_tag=experiment_tag,
        )

    def handle_episode(self, episode: Dict[str, Any]) -> None:
        """Feed one DevelopmentLoop record through the MCP pipeline."""

        self._episodes.append(dict(episode))
        metrics = kpi_mod.compute_all(list(self._episodes))
        metrics.setdefault("fastpath.override_rate", self.fastpath.override_rate)
        self._log_kpis(episode, metrics)
        self._evaluate_thresholds(episode, metrics)
        if self.self_reporter is not None:
            self.self_reporter.log_episode(episode, metrics, tag=self.experiment_tag)

    # ------------------------------------------------------------------ logging
    def _log_kpis(self, episode: Dict[str, Any], metrics: Dict[str, float]) -> None:
        payload = {
            "ts": time.time(),
            "episode_id": episode.get("episode_id"),
            "stage": episode.get("stage"),
            "step": episode.get("step"),
            "metrics": metrics,
        }
        if self.experiment_tag:
            payload["tag"] = self.experiment_tag
        self._write_json(self.kpi_log_path, payload)

    def _emit_action(
        self,
        tag: str,
        action: str,
        episode: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "ts": time.time(),
            "episode_id": episode.get("episode_id"),
            "stage": episode.get("stage"),
            "step": episode.get("step"),
            "kpi": tag,
            "action": action,
        }
        if self.experiment_tag:
            payload["tag"] = self.experiment_tag
        if details:
            payload.update(details)
        self._write_json(self.action_log_path, payload)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- thresholds
    def _evaluate_thresholds(self, episode: Dict[str, Any], metrics: Dict[str, float]) -> None:
        step = int(episode.get("step", -1))
        cfg = self.config
        if metrics.get("body.R", 1.0) < cfg.body_R_warn and self._ready("body.R", step):
            if self.learner_hooks:
                self.learner_hooks.reinforce_value_weight(episode, factor=1.02)
            self._emit_action(
                "body.R",
                "lora_reload_body",
                episode,
                {"value": metrics.get("body.R")},
            )
        if metrics.get("body.rho", 0.0) > cfg.body_rho_warn and self._ready("body.rho", step):
            self._emit_action(
                "body.rho",
                "map_elites_shift",
                episode,
                {"value": metrics.get("body.rho")},
            )

        love = metrics.get("affect.love")
        if love is not None:
            if love < cfg.love_low and self._ready("love.low", step):
                if self.learner_hooks:
                    self.learner_hooks.apply_love_softening(episode, delta=0.05)
                self._emit_action(
                    "affect.love",
                    "tone_soften",
                    episode,
                    {"value": love},
                )
                self.fastpath.register_event(episode, False)
            elif love > cfg.love_high and self._ready("love.high", step):
                if self.learner_hooks:
                    self.learner_hooks.apply_love_cooldown(episode, delta=0.04)
                profile = self.fastpath.pick_profile()
                override_used = False
                if self.fastpath.eligible and self.learner_hooks:
                    override_used = self.learner_hooks.apply_fastpath_style_override(
                        episode,
                        profile=profile,
                        magnitude=0.1,
                    )
                self.fastpath.register_event(episode, override_used)
                self._emit_action(
                    "affect.love",
                    "style_override" if override_used else "tone_cooldown",
                    episode,
                    {"value": love, "profile": profile},
                )
            else:
                self.fastpath.register_event(episode, False)
        else:
            self.fastpath.register_event(episode, False)

        trust = metrics.get("value.intent_trust")
        if trust is not None and trust < cfg.trust_low and self._ready("trust.low", step):
            if self.router is not None and hasattr(self.router, "downshift"):
                try:
                    self.router.downshift()
                except Exception:
                    pass
            if self.learner_hooks:
                self.learner_hooks.reinforce_value_weight(episode, factor=1.1)
            self._emit_action(
                "value.intent_trust",
                "temp_down",
                episode,
                {"value": trust},
            )

        safety = metrics.get("ethics.safety_violation")
        if safety and safety > cfg.safety_floor and self._ready("safety", step):
            self._emit_action(
                "ethics.safety_violation",
                "halt_learning",
                episode,
                {"value": safety},
            )

    def _ready(self, tag: str, step: int) -> bool:
        last = self._last_action_step.get(tag)
        if last is None or (step - last) >= self.config.cooldown_steps:
            self._last_action_step[tag] = step
            return True
        return False


__all__ = ["MCPConfig", "MCPController", "FastpathReleaseTracker"]




