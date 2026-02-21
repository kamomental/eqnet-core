"""Multi-tenant manager for EQNetHub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from .api import EQNetConfig, EQNetHub
from eqnet.persona.loader import PersonaConfig, load_persona_from_dir

LOGGER = logging.getLogger(__name__)


class EQNetHubManager:
    """Create per-user/per-persona EQNetHub instances on demand."""

    def __init__(
        self,
        base_dir: Path,
        embed_text_fn: Callable[[str], any],
        *,
        persona_dir: Optional[Path] = None,
        runtime_config_path: Path | str = Path("config/runtime.yaml"),
    ) -> None:
        self.base_dir = base_dir
        self.persona_dir = persona_dir or Path("personas")
        self.runtime_config_path = Path(runtime_config_path)
        self.embed_text_fn = embed_text_fn
        self._hubs: Dict[str, EQNetHub] = {}
        self._runtime_policy = self._load_runtime_policy()

    def _load_runtime_policy(self) -> Dict[str, Any]:
        path = self.runtime_config_path
        if not path.exists():
            return {}
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            LOGGER.warning("failed to load runtime policy from %s", path, exc_info=True)
        return {}

    def for_user(self, user_id: str) -> EQNetHub:
        if user_id not in self._hubs:
            user_root = self.base_dir / user_id
            memory_thermo_policy = self._runtime_policy.get("memory_thermo_policy")
            if not isinstance(memory_thermo_policy, dict):
                memory_thermo_policy = None
            cfg = EQNetConfig(
                telemetry_dir=user_root / "telemetry",
                reports_dir=user_root / "reports",
                state_dir=user_root / "state",
                memory_thermo_policy=memory_thermo_policy,
                runtime_policy=dict(self._runtime_policy),
            )
            persona = load_persona_from_dir(self.persona_dir, user_id)
            self._hubs[user_id] = EQNetHub(
                config=cfg,
                embed_text_fn=self.embed_text_fn,
                persona=persona,
            )
        return self._hubs[user_id]

