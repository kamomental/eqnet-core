"""Multi-tenant manager for EQNetHub."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from .api import EQNetConfig, EQNetHub
from eqnet.persona.loader import PersonaConfig, load_persona_from_dir


class EQNetHubManager:
    """Create per-user/per-persona EQNetHub instances on demand."""

    def __init__(
        self,
        base_dir: Path,
        embed_text_fn: Callable[[str], any],
        *,
        persona_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir
        self.persona_dir = persona_dir or Path("personas")
        self.embed_text_fn = embed_text_fn
        self._hubs: Dict[str, EQNetHub] = {}

    def for_user(self, user_id: str) -> EQNetHub:
        if user_id not in self._hubs:
            user_root = self.base_dir / user_id
            cfg = EQNetConfig(
                telemetry_dir=user_root / "telemetry",
                reports_dir=user_root / "reports",
                state_dir=user_root / "state",
            )
            persona = load_persona_from_dir(self.persona_dir, user_id)
            self._hubs[user_id] = EQNetHub(
                config=cfg,
                embed_text_fn=self.embed_text_fn,
                persona=persona,
            )
        return self._hubs[user_id]

