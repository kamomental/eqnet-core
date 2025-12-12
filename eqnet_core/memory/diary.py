from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from eqnet_core.models.conscious import ConsciousEpisode


class DiaryWriter:
    """Simple JSONL diary that mirrors conscious episodes."""

    def __init__(self, path: str | Path = "logs/conscious_diary.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_conscious_episode(self, episode: ConsciousEpisode) -> None:
        entry = self._serialize_episode(episode)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _serialize_episode(self, episode: ConsciousEpisode) -> Dict[str, object]:
        talk_mode = episode.self_state.current_mode.value
        return {
            "ts": episode.timestamp.isoformat(),
            "id": episode.id,
            "route": episode.route.value,
            "talk_mode": talk_mode,
            "summary": episode.world_state.summary_text,
            "narrative": episode.narrative,
            "salient_entities": episode.world_state.salient_entities,
            "context_tags": episode.world_state.context_tags,
            "prediction_error": episode.world_state.prediction_error,
            "hazard_score": episode.world_state.hazard_score,
            "qualia": episode.qualia.to_dict(),
            "value_gradient": (episode.value_gradient or episode.qualia.value_gradient).to_dict(),
            "dominant_self_layer": episode.dominant_self_layer.value if episode.dominant_self_layer else None,
            "implementation": episode.implementation.to_dict() if episode.implementation else None,
            "self_force": episode.self_force.to_dict() if episode.self_force else None,
            "raw_self_force": episode.raw_self_force.to_dict() if episode.raw_self_force else None,
            "boundary_signal": episode.boundary_signal.to_dict() if episode.boundary_signal else None,
            "reset_event": episode.reset_event.to_dict() if episode.reset_event else None,
        }
