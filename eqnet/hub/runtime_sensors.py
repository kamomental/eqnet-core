"""Helpers for wiring streaming sensor frames into runtime loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import json
import time

from eqnet.hub.streaming_sensor import StreamingSensorState


@dataclass
class RuntimeSensors:
    """Minimal adapter that turns raw frames into ``StreamingSensorState``.

    ``tick`` can be called from any loop/driver (HubRuntime, tests, or the
    mock daemon). The helper keeps the latest snapshot accessible via the
    ``snapshot`` property so downstream code can pull metrics without caring
    about how the frame was sourced.
    """

    log_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot: Optional[StreamingSensorState] = None

    @property
    def snapshot(self) -> Optional[StreamingSensorState]:
        return self._snapshot

    def tick(self, raw_frame: Mapping[str, Any] | None) -> Optional[StreamingSensorState]:
        if not raw_frame:
            return self._snapshot
        snapshot = StreamingSensorState.from_raw(dict(raw_frame))
        self._snapshot = snapshot
        self._maybe_log(raw_frame)
        return snapshot

    def _maybe_log(self, raw_frame: Mapping[str, Any]) -> None:
        if self.log_path is None:
            return
        payload = {
            "ts": time.time(),
            "raw_frame": raw_frame,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def replay_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
        """Yield raw_frame dictionaries from a JSONL capture."""

        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = record.get("raw_frame") if isinstance(record, dict) else None
                if isinstance(raw, dict):
                    yield raw

