from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Set, TextIO

from common import Event


@dataclass(frozen=True)
class CollectorConfig:
    allowed_event_types: Set[str]
    strict_integrity: bool


class Collector:
    """Extract allowed events from a JSONL stream."""

    def __init__(self, config: CollectorConfig) -> None:
        self._cfg = config

    def parse_line(self, line: str) -> Optional[Event]:
        line = line.strip()
        if not line:
            return None
        try:
            ev = Event.from_json_line(line)
        except Exception:
            return None

        if ev.event_type not in self._cfg.allowed_event_types:
            return None

        if self._cfg.strict_integrity:
            integrity = (ev.trace or {}).get("integrity") or {}
            if "payload_hash" not in integrity or "prev_hash" not in integrity:
                return None

        return ev

    def iter_events(self, fp: TextIO) -> Iterator[Event]:
        for line in fp:
            ev = self.parse_line(line)
            if ev is not None:
                yield ev
