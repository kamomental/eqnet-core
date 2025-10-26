"""Culture feedback logger writing triples."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import json


@dataclass
class CultureConfig:
    output: Path = Path("logs/culture/triples.jsonl")


@dataclass
class CultureLogger:
    config: CultureConfig = field(default_factory=CultureConfig)
    buffer: List[Dict[str, object]] = field(default_factory=list)

    def log(self, speaker: str, peer: str, delta_aff: float, tags: List[str] | None = None) -> None:
        record = {
            "s": f"peer:{speaker}",
            "p": "evoked",
            "o": f"aff:{delta_aff:+.2f}",
            "ctx": tags or [],
        }
        self.buffer.append(record)

    def flush(self) -> None:
        if not self.buffer:
            return
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output.open("a", encoding="utf-8") as fh:
            for rec in self.buffer:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.buffer.clear()

