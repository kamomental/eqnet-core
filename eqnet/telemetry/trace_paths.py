from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TracePathConfig:
    base_dir: Path
    source_loop: str
    run_id: Optional[str] = None


def trace_output_path(cfg: TracePathConfig, *, timestamp_ms: Optional[int]) -> Path:
    ts = int(timestamp_ms) if timestamp_ms is not None else int(time.time() * 1000)
    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    day = dt.strftime("%Y-%m-%d")

    pid = os.getpid()
    run_id = cfg.run_id or os.getenv("EQNET_RUN_ID")
    if run_id:
        filename = f"{cfg.source_loop}-{run_id}-{pid}.jsonl"
    else:
        filename = f"{cfg.source_loop}-{pid}.jsonl"

    return cfg.base_dir / day / filename


__all__ = ["TracePathConfig", "trace_output_path"]
