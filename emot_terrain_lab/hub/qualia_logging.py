"""Telemetry helpers for recording QualiaState snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from eqnet.runtime.state import QualiaState


def _qualia_log_path(base_dir: Path, ts: datetime) -> Path:
    date_str = ts.strftime("%Y%m%d")
    return base_dir / f"qualia-{date_str}.jsonl"


def append_qualia_telemetry(
    base_dir: Path,
    qstate: QualiaState,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Append one QualiaState entry to telemetry/qualia-*.jsonl.

    Notes
    -----
    - このログは Ethical & Legal Guardrails に従い、セルフケア／研究／
      フィクション用途に限定して扱うこと。
    """

    base_dir.mkdir(parents=True, exist_ok=True)
    path = _qualia_log_path(base_dir, qstate.timestamp)

    record: Dict[str, Any] = {
        "timestamp": qstate.timestamp.isoformat(),
        "qualia_vec": qstate.qualia_vec.tolist(),
        "membrane_state": qstate.membrane_state,
        "flux": qstate.flux,
        "narrative_ref": qstate.narrative_ref,
    }
    if extra:
        record.update(extra)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

