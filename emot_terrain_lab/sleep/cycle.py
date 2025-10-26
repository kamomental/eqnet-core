# -*- coding: utf-8 -*-
"""Nightly consolidation routine for replay traces and self state."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


def nightly_consolidate(
    *,
    replay_log_path: str = "logs/replay_firings.jsonl",
    mem: Optional[object] = None,
    self_model: Optional[object] = None,
    value_weights: Optional[Dict[str, float]] = None,
    half_life_tau_hours: float = 6.0,
    ewc_lambda: float = 0.0,
    lora_lr: float = 0.0,
    out_path: str = "logs/nightly_summary.json",
) -> Dict[str, object]:
    """
    Aggregate replay firings and snapshot narrative coherence for inspection.
    """

    now = time.time()
    events: List[Dict[str, object]] = []
    total_utility = 0.0
    oldest_ts = float("inf")
    newest_ts = 0.0
    if os.path.exists(replay_log_path):
        with open(replay_log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                events.append(ev)
                ts = float(ev.get("ts", now))
                oldest_ts = min(oldest_ts, ts)
                newest_ts = max(newest_ts, ts)
                total_utility += float(ev.get("U_top", 0.0))

    count = len(events)
    avg_utility = total_utility / count if count else 0.0

    # Soft forgetting hook: scale placeholder weight decay by tau half-life.
    decay = math.exp(-math.log(2.0) * (1.0 / max(half_life_tau_hours, 1e-3)))
    decay_applied = False
    if mem is not None and hasattr(mem, "decay"):
        try:
            mem.decay(decay)
            decay_applied = True
        except Exception:
            pass

    coherence = None
    if self_model is not None and hasattr(self_model, "coherence"):
        try:
            coherence = float(self_model.coherence())
        except Exception:
            coherence = None

    adjusted_weights = dict(value_weights or {})
    if coherence is not None and coherence < 0.5:
        adjusted = adjusted_weights.get("coherence", 0.12) + 0.02
        adjusted_weights["coherence"] = min(0.2, adjusted)

    summary = {
        "ts": now,
        "events": count,
        "U_avg": avg_utility,
        "narrative_coherence": coherence,
        "oldest_ts": oldest_ts if oldest_ts != float("inf") else None,
        "newest_ts": newest_ts if newest_ts else None,
        "value_weights": adjusted_weights,
        "notes": {"ewc_lambda": ewc_lambda, "lora_lr": lora_lr},
        "decay_factor": decay,
        "decay_applied": decay_applied,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


__all__ = ["nightly_consolidate"]
