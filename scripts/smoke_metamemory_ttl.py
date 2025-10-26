# -*- coding: utf-8 -*-
"""Quick smoke checks for metamemory, memory TTL, and mode hysteresis."""

from __future__ import annotations

from typing import Any, Dict, List

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emot_terrain_lab.hub.mode_hysteresis import ModeHysteresis
from emot_terrain_lab.memory.ttl import MemoryTTLManager
from emot_terrain_lab.mind.metamemory import MetaMemory


class _MockTimeKeeper:
    def __init__(self, tau: float = 12.0) -> None:
        self._tau = tau

    def tau_now(self) -> float:
        return self._tau


class _MockMemoryIndex:
    def topk(self, cue: Dict[str, Any], *, k: int = 16, limit: int = 512) -> List[Dict[str, Any]]:
        text = str(cue.get("text", "")).lower()
        sims = [0.8 if word in text else 0.2 for word in ("sensor", "monitor", "kinesis")]
        return [
            {"sim": sims[0], "partial": {"initial": "K", "year": "2021"}},
            {"sim": sims[1], "partial": {"category": "safety"}},
            {"sim": sims[2], "partial": {}},
        ][:k]


def run_ttl_smoke() -> None:
    tk = _MockTimeKeeper()
    manager = MemoryTTLManager(
        {
            "ttl_tau": {"default": 24.0, "low_value": 6.0},
            "halflife_tau": 24.0,
            "ephemeral_tags": ["private"],
            "ephemeral_tau": 4.0,
        },
        tk,
    )
    events = [
        {"tau": 2.0, "value": {"total": 0.5}, "tags": [], "weight": 1.0},
        {"tau": 10.0, "value": {"total": 0.05}, "tags": [], "weight": 1.0},
        {"tau": 1.0, "value": {"total": 0.1}, "tags": ["private"], "weight": 1.0},
    ]
    kept, stats = manager.gc(events)
    print("TTL kept/dropped:", stats, "kept_len", len(kept))


def run_metamemory_smoke() -> None:
    mm = MetaMemory(
        {
            "theta_fok": 0.6,
            "theta_rec": 0.3,
            "cooldown_tau": 0.5,
            "tot_mode": {"horizon_max": 2, "read_only": True},
        },
        _MockMemoryIndex(),
    )
    out = mm.estimate({"text": "Kinesis sensor monitor"}, tau_now=12.0)
    print("Metamemory:", out)


def run_hysteresis_smoke() -> None:
    hysteresis = ModeHysteresis({"enter": 0.4, "exit": 0.25})
    first = hysteresis.decide("supportive", 0.45)
    second = hysteresis.decide(first.new_mode, 0.2)
    print("Hysteresis decisions:", first, second)


if __name__ == "__main__":
    run_ttl_smoke()
    run_metamemory_smoke()
    run_hysteresis_smoke()
