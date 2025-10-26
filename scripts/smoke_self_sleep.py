# -*- coding: utf-8 -*-
"""Minimal smoke script for self-narrative and nightly consolidation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from emot_terrain_lab.mind.self_model import SelfModel
from emot_terrain_lab.sleep.cycle import nightly_consolidate


def main() -> None:
    sm = SelfModel()
    sm.note({"id": 0, "intent": "ask"})
    sm.note({"id": 1, "intent": "ask"})
    print("coherence:", sm.coherence())

    summary = nightly_consolidate(
        mem=None,
        self_model=sm,
        value_weights={
            "extrinsic": 0.40,
            "novelty": 0.08,
            "social": 0.18,
            "coherence": 0.12,
            "homeostasis": 0.15,
            "qualia_fit": 0.07,
            "norm_penalty": 0.55,
        },
    )
    print("nightly value weights:", summary["value_weights"])
    print("nightly coherence:", summary["narrative_coherence"])


if __name__ == "__main__":
    main()
