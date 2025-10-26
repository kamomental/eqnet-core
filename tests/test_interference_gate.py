# -*- coding: utf-8 -*-

import time

from devlife.mind.replay_memory import ReplayTrace
from emot_terrain_lab.mind.interference_gate import InterferenceGate


class DummyTK:
    def __init__(self) -> None:
        self._tau = 0.0

    def tau_now(self) -> float:
        return self._tau

    def advance(self, delta: float) -> None:
        self._tau += delta


def _trace(action: str, domain: str = "chat") -> ReplayTrace:
    return ReplayTrace(
        trace_id="t",
        episode_id="e",
        timestamp=time.time(),
        source="internal",
        horizon=1,
        uncertainty=0.0,
        mood={},
        value={},
        controls={},
        imagined={"best_action": action},
        meta={"domain": domain},
        tags=[],
        weight=1.0,
        tau=0.0,
    )


def test_interference_gate_masks_similar_actions() -> None:
    tk = DummyTK()
    gate = InterferenceGate(
        {
            "similarity_threshold": 0.5,
            "mask_tau": 2.0,
            "ttl_override_tau": 1.0,
            "hold_tau": 2.0,
        },
        timekeeper=tk,
    )
    first = gate.evaluate(_trace("offer support gently"))
    assert first["action"] == "pass"
    second = gate.evaluate(_trace("offer gentle support now"))
    assert second["action"] == "mask"
    assert second["ttl_override_tau"] == 1.0
    tk.advance(3.0)
    third = gate.evaluate(_trace("offer gentle support now"))
    assert third["action"] == "pass"
