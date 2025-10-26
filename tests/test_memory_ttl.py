# -*- coding: utf-8 -*-

from types import SimpleNamespace

from emot_terrain_lab.memory.ttl import MemoryTTLManager


class DummyTK:
    def __init__(self, tau: float) -> None:
        self._tau = tau

    def tau_now(self) -> float:
        return self._tau


def test_ttl_scale_and_override() -> None:
    cfg = {"ttl_tau": {"default": 24.0, "low_value": 6.0}, "halflife_tau": 24.0}
    manager = MemoryTTLManager(cfg, DummyTK(tau=10.0))
    event = {"value": {"total": 1.0}, "tau": 0.0}
    base_ttl = manager._ttl_for(event)
    assert base_ttl == 24.0

    event["meta"] = {"ttl_scale": 0.5}
    assert manager._ttl_for(event) == 12.0

    event["meta"]["ttl_override_tau"] = 5.0
    assert manager._ttl_for(event) == 5.0
