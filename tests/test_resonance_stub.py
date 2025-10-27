import numpy as np

from devlife.runtime.loop import DevelopmentLoop, StageConfig, SleepConfig


class _Dummy:
    def __getattr__(self, item):
        raise AttributeError(item)


def _make_loop() -> DevelopmentLoop:
    return DevelopmentLoop(
        _Dummy(),
        _Dummy(),
        _Dummy(),
        _Dummy(),
        _Dummy(),
        stages=[StageConfig(name="test", duration_steps=0)],
        sleep=SleepConfig(interval_steps=1),
    )


def _fake_sensors(value: float) -> dict:
    channels = np.full((1, 4, 4), value, dtype=float)
    return {"channels": channels, "kappa": 0.1, "stats": {"mean": value, "var": 0.0, "edge": 0.0}}


def test_resonance_bias_increases_rho() -> None:
    loop = _make_loop()
    sensors = _fake_sensors(0.2)
    baseline = loop._compute_field_metrics(sensors, sensors, {"energy": 0.0})
    loop._prev_I_value = 0.4
    loop.ingest_resonance_from_peer(0.9, k_res=0.1)
    adjusted = loop._compute_field_metrics(sensors, sensors, {"energy": 0.0})
    assert adjusted["rho"] >= baseline["rho"]
