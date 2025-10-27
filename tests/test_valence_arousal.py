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


def test_valence_trends_with_entropy_and_rho() -> None:
    loop = _make_loop()
    high_rho = loop._compute_valence(0.3, 0.9)
    low_rho = loop._compute_valence(0.3, 0.2)
    low_entropy = loop._compute_valence(0.2, 0.5)
    high_entropy = loop._compute_valence(0.8, 0.5)
    assert high_rho > low_rho
    assert low_entropy > high_entropy
    assert -1.0 <= high_rho <= 1.0


def test_arousal_matches_ignition_velocity() -> None:
    loop = _make_loop()
    dt_sec = max(float(loop._runtime_cfg.replay.min_interval_ms) / 1000.0, 1e-3)
    first = loop._compute_arousal(0.4, dt_sec)
    second = loop._compute_arousal(0.7, dt_sec)
    third = loop._compute_arousal(0.6, dt_sec)
    excitation = min(max(0.7 - 0.4, 0.0) / dt_sec, 1.0)
    expected = 0.2 * excitation + 0.8 * 0.7
    assert first == 0.0
    assert np.isclose(second, expected, rtol=1e-6)
    assert third == 0.48
