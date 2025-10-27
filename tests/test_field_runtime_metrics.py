import json

import numpy as np
import pytest

from devlife.runtime.loop import DevelopmentLoop, StageConfig, SleepConfig


class _Dummy:
    """Minimal stub so DevelopmentLoop wiring stays lightweight in tests."""

    def __getattr__(self, item):  # pragma: no cover - only hit when a method is unexpectedly used
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


def _fake_sensors(value: float = 0.0) -> dict:
    channels = np.full((1, 4, 4), value, dtype=float)
    return {"channels": channels, "kappa": 0.1, "stats": {"mean": value, "var": 0.0, "edge": 0.0}}


def _ignition_value(entropy_norm: float, rho_norm: float, delta_r: float = 0.0) -> float:
    """Helper that returns a single ignition value for the given inputs."""
    loop = _make_loop()
    base_reward = -0.5
    loop._prev_reward = base_reward
    stats = {"homeo_error": base_reward + delta_r}
    _, I_value, _, _ = loop._ignition_index(
        stats,
        entropy_z=0.0,
        field_metrics={"entropy_norm": entropy_norm, "rho_norm": rho_norm},
    )
    return float(I_value)


def test_field_metrics_ingest_overrides_proxy() -> None:
    loop = _make_loop()
    loop.ingest_field_metrics({"entropy_norm": 0.2, "enthalpy_norm": 0.9, "rho_norm": 0.7})
    sensors = _fake_sensors()
    metrics = loop._compute_field_metrics(sensors, sensors, {"energy": 0.1})
    assert pytest.approx(metrics["S"], rel=1e-6) == 0.2
    assert pytest.approx(metrics["H"], rel=1e-6) == 0.9
    assert metrics["field_source"] == "push"
    assert "valence" in metrics
    assert -1.0 <= metrics["valence"] <= 1.0


def test_field_metrics_log_loader(tmp_path) -> None:
    payload = [{"entropy_norm": 0.3, "enthalpy_norm": 0.4, "rho_norm": 0.8}]
    log_path = tmp_path / "field_metrics.json"
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    loop = _make_loop()
    loop.load_field_metrics_log(log_path)
    sensors = _fake_sensors()
    metrics = loop._compute_field_metrics(sensors, sensors, {"energy": 0.0})
    assert pytest.approx(metrics["S"]) == 0.3
    assert pytest.approx(metrics["rho"]) == 0.8
    assert metrics["field_source"] == "log"
    assert "valence" in metrics


def test_ignition_respects_entropy_and_reward() -> None:
    loop_low = _make_loop()
    loop_low._prev_reward = -0.4
    _, I_low, _, _ = loop_low._ignition_index(
        {"homeo_error": -0.1},
        entropy_z=0.0,
        field_metrics={"entropy_norm": 0.1},
    )

    loop_high = _make_loop()
    loop_high._prev_reward = -0.4
    _, I_high, _, _ = loop_high._ignition_index(
        {"homeo_error": -0.1},
        entropy_z=0.0,
        field_metrics={"entropy_norm": 0.9},
    )

    assert 0.0 <= I_low <= 1.0
    assert 0.0 <= I_high <= 1.0
    assert I_low > I_high

    loop_delta_small = _make_loop()
    loop_delta_small._prev_reward = -0.6
    _, I_small, _, _ = loop_delta_small._ignition_index(
        {"homeo_error": -0.55},
        entropy_z=0.0,
        field_metrics={"entropy_norm": 0.5},
    )

    loop_delta_large = _make_loop()
    loop_delta_large._prev_reward = -0.6
    _, I_large, _, _ = loop_delta_large._ignition_index(
        {"homeo_error": -0.1},
        entropy_z=0.0,
        field_metrics={"entropy_norm": 0.5},
    )

    assert I_large > I_small


def test_ignition_monotonic_and_boundary_sequences() -> None:
    s_values = np.linspace(0.1, 0.9, 7)
    I_s = [_ignition_value(float(s), 0.5, delta_r=0.0) for s in s_values]
    assert all(left >= right - 1e-6 for left, right in zip(I_s, I_s[1:]))

    rho_values = np.linspace(0.1, 0.9, 7)
    I_rho = [_ignition_value(0.3, float(rho), delta_r=0.0) for rho in rho_values]
    assert all(left <= right + 1e-6 for left, right in zip(I_rho, I_rho[1:]))

    boundary_cases = [
        {"entropy_norm": float("nan"), "rho_norm": 0.0},
        {"entropy_norm": float("inf"), "rho_norm": 1.0},
        {"entropy_norm": -float("inf"), "rho_norm": float("nan")},
    ]
    for metrics in boundary_cases:
        loop = _make_loop()
        loop._prev_reward = -0.4
        _, I_value, _, _ = loop._ignition_index({"homeo_error": -0.4}, entropy_z=0.0, field_metrics=metrics)
        assert 0.0 <= I_value <= 1.0
