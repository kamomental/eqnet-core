import json
import math
from pathlib import Path

import numpy as np

from ops import resonance_metrics
from telemetry import plot_resonance


def _write_log(path: Path, rho_values):
    lines = []
    for step, rho in enumerate(rho_values, start=1):
        lines.append(
            json.dumps(
                {
                    "event": "field.metrics",
                    "data": {"stage": "test", "step": step, "rho": float(rho)},
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_resonance_corr_and_lag(tmp_path):
    a_values = np.sin(np.linspace(0, 4 * np.pi, 500))
    lag_samples = 5
    b_values = np.concatenate((np.zeros(lag_samples), a_values[:-lag_samples]))

    log_a = tmp_path / "agent_a.jsonl"
    log_b = tmp_path / "agent_b.jsonl"
    _write_log(log_a, a_values)
    _write_log(log_b, b_values)

    metrics = resonance_metrics.compute_resonance_metrics(
        [("agent_a", log_a), ("agent_b", log_b)],
        resample_ms=20,
        zscore=True,
        detrend=True,
        window="hann",
        return_series=True,
    )
    pair = metrics["pairs"][0]
    assert pair["agents"] == ["agent_a", "agent_b"]
    assert pair["rho_corr"] > 0.85
    assert pair["n_eff"] >= 2.0
    assert pair["rho_cross_corr_peak"] > 0.8
    lag_est = pair.get("rho_cross_corr_lag")
    assert math.isfinite(lag_est)


def test_resonance_cli(tmp_path):
    t = np.linspace(0, 1, 120)
    a_values = 0.5 + 0.2 * np.sin(2 * np.pi * t)
    b_values = 0.4 + 0.25 * np.cos(2 * np.pi * t)
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    _write_log(log_a, a_values)
    _write_log(log_b, b_values)

    out_path = tmp_path / "resonance.json"
    args = [
        "--logs",
        f"A={log_a}",
        f"B={log_b}",
        "--resample-ms",
        "25",
        "--zscore",
        "--detrend",
        "--window",
        "hann",
        "--matrix",
        "--plots-dir",
        str(tmp_path / "plots"),
        "--out",
        str(out_path),
    ]
    resonance_metrics.main(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "resonance.v1"
    assert payload["pairs"][0]["agents"] == ["A", "B"]
    assert out_path.with_suffix(".matrix.json").exists()
    assert (tmp_path / "plots").exists()


def test_resonance_with_options(tmp_path):
    a_values = np.linspace(0, 1, 200)
    b_values = np.sin(np.linspace(0, 4 * np.pi, 200))
    log_a = tmp_path / "opt_a.jsonl"
    log_b = tmp_path / "opt_b.jsonl"
    _write_log(log_a, a_values)
    _write_log(log_b, b_values)

    metrics = resonance_metrics.compute_resonance_metrics(
        [("A", log_a), ("B", log_b)],
        resample_ms=50,
        zscore=True,
        detrend=True,
        window="hann",
        alpha=0.05,
        beta=0.02,
    )
    pair = metrics["pairs"][0]
    assert "objective" in pair
    assert pair["n_eff"] >= 2.0


def test_objective_history_plot(tmp_path):
    history = tmp_path / "resonance_history.jsonl"
    entries = [
        {"ts": 1, "objective": 0.1},
        {"ts": 2, "objective": 0.3},
        {"ts": 3, "objective": 0.2},
    ]
    history.write_text("\n".join(json.dumps(r) for r in entries), encoding="utf-8")
    out_png = tmp_path / "objective.png"
    plot_resonance.plot_objective_history(history, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0

from telemetry import plot_resonance

