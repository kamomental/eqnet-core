from datetime import datetime

import numpy as np

from eqnet.qualia_model import (
    FutureReplayConfig,
    ReplayMode,
    compute_future_risk,
    compute_future_hopefulness,
    simulate_future,
)
from eqnet.runtime.state import QualiaState


def _make_history(fog: float = 0.5) -> list[QualiaState]:
    ts = datetime.utcnow()
    return [
        QualiaState(
            timestamp=ts,
            qualia_vec=np.zeros(4, dtype=float),
            membrane_state={"fog_level": fog},
        )
    ]


def test_simulate_future_modes_have_same_length() -> None:
    history = _make_history()
    cfg = FutureReplayConfig(steps=3)
    traj_pred = simulate_future(history, ReplayMode.PREDICTIVE, cfg)
    traj_img = simulate_future(
        history,
        ReplayMode.IMAGERY,
        cfg,
        intention_vec=np.ones(4, dtype=float) * 0.1,
    )

    assert len(traj_pred) == cfg.steps + 1
    assert len(traj_img) == cfg.steps + 1
    assert traj_pred[0].shape == history[-1].qualia_vec.shape
    assert traj_img[0].shape == history[-1].qualia_vec.shape


def test_fog_scales_noise_variance() -> None:
    cfg = FutureReplayConfig(steps=2)
    history_low = _make_history(fog=0.0)
    history_high = _make_history(fog=1.0)

    norms_low, norms_high = [], []
    for _ in range(30):
        low = simulate_future(history_low, ReplayMode.PREDICTIVE, cfg)
        high = simulate_future(history_high, ReplayMode.PREDICTIVE, cfg)
        norms_low.append(np.linalg.norm(low[-1] - low[0]))
        norms_high.append(np.linalg.norm(high[-1] - high[0]))

    assert np.mean(norms_high) > np.mean(norms_low)

def test_compute_future_risk_thresholds() -> None:
    cfg = FutureReplayConfig(steps=2, noise_scale=0.0, window=1)
    history: list[QualiaState] = []
    ts = datetime.utcnow()
    for offset in (0.0, 0.05, 0.1):
        history.append(
            QualiaState(
                timestamp=ts,
                qualia_vec=np.array([0.8 - offset, 0.1 + offset], dtype=float),
                membrane_state={"fog_level": 0.3},
            )
        )
    risk = compute_future_risk(
        history,
        cfg,
        stress_index=1,
        stress_threshold=0.12,
        body_index=0,
        body_threshold=0.75,
    )
    assert 0.0 <= risk <= 1.0
    assert risk > 0.0

def test_compute_future_hopefulness_positive() -> None:
    cfg = FutureReplayConfig(steps=2, noise_scale=0.0, window=1)
    ts = datetime.utcnow()
    history = [
        QualiaState(
            timestamp=ts,
            qualia_vec=np.array([0.1, 0.2], dtype=float),
            membrane_state={"fog_level": 0.2},
        ),
        QualiaState(
            timestamp=ts,
            qualia_vec=np.array([0.12, 0.22], dtype=float),
            membrane_state={"fog_level": 0.2},
        ),
    ]
    hope = compute_future_hopefulness(
        history,
        cfg,
        intention_vec=np.array([0.05, 0.08], dtype=float),
    )
    assert 0.0 <= hope <= 1.0
    assert hope > 0.0

