from datetime import datetime

import numpy as np

from eqnet.qualia_model import FutureReplayConfig, ReplayMode, simulate_future
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
