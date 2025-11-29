from datetime import datetime

import numpy as np

from eqnet.runtime.state import QualiaState


def test_qualia_state_coerces_to_1d() -> None:
    ts = datetime.utcnow()
    state = QualiaState(timestamp=ts, qualia_vec=[[1.0, 2.0, 3.0]])
    assert isinstance(state.qualia_vec, np.ndarray)
    assert state.qualia_vec.ndim == 1
    assert state.qualia_vec.shape == (3,)
