from dataclasses import dataclass

from inner_os.forgetting_core import ForgettingCore


@dataclass
class _Advice:
    lstm: dict
    ssm: dict
    intero: dict
    replay: dict
    persona: dict


class _StubController:
    def advise(self, **kwargs):
        return _Advice(
            lstm={"forget_bias_delta": 0.22},
            ssm={},
            intero={},
            replay={"horizon": 2},
            persona={"halflife_tau": 18.0},
        )


def test_forgetting_core_compacts_controller_advice() -> None:
    core = ForgettingCore(_StubController())
    snapshot = core.snapshot(
        stress=0.42,
        recovery_need=0.33,
        terrain_transition_roughness=0.28,
        transition_intensity=0.31,
        recent_strain=0.4,
    )
    data = snapshot.to_dict()
    assert data["forgetting_pressure"] > 0.0
    assert data["replay_horizon"] == 2
    assert data["persona_halflife_tau"] == 18.0
