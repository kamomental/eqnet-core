import numpy as np

from inner_os.observation_model import TensorObservationModel
from inner_os.qualia_projector import BasicQualiaProjector
from inner_os.qualia_projector import _normalize
from inner_os.self_estimator import ResidualLinearSelfEstimator, evaluate_estimator_health


def _build_estimate_and_health(*, ext, body_obs, body_state, noise=0.05):
    model = TensorObservationModel(
        latent_dim=4,
        ext_size=2,
        body_size=1,
        boundary_size=1,
        action_size=0,
        default_noise=noise,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=4,
        A_x=np.eye(4, dtype=np.float32),
    )
    obs = model.encode(
        {
            "ext": ext,
            "body_obs": body_obs,
            "boundary": [0.1],
        },
        body=body_state,
        prev_action=[],
        dt=0.1,
    )
    estimate = estimator.step(obs, body=body_state, prev_action=[])
    health = evaluate_estimator_health(estimate, obs)
    return obs, estimate, health


def test_qualia_projector_returns_component_vectors_and_respects_shape_contract() -> None:
    obs, estimate, health = _build_estimate_and_health(ext=[0.8, 0.4], body_obs=[0.3], body_state=[0.3])
    projector = BasicQualiaProjector()

    state = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=np.zeros(4, dtype=np.float32),
        prev_qualia=np.zeros(4, dtype=np.float32),
        prev_habituation=np.zeros(4, dtype=np.float32),
        protection_grad_x=np.zeros(4, dtype=np.float32),
        dt=0.1,
    )

    assert state.qualia.shape == (4,)
    assert state.gate.shape == (4,)
    assert state.precision.shape == (4,)
    assert state.observability.shape == (4,)
    assert state.body_coupling.shape == (4,)
    assert state.trust_applied == health.trust
    assert "value_grad" in state.normalization_stats
    assert "range_trust" in state.normalization_stats["value_grad"]


def test_qualia_projector_uses_predict_only_health_to_hold_close_to_previous_qualia() -> None:
    model = TensorObservationModel(
        latent_dim=3,
        ext_size=1,
        body_size=1,
        boundary_size=1,
        action_size=0,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=3,
        A_x=np.eye(3, dtype=np.float32),
    )
    obs = model.encode(
        {"ext": [None], "body_obs": [None], "boundary": [None]},
        body=[0.0],
        prev_action=[],
        dt=0.1,
    )
    estimate = estimator.step(obs, body=[0.0], prev_action=[])
    health = evaluate_estimator_health(estimate, obs)
    projector = BasicQualiaProjector()
    prev_q = np.array([0.7, 0.1, 0.2], dtype=np.float32)

    state = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=np.zeros(3, dtype=np.float32),
        prev_qualia=prev_q,
        prev_habituation=np.zeros(3, dtype=np.float32),
        protection_grad_x=np.zeros(3, dtype=np.float32),
        dt=0.1,
    )

    assert state.degraded is True
    assert "predict_only" in (state.reason or "")
    assert np.linalg.norm(state.qualia - prev_q) < np.linalg.norm(prev_q)


def test_qualia_projector_body_coupling_uses_only_body_rows() -> None:
    obs, estimate, health = _build_estimate_and_health(ext=[0.5, 0.2], body_obs=[0.9], body_state=[0.9])
    projector = BasicQualiaProjector()

    state = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=np.zeros(4, dtype=np.float32),
        prev_qualia=np.zeros(4, dtype=np.float32),
        prev_habituation=np.zeros(4, dtype=np.float32),
        protection_grad_x=np.zeros(4, dtype=np.float32),
        dt=0.1,
    )

    assert np.any(state.body_coupling > 0.0)
    assert state.body_coupling[0] == 0.0
    assert state.body_coupling[1] == 0.0


def test_qualia_projector_value_gradient_changes_gate_and_qualia() -> None:
    obs, estimate, health = _build_estimate_and_health(ext=[0.4, 0.3], body_obs=[0.2], body_state=[0.2])
    projector = BasicQualiaProjector()
    zeros = np.zeros(4, dtype=np.float32)
    boosted = np.array([0.0, 0.0, 3.0, 0.0], dtype=np.float32)

    base = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=zeros,
        prev_qualia=zeros,
        prev_habituation=zeros,
        protection_grad_x=zeros,
        dt=0.1,
    )
    shifted = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=zeros,
        prev_qualia=zeros,
        prev_habituation=zeros,
        protection_grad_x=boosted,
        dt=0.1,
    )

    assert not np.allclose(base.gate, shifted.gate)
    assert shifted.gate[2] > base.gate[2]
    assert shifted.value_grad[2] > base.value_grad[2]


def test_qualia_projector_normalization_keeps_small_axes_visible_under_outlier() -> None:
    values = np.array([0.2, 0.21, 0.22, 8.0], dtype=np.float32)
    result = _normalize(values, global_range=1.0)
    normalized = result.values

    assert float(normalized[1]) > 0.0
    assert float(normalized[2]) > float(normalized[0])
    assert float(normalized[-1]) <= 1.0
    assert result.stats.range_trust < 0.5


def test_qualia_projector_habituation_suppresses_gate() -> None:
    obs, estimate, health = _build_estimate_and_health(ext=[0.6, 0.1], body_obs=[0.2], body_state=[0.2])
    projector = BasicQualiaProjector()
    zeros = np.zeros(4, dtype=np.float32)
    prev_q = np.array([1.2, 0.0, 0.0, 0.0], dtype=np.float32)
    prev_h = np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32)

    fresh = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=zeros,
        prev_qualia=zeros,
        prev_habituation=zeros,
        protection_grad_x=zeros,
        dt=0.1,
    )
    habituated = projector.project(
        obs=obs,
        est=estimate,
        health=health,
        memory=zeros,
        prev_qualia=prev_q,
        prev_habituation=prev_h,
        protection_grad_x=zeros,
        dt=0.1,
    )

    assert habituated.habituation[0] > fresh.habituation[0]
    assert habituated.gate[0] <= fresh.gate[0]


def test_qualia_projector_rejects_state_length_mismatch() -> None:
    obs, estimate, health = _build_estimate_and_health(ext=[0.2, 0.2], body_obs=[0.2], body_state=[0.2])
    projector = BasicQualiaProjector()

    try:
        projector.project(
            obs=obs,
            est=estimate,
            health=health,
            memory=np.zeros(4, dtype=np.float32),
            prev_qualia=np.zeros(3, dtype=np.float32),
            prev_habituation=np.zeros(4, dtype=np.float32),
            protection_grad_x=np.zeros(4, dtype=np.float32),
            dt=0.1,
        )
    except ValueError as exc:
        assert "prev_qualia" in str(exc)
    else:
        raise AssertionError("expected ValueError for state length mismatch")
