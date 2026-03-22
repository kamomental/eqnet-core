import numpy as np

from inner_os.kernel_runtime import KERNEL_UPDATE_ORDER, KernelStepContract
from inner_os.observation_model import TensorObservationModel
from inner_os.self_estimator import ResidualLinearSelfEstimator, evaluate_estimator_health


def test_residual_linear_self_estimator_returns_covariance_innovation_and_jacobian() -> None:
    model = TensorObservationModel(
        latent_dim=4,
        ext_size=2,
        body_size=1,
        boundary_size=1,
        action_size=0,
        default_noise=0.05,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=4,
        A_x=np.eye(4, dtype=np.float32),
    )
    obs = model.encode(
        {
            "ext": [0.8, 0.4],
            "body_obs": [0.2],
            "boundary": [0.1],
        },
        body=[0.2],
        prev_action=[],
        dt=0.1,
    )

    estimate = estimator.step(obs, body=[0.2], prev_action=[])
    post_prediction = model.predict(estimate.x_hat, body=[0.2], prev_action=[])

    assert estimate.cov.shape == (4, 4)
    assert estimate.innovation.shape == obs.y.shape
    assert estimate.innovation_cov.shape == (4, 4)
    assert estimate.H.shape == (4, 4)
    assert np.linalg.norm(obs.y - post_prediction) < np.linalg.norm(obs.y)


def test_estimator_health_flags_overconfident_large_residual() -> None:
    model = TensorObservationModel(
        latent_dim=2,
        ext_size=2,
        body_size=0,
        boundary_size=0,
        action_size=0,
        default_noise=0.001,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=2,
        A_x=np.eye(2, dtype=np.float32),
        initial_cov=np.eye(2, dtype=np.float32) * 0.01,
        Q=np.eye(2, dtype=np.float32) * 0.0001,
    )
    obs = model.encode(
        {"ext": [5.0, -5.0]},
        body=[],
        prev_action=[],
        dt=0.1,
    )

    estimate = estimator.step(obs, body=[], prev_action=[])
    health = evaluate_estimator_health(estimate, obs, obs_mismatch_threshold=1.0, nis_threshold=10.0)

    assert health.observation_contract_break is True
    assert health.overconfident_estimate is True
    assert health.degraded is True
    assert health.reason
    assert 0.0 <= health.trust < 1.0


def test_residual_linear_self_estimator_falls_back_to_predict_only_when_all_channels_missing() -> None:
    model = TensorObservationModel(
        latent_dim=2,
        ext_size=1,
        body_size=1,
        boundary_size=0,
        action_size=0,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=2,
        A_x=np.eye(2, dtype=np.float32),
    )
    obs = model.encode({"ext": [None], "body_obs": [None]}, body=[0.0], prev_action=[], dt=0.1)

    estimate = estimator.step(obs, body=[0.0], prev_action=[])
    health = evaluate_estimator_health(estimate, obs)

    assert estimate.nis == 0.0
    assert health.degraded is True
    assert "predict_only" in health.reason


def test_residual_linear_self_estimator_rejects_observation_shape_mismatch() -> None:
    class BadJacobianModel(TensorObservationModel):
        def jacobian(self, x_hat, body, prev_action):
            return np.zeros((1, self.latent_dim), dtype=np.float32)

    model = BadJacobianModel(
        latent_dim=2,
        ext_size=2,
        body_size=0,
        boundary_size=0,
        action_size=0,
    )
    estimator = ResidualLinearSelfEstimator(
        observation_model=model,
        latent_dim=2,
        A_x=np.eye(2, dtype=np.float32),
    )
    obs = model.encode({"ext": [0.2, 0.3]}, body=[], prev_action=[], dt=0.1)

    try:
        estimator.step(obs, body=[], prev_action=[])
    except ValueError as exc:
        assert "jacobian" in str(exc).lower() or "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for jacobian shape mismatch")


def test_kernel_runtime_contract_freezes_update_order() -> None:
    contract = KernelStepContract()

    assert contract.update_order == KERNEL_UPDATE_ORDER
    assert contract.update_order == (
        "observe",
        "estimate",
        "memory",
        "qualia",
        "conscious",
        "protect",
        "act",
        "plant",
    )
