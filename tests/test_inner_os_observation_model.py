import numpy as np

from inner_os.observation_model import TensorObservationModel


def test_tensor_observation_model_encodes_channel_contract_and_mask() -> None:
    model = TensorObservationModel(
        latent_dim=6,
        ext_size=2,
        body_size=1,
        boundary_size=1,
        action_size=1,
        default_noise=0.1,
    )

    obs = model.encode(
        {
            "ext": [0.2, 0.4],
            "body_obs": [0.6],
            "boundary": [None],
            "action_feedback": [0.8],
        },
        body=[0.5],
        prev_action=[0.1],
        dt=0.1,
        timestamp=1.5,
    )

    assert obs.y.shape == (5,)
    assert obs.mask.tolist() == [True, True, True, False, True]
    assert obs.layout.ext_slice == (0, 2)
    assert obs.layout.body_slice == (2, 3)
    assert obs.layout.channel_for("body").kind == "body"
    assert obs.layout.channel_for("boundary").missing_policy == "masked"
    assert np.allclose(np.diag(obs.R), np.full(5, 0.1, dtype=np.float32))


def test_tensor_observation_model_predict_and_jacobian_follow_layout() -> None:
    model = TensorObservationModel(
        latent_dim=5,
        ext_size=2,
        body_size=1,
        boundary_size=1,
        action_size=1,
    )

    predicted = model.predict(
        x_hat=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        body=[0.5],
        prev_action=[0.25],
    )
    jacobian = model.jacobian(
        x_hat=np.zeros(5, dtype=np.float32),
        body=[0.0],
        prev_action=[0.0],
    )

    assert predicted.shape == (5,)
    assert jacobian.shape == (5, 5)
    assert np.allclose(predicted[:2], [1.0, 2.0])
    assert np.isclose(predicted[2], 3.5)
    assert np.isclose(predicted[3], 4.0)
    assert np.isclose(predicted[4], 5.25)


def test_tensor_observation_model_rejects_non_positive_dt() -> None:
    model = TensorObservationModel(
        latent_dim=2,
        ext_size=1,
        body_size=1,
        boundary_size=0,
        action_size=0,
    )

    try:
        model.encode({"ext": [0.2], "body_obs": [0.3]}, body=[0.0], prev_action=[], dt=0.0)
    except ValueError as exc:
        assert "dt" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive dt")
