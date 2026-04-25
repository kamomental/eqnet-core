import numpy as np

from inner_os.orchestration.field_normalization import (
    FieldNormalizationConfig,
    normalize_field_values,
)


def test_field_normalization_trusts_local_range_when_field_is_consistent() -> None:
    result = normalize_field_values(
        np.array([0.2, 0.25, 0.3, 0.35], dtype=np.float32),
        FieldNormalizationConfig(global_range=1.0),
    )

    assert result.stats.range_trust > 0.8
    assert "noisy_local_range" not in result.stats.reasons
    assert float(result.values[-1]) > float(result.values[0])


def test_field_normalization_blends_to_global_range_when_local_cv_is_noisy() -> None:
    result = normalize_field_values(
        np.array([0.2, 0.21, 0.22, 8.0], dtype=np.float32),
        FieldNormalizationConfig(global_range=1.0),
    )

    assert result.stats.range_trust < 0.5
    assert "noisy_local_range" in result.stats.reasons
    assert abs(result.stats.effective_range - result.stats.global_range) < abs(
        result.stats.effective_range - result.stats.local_range
    )
    assert float(result.values[1]) > 0.0


def test_field_normalization_fog_reduces_range_trust_and_gradient_confidence() -> None:
    clear = normalize_field_values(
        np.array([0.2, 0.25, 0.3, 0.35], dtype=np.float32),
        FieldNormalizationConfig(global_range=1.0, fog_density=0.0),
    )
    foggy = normalize_field_values(
        np.array([0.2, 0.25, 0.3, 0.35], dtype=np.float32),
        FieldNormalizationConfig(global_range=1.0, fog_density=0.7),
    )

    assert foggy.stats.range_trust < clear.stats.range_trust
    assert foggy.stats.gradient_confidence < clear.stats.gradient_confidence
    assert "fog_reduced_trust" in foggy.stats.reasons
