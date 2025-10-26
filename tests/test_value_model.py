from devlife.value.model import compute_value_summary, load_value_config


def test_compute_value_summary_basic():
    summary = compute_value_summary(
        extrinsic_signal=0.1,
        novelty_signal=0.2,
        social_alignment=0.7,
        coherence_score=0.6,
        homeostasis_error=0.05,
        qualia_consistency=0.8,
        norm_penalty=0.1,
    )
    assert "total" in summary
    assert "components" in summary
    comps = summary["components"]
    assert 0.0 <= comps["extrinsic"] <= 1.0
    config = load_value_config()
    assert "value_weights" in config
