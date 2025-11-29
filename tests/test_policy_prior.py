from eqnet.runtime.policy import PolicyPrior, apply_imagery_update


def test_policy_prior_updates_with_reward_signs() -> None:
    prior = PolicyPrior(warmth=0.5, calmness=0.5)

    good = apply_imagery_update(
        prior,
        imagined_traj=[],
        avg_potential=-1.0,
        avg_life_indicator=1.0,
    )
    bad = apply_imagery_update(
        prior,
        imagined_traj=[],
        avg_potential=1.0,
        avg_life_indicator=0.0,
    )

    assert 0.0 <= good.warmth <= 1.0
    assert 0.0 <= good.calmness <= 1.0
    assert good.warmth >= prior.warmth
    assert good.calmness >= prior.calmness

    assert 0.0 <= bad.warmth <= 1.0
    assert 0.0 <= bad.calmness <= 1.0
    assert bad.warmth <= prior.warmth
    assert bad.calmness <= prior.calmness
