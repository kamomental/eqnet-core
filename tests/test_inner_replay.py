# -*- coding: utf-8 -*-

from emot_terrain_lab.mind.inner_replay import InnerReplayController, ReplayConfig, ReplayInputs


def test_inner_replay_execute_branch() -> None:
    cfg = ReplayConfig(theta_prep=0.3, tau_conscious_s=0.2, steps=4, seed=7, keep_s_trace=True)
    ctrl = InnerReplayController(cfg)
    inputs = ReplayInputs(
        chaos_sens=0.9,
        tom_cost=0.2,
        delta_aff_abs=0.1,
        risk=0.2,
        uncertainty=0.1,
        reward_estimate=0.9,
        mood_valence=0.3,
        mood_arousal=0.2,
    )
    outcome = ctrl.run_cycle(inputs, wall_now=1000.0)
    assert outcome.decision == "execute"
    assert 0.0 <= outcome.prep_features["t_prep"] <= cfg.tau_conscious_s
    assert outcome.felt_intent_time >= 1000.0
    assert outcome.u_hat > 0.0


def test_inner_replay_boundary_cancel() -> None:
    cfg = ReplayConfig(
        theta_prep=0.1,
        tau_conscious_s=0.1,
        steps=2,
        seed=11,
        tau_execute=0.4,
        beta_veto=1.0,
    )
    ctrl = InnerReplayController(cfg)
    inputs = ReplayInputs(
        chaos_sens=0.1,
        tom_cost=0.0,
        delta_aff_abs=0.0,
        risk=0.0,
        uncertainty=0.0,
        reward_estimate=0.4,
    )
    outcome = ctrl.run_cycle(inputs)
    assert outcome.decision == "cancel"


def test_inner_replay_handles_nan_inputs() -> None:
    cfg = ReplayConfig(theta_prep=0.2, tau_conscious_s=0.1, steps=3, seed=5)
    ctrl = InnerReplayController(cfg)
    inputs = ReplayInputs(
        chaos_sens=float("nan"),
        tom_cost=-1.0,
        delta_aff_abs=float("inf"),
        risk=-2.0,
        uncertainty=0.5,
        reward_estimate=0.2,
        mood_valence=0.0,
        mood_arousal=0.0,
    )
    outcome = ctrl.run_cycle(inputs)
    assert outcome.decision in {"execute", "cancel"}
    assert outcome.veto_score >= 0.0
