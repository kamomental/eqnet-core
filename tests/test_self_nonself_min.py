# -*- coding: utf-8 -*-

from emot_terrain_lab.mind.self_nonself import NarrativePosterior, RolePosterior, kld


def test_role_posterior_generates_distribution() -> None:
    posterior = RolePosterior(["caregiver", "playful", "researcher"])
    posterior.nudge({"caregiver": 0.5, "playful": 0.2})
    probs = posterior.posterior()
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["caregiver"] > probs["researcher"]


def test_narrative_coherence_and_kld() -> None:
    narr = NarrativePosterior()
    narr.update([("supports", "goal"), ("contradicts", "plan")])
    coherence = narr.coherence()
    assert 0.0 <= coherence <= 1.0
    prev = {"caregiver": 0.5, "playful": 0.5}
    curr = {"caregiver": 0.8, "playful": 0.2}
    assert kld(curr, prev) > 0.0
