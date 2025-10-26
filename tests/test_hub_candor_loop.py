# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.qualia_bridge import measure_language_loss
from emot_terrain_lab.nlg.disclosure import craft_payload, decide_disclosure
from emot_terrain_lab.sense.envelope import SenseEnvelope
from emot_terrain_lab.sense.residuals import compute_residual


def test_candor_loop_emits_bridges_when_loss_high() -> None:
    env = SenseEnvelope(
        id="test",
        modality="vision",
        features={"jiggle_hz": 0.9, "firmness": 0.2},
        confidence=0.8,
        source="external",
        t_tau=0.0,
        tags=["pudding"],
    )
    lang = measure_language_loss(env, "固めのプリン", None)
    residual = compute_residual(
        env.features,
        lang["reconstructed"],
        {"jiggle_hz": 0.9, "firmness": 0.6},
        {"jiggle_hz": 0.5, "firmness": 0.5},
    )
    decision = decide_disclosure(residual["delta"], residual["top"], {"warn": 0.1, "must": 0.2, "ask": 0.3})
    payload = craft_payload(
        decision["level"],
        decision["targets"],
        locale="ja",
        persona="default",
        templates={
            "ja": {"warn": "warn {targets}", "must": "must {targets}", "ask": "ask {ask}"},
            "targets_map": {"firmness": "かたさ"},
            "asks": {"firmness": ["固さを教えて"]},
        },
        metaphors={"missing_to_phrase": {"firmness": ["しっかり食感"]}},
    )

    assert payload["disclosure"]
    assert "bridges" in payload
