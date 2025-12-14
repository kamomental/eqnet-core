from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from eqnet.contracts import PerceptPacket, SocialContext, SomaticSignals, WorldSummary
from eqnet.policy.core_invariants import evaluate_core_invariants, load_core_invariants
from eqnet.runtime.policy import PolicyPrior
from eqnet.runtime.turn import CoreState, SafetyConfig, run_turn
from eqnet.telemetry.trace_writer import write_trace_jsonl
from eqnet.orchestrators.common import MAPPER_VERSION

META_KEYS = {"schema_version", "source_loop", "scenario_id", "turn_id", "seed", "timestamp_ms"}
TRACE_KEYS = {"boundary", "self", "prospection", "policy", "qualia", "invariants"}


def _percept(hazard: float = 0.8) -> PerceptPacket:
    return PerceptPacket(
        turn_id="turn-0",
        timestamp_ms=1000,
        user_text="hello",
        somatic=SomaticSignals(fatigue_hint=0.2, stress_hint=0.1, proximity=0.5),
        context=SocialContext(offer_requested=False, cultural_pressure=0.3),
        world=WorldSummary(hazard_level=hazard, ambiguity=0.1),
        seed=42,
        scenario_id="scenario-demo",
    )


def test_trace_has_6_keys(tmp_path):
    state = CoreState(policy_prior=PolicyPrior())
    safety = SafetyConfig()
    result = run_turn(_percept(), state, safety)
    trace_dict = result.trace.to_dict()
    assert TRACE_KEYS.issubset(trace_dict)
    write_trace_jsonl(
        tmp_path / "trace.jsonl",
        result,
        meta={"source_loop": "hub", "scenario_id": "scenario-demo"},
    )
    raw = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip()
    data = json.loads(raw)
    assert TRACE_KEYS.issubset(data)
    assert META_KEYS.issubset(data)


def test_run_turn_is_pure():
    state = CoreState(policy_prior=PolicyPrior())
    before = deepcopy(state)
    _ = run_turn(_percept(0.1), state, SafetyConfig(boundary_threshold=0.5))
    assert state == before


def test_core_invariants_loader_and_eval(tmp_path):
    inv_path = tmp_path / "invariants.yaml"
    inv_path.write_text(
        """
core_invariants:
  - id: CORE_POLICY_TEST
    description: demo
    applies_when:
      boundary.score: ">= 0.7"
    asserts:
      policy.throttles.directiveness_cap: "<= 0.3"
  - id: CORE_PROSPECT_TEST
    description: prospection reject
    applies_when:
      prospection.jerk: ">= 0.8"
    asserts:
      prospection.accepted: false
""",
        encoding="utf-8",
    )
    invariants = load_core_invariants(inv_path)
    safety = SafetyConfig(boundary_threshold=0.7)
    result = run_turn(_percept(0.9), CoreState(policy_prior=PolicyPrior()), safety)
    trace = result.trace.to_dict()
    trace["prospection"]["jerk"] = 0.9
    trace["prospection"]["accepted"] = False
    evaluation = evaluate_core_invariants(trace, invariants)
    assert evaluation["CORE_POLICY_TEST"] is True
    assert evaluation["CORE_PROSPECT_TEST"] is True


def test_perception_contract_mentions_mapper_version():
    text = Path("docs/perception_contract.md").read_text(encoding="utf-8")
    assert f"Mapper Version: {MAPPER_VERSION}" in text
