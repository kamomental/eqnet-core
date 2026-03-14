from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, _apply_inner_os_expression_controls
from inner_os.integration_hooks import IntegrationHooks
from inner_os.memory_core import MemoryCore


@dataclass
class _SensorSnapshot:
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _SensorHarness:
    snapshot: _SensorSnapshot


@dataclass
class _SelfHarness:
    current_energy: float = 0.8


@dataclass
class _ProcessTurnHarness:
    _integration_hooks: IntegrationHooks
    _runtime_sensors: _SensorHarness = field(
        default_factory=lambda: _SensorHarness(_SensorSnapshot({"body_stress_index": 0.25, "activity_level": 0.5}))
    )
    _last_shadow_estimate: Optional[Dict[str, Any]] = None
    _last_gate_context: Dict[str, Any] = field(default_factory=dict)
    _self_model: _SelfHarness = field(default_factory=_SelfHarness)
    _last_step_context: Optional[str] = None
    _surface_world_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "zone_id": "market",
            "mode": "reality",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        }
    )

    process_turn = EmotionalHubRuntime.process_turn
    _normalize_perception_summary = EmotionalHubRuntime._normalize_perception_summary
    _visual_reflection_line = EmotionalHubRuntime._visual_reflection_line
    _attach_visual_reflection = EmotionalHubRuntime._attach_visual_reflection
    _inner_os_relational_context = EmotionalHubRuntime._inner_os_relational_context
    _apply_inner_os_surface_policy = EmotionalHubRuntime._apply_inner_os_surface_policy
    _shape_inner_os_surface_text = EmotionalHubRuntime._shape_inner_os_surface_text
    _compose_inner_os_surface_text = EmotionalHubRuntime._compose_inner_os_surface_text
    _inner_os_surface_probe = EmotionalHubRuntime._inner_os_surface_probe
    _inner_os_surface_reopening_line = EmotionalHubRuntime._inner_os_surface_reopening_line
    _inner_os_surface_closing = EmotionalHubRuntime._inner_os_surface_closing
    _inner_os_surface_policy_level = EmotionalHubRuntime._inner_os_surface_policy_level
    _inner_os_surface_policy_intent = EmotionalHubRuntime._inner_os_surface_policy_intent

    def _surface_safety_bias(self) -> float:
        return 0.2

    def _surface_mode(self) -> str:
        return "reality"

    def _dominant_memory_anchor(self) -> str:
        return "harbor slope"

    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am trying to stay with what is grounded. I will avoid overreading the scene.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": [{"id": "vision-1"}]},
            perception_summary={"text": "soft evening slope and signboard"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": {
                "reply": "tentative memory reply",
                "fidelity": 0.88,
                "meta": {"source_class": "self", "audit_event": "OK"},
                "candidate": {"label": "harbor slope"},
            },
        }


@dataclass
class _DegradedHarness(_ProcessTurnHarness):
    def step(
        self,
        *,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_step_context = context
        response = SimpleNamespace(
            text="I am keeping the reading tentative.",
            latency_ms=12.0,
            safety={"rating": "G"},
            controls_used={"mode": "watch"},
            retrieval_summary={"hits": []},
            perception_summary={"error": "vision_error", "detail": "500 Server Error"},
        )
        return {
            "talk_mode": "watch",
            "response_route": "watch",
            "metrics": {},
            "persona_meta": {},
            "heart": {},
            "shadow": None,
            "qualia_gate": {},
            "affect": None,
            "response": response,
            "memory_reference": None,
        }


def test_apply_inner_os_expression_controls_softens_llm_controls() -> None:
    base = {"temperature": 0.48, "top_p": 0.92, "directness": 0.22}
    adjusted = _apply_inner_os_expression_controls(
        base,
        {"tentative_bias": 0.64, "assertiveness_cap": 0.58, "question_bias": 0.35},
    )
    assert adjusted["directness"] < base["directness"]
    assert adjusted["temperature"] < base["temperature"]
    assert adjusted["top_p"] < base["top_p"]
    assert adjusted["inner_os_tentative_bias"] == 0.64
    assert adjusted["inner_os_assertiveness_cap"] == 0.58
    assert adjusted["inner_os_question_bias"] == 0.35


def test_shape_inner_os_surface_text_shortens_check_in_text() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    shaped = harness._shape_inner_os_surface_text(
        "I am trying to stay with what is grounded. I will avoid overreading the scene.",
        intent="check_in",
    )
    assert shaped == "I am trying to stay with what is grounded."


def test_inner_os_surface_closing_depends_on_question_bias() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.12) == ""
    assert harness._inner_os_surface_closing(intent="clarify", question_bias=0.34) == "Then I can answer a little more cleanly."
    assert harness._inner_os_surface_closing(intent="check_in", question_bias=0.34) == "I want to stay with this gently first."


def test_apply_inner_os_surface_policy_prefixes_clarifying_line() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert updated.text.endswith("Then I can answer a little more cleanly.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"


def test_apply_inner_os_surface_policy_keeps_opening_only_when_question_bias_is_low() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.12},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check one small thing before I go further.")
    assert not updated.text.endswith("Then I can answer a little more cleanly.")


def test_apply_inner_os_surface_policy_uses_check_in_prefix() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.34},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert updated.text.startswith("Let me check in gently before I go further.")
    assert "I will avoid overreading the scene." not in updated.text
    assert updated.text.endswith("I want to stay with this gently first.")
    assert updated.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:check_in"



def test_apply_inner_os_surface_policy_adds_relational_probe_for_check_in_afterglow() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded. I will avoid overreading the scene.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.36, "carry_gentleness": True, "interaction_afterglow_intent": "check_in"},
        {"intent": "check_in"},
    )
    assert updated is not None
    assert "Would it help if I stay close to what is visible first?" in updated.text
    assert updated.text.startswith("Let me check in gently before I go further.")


def test_apply_inner_os_surface_policy_adds_reopening_line_when_recovery_returns() -> None:
    harness = _ProcessTurnHarness(_integration_hooks=IntegrationHooks())
    response = SimpleNamespace(text="I am trying to stay with what is grounded.", controls_used={"mode": "watch"})
    updated = harness._apply_inner_os_surface_policy(
        response,
        {"clarify_first": True, "question_bias": 0.3, "allow_reopening": True, "recovery_reopening": 0.31},
        {"intent": "clarify"},
    )
    assert updated is not None
    assert "I think we can open this a little more clearly now." in updated.text


def test_process_turn_injects_inner_os_metadata(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _ProcessTurnHarness(_integration_hooks=hooks)
    harness._last_gate_context["inner_os_social_update_strength"] = 0.64
    harness._last_gate_context["inner_os_identity_update_strength"] = 0.52
    harness._last_gate_context["inner_os_interaction_afterglow"] = 0.31
    harness._last_gate_context["inner_os_interaction_afterglow_intent"] = "clarify"
    result = harness.process_turn(
        user_text="do you remember this new place",
        context="previous framing",
        transferred_lessons=[
            {
                "kind": "transferred_learning",
                "policy_hint": "pause_and_observe_under_ambiguity",
                "summary": "observe before approaching when ambiguity is high",
            }
        ],
    )
    assert result.metrics["inner_os/stress"] >= 0.0
    assert result.metrics["inner_os/temporal_pressure_after"] >= 0.0
    assert result.metrics["inner_os/transferred_lessons_used"] == 1.0
    assert result.metrics["inner_os/interaction_afterglow"] > 0.0
    assert result.metrics["inner_os/replay_intensity"] >= 0.0
    assert result.metrics["inner_os/anticipation_tension"] >= 0.0
    assert result.metrics["inner_os/recovery_reopening"] >= 0.0
    assert result.metrics["inner_os/consolidation_priority"] >= 0.0
    assert result.metrics["inner_os/interference_pressure"] >= 0.0
    assert result.metrics["inner_os/prospective_memory_pull"] >= 0.0
    assert result.metrics["inner_os/object_affordance_bias"] >= 0.0
    assert result.metrics["inner_os/defensive_salience"] >= 0.0
    assert result.metrics["inner_os/surface_policy_active"] == 1.0
    assert result.metrics["inner_os/surface_policy_layered"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_clarify"] == 1.0
    assert result.metrics["inner_os/surface_policy_intent_check_in"] == 0.0
    assert result.response is not None
    assert result.response.text.startswith("Let me check one small thing before I go further.")
    assert result.response.text.endswith("Then I can answer a little more cleanly.")
    assert result.response.retrieval_summary is not None
    assert "inner_os" in result.response.retrieval_summary
    assert result.response.controls is not None
    assert "inner_os" in result.response.controls
    assert result.response.controls["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.response.controls_used["inner_os_surface_policy"] == "clarify_first_prefix:clarify"
    assert "inner_os" in result.qualia_gate
    assert result.persona_meta["inner_os"]["route"] in {"reflex", "conscious", "watch"}
    assert result.persona_meta["inner_os"]["culture_id"] == "coastal"
    assert result.persona_meta["inner_os"]["community_id"] == "harbor_collective"
    assert result.response.controls["inner_os"]["hesitation_bias"] >= 0.0
    assert result.response.controls["inner_os"]["expression_hints"]["tentative_bias"] >= 0.0
    assert round(result.response.controls["inner_os"]["expression_hints"]["express_now"] + result.response.controls["inner_os"]["expression_hints"]["hold_back"], 4) == 1.0
    assert result.persona_meta["inner_os"]["continuity_score"] >= 0.0
    assert result.persona_meta["inner_os"]["culture_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["community_resonance"] >= 0.0
    assert result.persona_meta["inner_os"]["terrain_transition_roughness"] >= 0.0
    assert result.persona_meta["inner_os"]["meaning_pacing"] in {"steady", "slow"}
    assert result.persona_meta["inner_os"]["interaction_pacing"] in {"flow", "check"}
    assert result.persona_meta["inner_os"]["surface_policy_level"] in {"none", "layered", "prefix_only"}
    assert result.persona_meta["inner_os"]["surface_policy_intent"] == "clarify"
    assert result.persona_meta["inner_os"]["interaction_afterglow"] > 0.0
    assert result.persona_meta["inner_os"]["interaction_afterglow_intent"] == "clarify"
    assert result.persona_meta["inner_os"]["replay_intensity"] >= 0.0
    assert result.persona_meta["inner_os"]["anticipation_tension"] >= 0.0
    assert result.persona_meta["inner_os"]["meaning_inertia"] >= 0.0
    assert result.persona_meta["inner_os"]["recovery_reopening"] >= 0.0
    assert result.persona_meta["inner_os"]["consolidation_priority"] >= 0.0
    assert result.persona_meta["inner_os"]["interference_pressure"] >= 0.0
    assert result.persona_meta["inner_os"]["prospective_memory_pull"] >= 0.0
    assert result.persona_meta["inner_os"]["tentative_bias"] >= 0.0
    assert harness._last_step_context is not None
    assert "Prefer grounded observations" in harness._last_step_context
    assert "clarifying question" in harness._last_step_context
    assert "inner_os_temporal_pressure" in harness._last_gate_context
    assert harness._last_gate_context["inner_os_interaction_afterglow"] > 0.0
    assert harness._last_gate_context["inner_os_interaction_afterglow_intent"] == "clarify"
    assert harness._last_gate_context["inner_os_replay_intensity"] >= 0.0
    assert harness._last_gate_context["inner_os_anticipation_tension"] >= 0.0
    assert harness._last_gate_context["inner_os_recovery_reopening"] >= 0.0
    memory_path = tmp_path / "inner_os_memory.jsonl"
    assert memory_path.exists()
    records = [json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(record.get("kind") == "verified" for record in records)
    assert any(record.get("kind") == "reconstructed" for record in records)
    assert any(record.get("kind") == "identity_trace" for record in records)
    assert any(record.get("kind") == "relationship_trace" for record in records)
    assert any(record.get("kind") == "community_profile_trace" for record in records)
    observed_records = [record for record in records if record.get("kind") == "observed_real" or record.get("source") == "post_turn"]
    assert any(record.get("surface_policy_active") == 1.0 for record in observed_records)
    assert any(record.get("surface_policy_level") == "layered" for record in observed_records)
    assert any(record.get("surface_policy_intent") == "clarify" for record in observed_records)


def test_visual_reflection_uses_degraded_surface_text(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    harness = _DegradedHarness(_integration_hooks=hooks)
    normalized = harness._normalize_perception_summary({"error": "vision_error", "detail": "500 Server Error"})
    response = SimpleNamespace(text="I am keeping the reading tentative.", perception_summary=None)
    response = harness._attach_visual_reflection(response, normalized)
    assert response is not None
    assert normalized is not None
    assert normalized["status"] == "degraded"
    assert "visual read did not settle" in response.text
