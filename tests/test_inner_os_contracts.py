from inner_os.access.models import AttentionState, ForegroundState
from inner_os.access.selector import select_foreground
from inner_os.expression.models import ResponsePlan
from inner_os.memory.continuity import IdentityObservation, score_identity_continuity
from inner_os.grounding.models import ObservationBundle
from inner_os.self_model import person_registry_from_snapshot
from inner_os.self_model.models import PersonNode, PersonRegistry, SelfState
from inner_os.value_system.models import ValueState
from inner_os.value_system.terrain import compute_value_state
from inner_os.world_model.models import WorldState


def test_contract_models_are_typed_and_instantiable() -> None:
    observation = ObservationBundle()
    world = WorldState()
    self_state = SelfState()
    registry = PersonRegistry()
    value = ValueState()
    foreground = ForegroundState()
    response = ResponsePlan(speech_act="report")
    assert observation.observation_uncertainty == 1.0
    assert world.uncertainty == 1.0
    assert self_state.uncertainty == 1.0
    assert registry.uncertainty == 1.0
    assert value.value_axes == {}
    assert foreground.reportable_facts == []
    assert response.speech_act == "report"


def test_continuity_contract_keeps_ambiguity_explicit() -> None:
    update = score_identity_continuity(
        None,
        IdentityObservation(
            person_id_hint="user",
            summary="slow gait",
            stable_traits={"gait_cycle": 0.7},
            ambiguity=0.6,
        ),
    )
    assert update.person_id == "user"
    assert update.ambiguity == 0.6
    assert update.confidence > 0.0


def test_access_contract_exposes_candidates_and_continuity_focus() -> None:
    registry = PersonRegistry()
    registry.persons["user"] = PersonNode(
        person_id="user",
        confidence=0.9,
        ambiguity_flag=False,
    )
    value = compute_value_state(
        WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.3),
        SelfState(fatigue=0.6, arousal=0.4, social_tension=0.2),
        registry,
    )
    foreground = select_foreground(
        WorldState(object_states={"user": "person", "lamp": "object"}),
        SelfState(fatigue=0.6),
        value,
        AttentionState(continuity_bias=0.8),
        registry,
    )
    assert foreground.candidates
    assert "continuity" in foreground.selection_reasons
    assert foreground.continuity_focus == ["person:user"]
    assert any(note.startswith("terrain_energy:") for note in foreground.uncertainty_notes)
    assert foreground.reportability_scores["user"] >= foreground.reportability_scores["lamp"]
    assert "user" in foreground.memory_candidates
    assert "continuity" in foreground.memory_reasons["user"]


def test_person_registry_snapshot_can_bias_foreground_for_same_partner() -> None:
    registry = person_registry_from_snapshot(
        {
            "persons": {
                "user": {
                    "person_id": "user",
                    "stable_traits": {"community_marker": 1.0},
                    "adaptive_traits": {
                        "attachment": 0.84,
                        "familiarity": 0.79,
                        "trust_memory": 0.82,
                        "continuity_score": 0.74,
                        "social_grounding": 0.7,
                    },
                    "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                    "confidence": 0.86,
                    "ambiguity_flag": False,
                }
            },
            "uncertainty": 0.18,
        }
    )
    value = compute_value_state(
        WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.18),
        SelfState(fatigue=0.2, arousal=0.3, social_tension=0.1),
        registry,
    )
    foreground = select_foreground(
        WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.18),
        SelfState(fatigue=0.2),
        value,
        AttentionState(continuity_bias=0.9),
        registry,
    )
    assert foreground.candidates[0].entity_id == "user"
    assert "partner-trace" in foreground.candidates[0].reasons
    assert "social" in foreground.memory_reasons["user"]


def test_partner_trace_strength_varies_by_community_and_personality_axes() -> None:
    registry = person_registry_from_snapshot(
        {
            "persons": {
                "user": {
                    "person_id": "user",
                    "stable_traits": {
                        "community_marker": 1.0,
                        "culture_marker": 1.0,
                        "role_marker": 1.0,
                    },
                    "adaptive_traits": {
                        "attachment": 0.82,
                        "familiarity": 0.78,
                        "trust_memory": 0.8,
                        "continuity_score": 0.73,
                        "social_grounding": 0.68,
                    },
                    "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                    "confidence": 0.88,
                    "ambiguity_flag": False,
                }
            },
            "uncertainty": 0.18,
        }
    )
    world = WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.18)
    value = compute_value_state(
        world,
        SelfState(fatigue=0.2, arousal=0.3, social_tension=0.1),
        registry,
    )
    open_foreground = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.1),
        value,
        AttentionState(
            continuity_bias=0.9,
            affiliation_bias=0.74,
            caution_bias=0.18,
            community_bias=0.82,
            culture_bias=0.78,
        ),
        registry,
    )
    guarded_foreground = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.62),
        value,
        AttentionState(
            continuity_bias=0.9,
            affiliation_bias=0.22,
            caution_bias=0.82,
            community_bias=0.18,
            culture_bias=0.16,
        ),
        registry,
    )
    open_user = next(item for item in open_foreground.candidates if item.entity_id == "user")
    guarded_user = next(item for item in guarded_foreground.candidates if item.entity_id == "user")
    assert "community-trace" in open_user.reasons
    assert open_user.salience > guarded_user.salience
    assert guarded_user.salience < 1.0
    assert "community" in open_foreground.memory_reasons["user"]


def test_partner_style_relief_and_caution_can_shift_foreground_weight() -> None:
    registry = person_registry_from_snapshot(
        {
            "persons": {
                "user": {
                    "person_id": "user",
                    "stable_traits": {"community_marker": 1.0},
                    "adaptive_traits": {
                        "attachment": 0.82,
                        "familiarity": 0.78,
                        "trust_memory": 0.8,
                        "continuity_score": 0.73,
                        "social_grounding": 0.68,
                    },
                    "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                    "confidence": 0.88,
                    "ambiguity_flag": False,
                }
            },
            "uncertainty": 0.18,
        }
    )
    world = WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.18)
    value = compute_value_state(world, SelfState(fatigue=0.2, social_tension=0.18), registry)
    warm = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.18),
        value,
        AttentionState(continuity_bias=0.9, partner_style_relief=0.18, partner_style_caution=0.0),
        registry,
    )
    cautious = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.18),
        value,
        AttentionState(continuity_bias=0.9, partner_style_relief=0.0, partner_style_caution=0.2),
        registry,
    )
    warm_user = next(item for item in warm.candidates if item.entity_id == "user")
    cautious_user = next(item for item in cautious.candidates if item.entity_id == "user")
    assert warm_user.salience > cautious_user.salience
    assert "partner-style-relief" in warm_user.reasons
    assert "partner-style-caution" in cautious_user.reasons


def test_relational_mood_bias_can_shift_foreground_weight() -> None:
    registry = person_registry_from_snapshot(
        {
            "persons": {
                "user": {
                    "person_id": "user",
                    "stable_traits": {"community_marker": 1.0},
                    "adaptive_traits": {
                        "attachment": 0.82,
                        "familiarity": 0.78,
                        "trust_memory": 0.8,
                        "continuity_score": 0.73,
                        "social_grounding": 0.68,
                    },
                    "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                    "confidence": 0.88,
                    "ambiguity_flag": False,
                }
            },
            "uncertainty": 0.18,
        }
    )
    world = WorldState(object_states={"user": "person", "lamp": "object"}, uncertainty=0.18)
    value = compute_value_state(world, SelfState(fatigue=0.2, social_tension=0.18), registry)
    baseline = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.18),
        value,
        AttentionState(continuity_bias=0.9),
        registry,
    )
    mooded = select_foreground(
        world,
        SelfState(fatigue=0.2, social_tension=0.18),
        value,
        AttentionState(
            continuity_bias=0.9,
            relational_future_pull=0.22,
            shared_world_pull=0.24,
            relational_care=0.18,
        ),
        registry,
    )
    baseline_user = next(item for item in baseline.candidates if item.entity_id == "user")
    mooded_user = next(item for item in mooded.candidates if item.entity_id == "user")
    assert mooded_user.salience > baseline_user.salience
    assert "future-pull" in mooded_user.reasons
    assert "shared-world" in mooded_user.reasons
    assert "care-trace" in mooded_user.reasons
