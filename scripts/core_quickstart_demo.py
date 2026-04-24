from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"module load failed: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REACTION_CONTRACT_MODULE = _load_module(
    "core_quickstart_reaction_contract",
    "inner_os/expression/reaction_contract.py",
)
_JOINT_STATE_MODULE = _load_module(
    "core_quickstart_joint_state",
    "inner_os/joint_state.py",
)
_ATTRIBUTION_MODULE = _load_module(
    "core_quickstart_self_other_attribution",
    "inner_os/self_model/self_other_attribution_state.py",
)
_SHARED_PRESENCE_MODULE = _load_module(
    "core_quickstart_shared_presence",
    "inner_os/shared_presence_state.py",
)
_SUBJECTIVE_SCENE_MODULE = _load_module(
    "core_quickstart_subjective_scene",
    "inner_os/world_model/subjective_scene_state.py",
)
_CONTRACT_EVAL_MODULE = _load_module(
    "core_quickstart_contract_eval",
    "inner_os/evaluation/conversation_contract_eval.py",
)

derive_reaction_contract = _REACTION_CONTRACT_MODULE.derive_reaction_contract
derive_joint_state = _JOINT_STATE_MODULE.derive_joint_state
SelfOtherAttributionState = _ATTRIBUTION_MODULE.SelfOtherAttributionState
derive_self_other_attribution_state = _ATTRIBUTION_MODULE.derive_self_other_attribution_state
SharedPresenceState = _SHARED_PRESENCE_MODULE.SharedPresenceState
derive_shared_presence_state = _SHARED_PRESENCE_MODULE.derive_shared_presence_state
SubjectiveSceneState = _SUBJECTIVE_SCENE_MODULE.SubjectiveSceneState
derive_subjective_scene_state = _SUBJECTIVE_SCENE_MODULE.derive_subjective_scene_state
CORE_QUICKSTART_EXPECTATIONS = _CONTRACT_EVAL_MODULE.CORE_QUICKSTART_EXPECTATIONS
evaluate_reaction_contract_against_expectation = (
    _CONTRACT_EVAL_MODULE.evaluate_reaction_contract_against_expectation
)


@dataclass(frozen=True)
class CoreDemoScenario:
    name: str
    description: str
    input_text: str
    camera_observation: dict[str, Any]
    world_state: dict[str, Any]
    self_state: dict[str, Any]
    organism_state: dict[str, Any]
    external_field_state: dict[str, Any]
    person_registry: dict[str, Any]
    shared_moment_state: dict[str, Any]
    listener_action_state: dict[str, Any]
    live_engagement_state: dict[str, Any]
    meaning_update_state: dict[str, Any]
    memory_dynamics_state: dict[str, Any]


SCENARIOS: dict[str, CoreDemoScenario] = {
    "small_shared_moment": CoreDemoScenario(
        name="small_shared_moment",
        description="小さい共有モーメントを小さく一緒に受ける。",
        input_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        camera_observation={
            "egocentric_closeness": 0.72,
            "workspace_proximity": 0.78,
            "frontal_alignment": 0.7,
            "motion_salience": 0.26,
            "self_reference_score": 0.42,
            "shared_reference_score": 0.72,
            "familiarity_hint": 0.7,
            "comfort_hint": 0.74,
            "perspective_match": 0.76,
            "contingency_match": 0.58,
            "sensorimotor_consistency": 0.54,
            "uncertainty": 0.16,
        },
        world_state={
            "scene_familiarity": 0.68,
            "continuity_hint": 0.72,
            "social_relation_pull": 0.6,
            "task_surface_salience": 0.54,
        },
        self_state={
            "curiosity": 0.42,
            "safety_margin": 0.74,
            "uncertainty": 0.14,
            "social_tension": 0.18,
            "task_load": 0.44,
        },
        organism_state={
            "attunement": 0.7,
            "grounding": 0.62,
            "relation_pull": 0.66,
            "play_window": 0.46,
            "protective_tension": 0.22,
            "social_mode": "near",
        },
        external_field_state={
            "continuity_pull": 0.7,
            "safety_envelope": 0.76,
            "social_pressure": 0.16,
            "ambiguity_load": 0.12,
            "novelty": 0.2,
        },
        person_registry={
            "confidence": 0.62,
            "other_presence": 0.34,
        },
        shared_moment_state={
            "moment_kind": "laugh",
            "score": 0.52,
            "jointness": 0.68,
            "afterglow": 0.46,
        },
        listener_action_state={
            "state": "warm_laugh_ack",
            "acknowledgement_room": 0.48,
            "laughter_room": 0.36,
            "filler_room": 0.16,
        },
        live_engagement_state={
            "state": "pickup_comment",
            "comment_pickup_room": 0.42,
            "topic_seed_room": 0.2,
            "riff_room": 0.24,
        },
        meaning_update_state={
            "relation_update": "shared_smile_window",
            "world_update": "shared_object_notice",
        },
        memory_dynamics_state={
            "dominant_relation_type": "same_anchor",
            "dominant_causal_type": "enabled_by",
            "monument_salience": 0.36,
            "activation_confidence": 0.42,
            "memory_tension": 0.18,
        },
    ),
    "guarded_uncertainty": CoreDemoScenario(
        name="guarded_uncertainty",
        description="踏み込みを控えて hold / defer に寄る。",
        input_text="まだうまく言葉にできないんだけど、ちょっと重い感じだけ残ってて。",
        camera_observation={
            "egocentric_closeness": 0.34,
            "workspace_proximity": 0.28,
            "frontal_alignment": 0.38,
            "motion_salience": 0.18,
            "self_reference_score": 0.24,
            "shared_reference_score": 0.18,
            "familiarity_hint": 0.28,
            "comfort_hint": 0.22,
            "perspective_match": 0.3,
            "contingency_match": 0.18,
            "sensorimotor_consistency": 0.16,
            "uncertainty": 0.72,
            "tension_hint": 0.64,
        },
        world_state={
            "scene_familiarity": 0.22,
            "continuity_hint": 0.34,
            "social_relation_pull": 0.24,
        },
        self_state={
            "curiosity": 0.24,
            "safety_margin": 0.3,
            "uncertainty": 0.64,
            "social_tension": 0.56,
            "task_load": 0.32,
        },
        organism_state={
            "attunement": 0.34,
            "grounding": 0.44,
            "relation_pull": 0.28,
            "play_window": 0.08,
            "protective_tension": 0.64,
            "social_mode": "guarded",
        },
        external_field_state={
            "continuity_pull": 0.32,
            "safety_envelope": 0.34,
            "social_pressure": 0.42,
            "ambiguity_load": 0.58,
            "novelty": 0.18,
        },
        person_registry={
            "confidence": 0.18,
            "other_presence": 0.2,
        },
        shared_moment_state={},
        listener_action_state={
            "state": "careful_listen",
            "acknowledgement_room": 0.18,
            "laughter_room": 0.0,
            "filler_room": 0.22,
        },
        live_engagement_state={
            "state": "quiet_hold",
            "comment_pickup_room": 0.08,
            "topic_seed_room": 0.04,
            "riff_room": 0.0,
        },
        meaning_update_state={
            "preserve_guard": "keep_boundary",
        },
        memory_dynamics_state={
            "dominant_relation_type": "unfinished_carry",
            "dominant_causal_type": "reopened_by",
            "monument_salience": 0.18,
            "activation_confidence": 0.22,
            "memory_tension": 0.52,
        },
    ),
}


def build_core_demo_result(
    *,
    scenario_name: str,
    input_text: str | None = None,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_name]
    text = (input_text or scenario.input_text).strip()

    subjective_scene = derive_subjective_scene_state(
        camera_observation=scenario.camera_observation,
        world_state=scenario.world_state,
        self_state=scenario.self_state,
        external_field_state=scenario.external_field_state,
    )
    attribution = derive_self_other_attribution_state(
        camera_observation=scenario.camera_observation,
        subjective_scene_state=subjective_scene.to_dict(),
        self_state=scenario.self_state,
        person_registry=scenario.person_registry,
    )
    base_joint = derive_joint_state(
        shared_moment_state=scenario.shared_moment_state,
        listener_action_state=scenario.listener_action_state,
        live_engagement_state=scenario.live_engagement_state,
        meaning_update_state=scenario.meaning_update_state,
        organism_state=scenario.organism_state,
        external_field_state=scenario.external_field_state,
        memory_dynamics_state=scenario.memory_dynamics_state,
        subjective_scene_state=subjective_scene.to_dict(),
        self_other_attribution_state=attribution.to_dict(),
    )
    shared_presence = derive_shared_presence_state(
        subjective_scene_state=subjective_scene.to_dict(),
        self_other_attribution_state=attribution.to_dict(),
        joint_state=base_joint.to_dict(),
        organism_state=scenario.organism_state,
        external_field_state=scenario.external_field_state,
    )
    joint_state = derive_joint_state(
        previous_state=base_joint,
        shared_moment_state=scenario.shared_moment_state,
        listener_action_state=scenario.listener_action_state,
        live_engagement_state=scenario.live_engagement_state,
        meaning_update_state=scenario.meaning_update_state,
        organism_state=scenario.organism_state,
        external_field_state=scenario.external_field_state,
        memory_dynamics_state=scenario.memory_dynamics_state,
        subjective_scene_state=subjective_scene.to_dict(),
        self_other_attribution_state=attribution.to_dict(),
        shared_presence_state=shared_presence.to_dict(),
    )

    contract_inputs = _build_contract_inputs(
        joint_state=joint_state.to_dict(),
        shared_presence=shared_presence,
        attribution=attribution,
        subjective_scene=subjective_scene,
        scenario=scenario,
    )
    reaction_contract = derive_reaction_contract(**contract_inputs)
    expectation = CORE_QUICKSTART_EXPECTATIONS[scenario.name]
    evaluation = evaluate_reaction_contract_against_expectation(
        reaction_contract=reaction_contract.to_dict(),
        expectation=expectation,
    )

    return {
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "input_text": text,
        },
        "subjective_scene": subjective_scene.to_dict(),
        "self_other_attribution": attribution.to_dict(),
        "shared_presence": shared_presence.to_dict(),
        "joint_state": joint_state.to_dict(),
        "expected_contract": expectation.to_dict(),
        "evaluation": evaluation.to_dict(),
        "reaction_contract": reaction_contract.to_dict(),
        "response_guideline": _render_response_guideline(reaction_contract.to_dict()),
    }


def _build_contract_inputs(
    *,
    joint_state: dict[str, Any],
    shared_presence: SharedPresenceState,
    attribution: SelfOtherAttributionState,
    subjective_scene: SubjectiveSceneState,
    scenario: CoreDemoScenario,
) -> dict[str, Any]:
    common_ground = _float01(joint_state.get("common_ground"))
    shared_delight = _float01(joint_state.get("shared_delight"))
    shared_tension = _float01(joint_state.get("shared_tension"))
    boundary_stability = shared_presence.boundary_stability
    unknown_likelihood = attribution.unknown_likelihood

    guarded = (
        unknown_likelihood >= 0.5
        or boundary_stability <= 0.4
        or shared_tension > shared_delight + 0.12
    )
    response_strategy = "respectful_wait" if guarded else "shared_world_next_step"
    response_channel = "hold" if guarded else "speak"
    execution_mode = "defer_with_presence" if guarded else "shared_progression"
    boundary_mode = "contain" if guarded else "soft_hold"
    shape_id = "reflect_hold" if guarded else "bright_bounce"
    conversation_phase = "reopening_thread" if guarded else "bright_continuity"
    response_length = "short"
    preserve = "leave_open" if guarded else "keep_it_small"
    offer = "" if guarded else "brief_shared_smile"
    recent_dialogue_state = "reopening_thread" if guarded else "continuing_thread"
    question_budget = 0

    return {
        "interaction_policy": {
            "response_strategy": response_strategy,
            "conversation_contract": {
                "response_action_now": {"question_budget": question_budget}
            },
            "recent_dialogue_state": {"state": recent_dialogue_state},
            "surface_profile": {"response_length": response_length},
        },
        "action_posture": {
            "boundary_mode": boundary_mode,
            "question_budget": question_budget,
            "social_topology_name": "one_to_one",
        },
        "actuation_plan": {
            "execution_mode": execution_mode,
            "response_channel": response_channel,
            "wait_before_action": "brief" if guarded else "",
        },
        "discourse_shape": {
            "shape_id": shape_id,
            "question_budget": question_budget,
        },
        "surface_context_packet": {
            "conversation_phase": conversation_phase,
            "constraints": {
                "max_questions": question_budget,
                "prefer_return_point": guarded,
            },
            "surface_profile": {"response_length": response_length},
            "source_state": {
                "utterance_reason_offer": offer,
                "utterance_reason_preserve": preserve,
                "joint_common_ground": common_ground,
                "organism_social_mode": scenario.organism_state.get("social_mode", ""),
                "recent_dialogue_state": recent_dialogue_state,
                "organism_protective_tension": scenario.organism_state.get(
                    "protective_tension", 0.0
                ),
                "external_field_social_pressure": scenario.external_field_state.get(
                    "social_pressure", 0.0
                ),
                "self_other_unknown_likelihood": unknown_likelihood,
                "self_other_dominant_attribution": attribution.dominant_attribution,
                "shared_presence_mode": shared_presence.dominant_mode,
                "shared_presence_co_presence": shared_presence.co_presence,
                "shared_presence_boundary_stability": shared_presence.boundary_stability,
                "subjective_scene_anchor_frame": subjective_scene.anchor_frame,
                "subjective_scene_shared_scene_potential": subjective_scene.shared_scene_potential,
                "subjective_scene_familiarity": subjective_scene.familiarity,
            },
        },
        "turn_delta": {
            "kind": conversation_phase,
            "preferred_act": "leave_return_point_from_anchor"
            if guarded
            else "light_bounce",
        },
    }


def _render_response_guideline(reaction_contract: dict[str, Any]) -> str:
    stance = str(reaction_contract.get("stance") or "")
    scale = str(reaction_contract.get("scale") or "")
    question_budget = int(reaction_contract.get("question_budget") or 0)
    interpretation_budget = str(reaction_contract.get("interpretation_budget") or "")
    timing_mode = str(reaction_contract.get("timing_mode") or "")

    if stance == "hold":
        return "まだ聞きに行かず、少し間を保つ。説明や解釈より hold / defer を優先する。"
    if (
        stance == "join"
        and scale == "small"
        and question_budget == 0
        and interpretation_budget == "none"
    ):
        return "小さく一緒に受ける。質問や解釈は足さず、共有モーメントのサイズを保つ。"
    if timing_mode == "quick_ack":
        return "短い相槌で受ける。主導権は奪わない。"
    return "現在の contract に従って、過不足のない反応を選ぶ。"


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EQNet の core loop と reaction contract を最短で確認する quickstart demo。",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="small_shared_moment",
        help="確認したい会話シナリオ。",
    )
    parser.add_argument(
        "--text",
        default="",
        help="シナリオ既定の入力文を上書きする。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果を JSON で出力する。",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = build_core_demo_result(
        scenario_name=args.scenario,
        input_text=args.text or None,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("EQNet Core Quickstart")
    print("=====================")
    print(f"シナリオ: {result['scenario']['name']}")
    print(f"説明: {result['scenario']['description']}")
    print(f"入力: {result['scenario']['input_text']}")
    print()
    print("[subjective_scene]")
    print(json.dumps(result["subjective_scene"], ensure_ascii=False, indent=2))
    print()
    print("[self_other_attribution]")
    print(json.dumps(result["self_other_attribution"], ensure_ascii=False, indent=2))
    print()
    print("[shared_presence]")
    print(json.dumps(result["shared_presence"], ensure_ascii=False, indent=2))
    print()
    print("[joint_state]")
    print(json.dumps(result["joint_state"], ensure_ascii=False, indent=2))
    print()
    print("[expected_contract]")
    print(json.dumps(result["expected_contract"], ensure_ascii=False, indent=2))
    print()
    print("[reaction_contract]")
    print(json.dumps(result["reaction_contract"], ensure_ascii=False, indent=2))
    print()
    print("[evaluation]")
    print(json.dumps(result["evaluation"], ensure_ascii=False, indent=2))
    print()
    print("[response_guideline]")
    print(result["response_guideline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
