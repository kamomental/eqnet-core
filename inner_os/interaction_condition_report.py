from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class InteractionConditionReport:
    scene_lines: tuple[str, ...] = ()
    relation_lines: tuple[str, ...] = ()
    memory_lines: tuple[str, ...] = ()
    integration_lines: tuple[str, ...] = ()
    report_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_interaction_condition_report(
    *,
    scene_state: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
    relation_context: Mapping[str, Any] | None = None,
    memory_context: Mapping[str, Any] | None = None,
) -> InteractionConditionReport:
    scene = dict(scene_state or {})
    resonance = dict(resonance_evaluation or {})
    relation = dict(relation_context or {})
    memory = dict(memory_context or {})
    estimated_other_person_state = dict(resonance.get("estimated_other_person_state") or {})

    scene_lines = _build_scene_lines(scene, estimated_other_person_state)
    relation_lines = _build_relation_lines(relation)
    memory_lines = _build_memory_lines(memory)
    integration_lines = _build_integration_lines(
        estimated_other_person_state=estimated_other_person_state,
        scene=scene,
        relation=relation,
        memory=memory,
    )

    report_lines = (
        tuple(f"場面が効いていること: {line}" for line in scene_lines)
        + tuple(f"相手との関係が効いていること: {line}" for line in relation_lines)
        + tuple(f"記憶が効いていること: {line}" for line in memory_lines)
        + tuple(f"統合した判断: {line}" for line in integration_lines)
    )

    return InteractionConditionReport(
        scene_lines=scene_lines,
        relation_lines=relation_lines,
        memory_lines=memory_lines,
        integration_lines=integration_lines,
        report_lines=report_lines,
    )


def _build_scene_lines(
    scene: Mapping[str, Any],
    estimated_other_person_state: Mapping[str, Any],
) -> tuple[str, ...]:
    lines: list[str] = []
    scene_family = str(scene.get("scene_family") or "").strip()
    privacy_level = _as_float(scene.get("privacy_level"))
    norm_pressure = _as_float(scene.get("norm_pressure"))
    safety_margin = _as_float(scene.get("safety_margin"))
    environmental_load = _as_float(scene.get("environmental_load"))

    if scene_family == "attuned_presence":
        lines.append("system は、いまは相手に強く踏み込まず、そばで受け止める方が合う場面だと見ています。")
    elif scene_family == "reverent_distance":
        lines.append("system は、いまは距離を少し保ち、相手に無理に近づかない方が合う場面だと見ています。")
    elif scene_family == "repair_window":
        lines.append("system は、いまは関係の引っかかりを増やさないように慎重に触れる場面だと見ています。")
    elif scene_family == "shared_world":
        lines.append("system は、いまは一緒に次の一歩を見やすい場面だと見ています。")
    elif scene_family == "guarded_boundary":
        lines.append("system は、いまは境界を守り、話を広げすぎない方がよい場面だと見ています。")

    if privacy_level <= 0.28:
        lines.append("system は、いまは人目や外部の影響があり、話を狭く保つ方がよい場面だと見ています。")
    elif privacy_level >= 0.72:
        lines.append("system は、いまは落ち着いて話しやすい場面だと見ています。")

    if norm_pressure >= 0.62:
        lines.append("system は、いまは丁寧さや言い方の慎重さを強めた方がよいと見ています。")
    if safety_margin <= 0.34:
        lines.append("system は、いまは安全余裕が少なく、急に踏み込まない方がよいと見ています。")
    if environmental_load >= 0.58:
        lines.append("system は、いまは周囲の負荷が高く、話題を広げすぎない方がよいと見ています。")

    detail_room_level = str(estimated_other_person_state.get("detail_room_level") or "").strip()
    if detail_room_level == "low":
        lines.append("system は、いまは相手が詳しく話す余裕が少ない場面だと見ています。")

    if not lines:
        if scene_family:
            lines.append(f"system は、いまの場面を「{scene_family}」として読み、話の広げ方を慎重に決めようとしています。")
        else:
            lines.append("system は、いまの場面条件を見ながら、話の広げ方を慎重に決めようとしています。")

    return tuple(dict.fromkeys(line for line in lines if line))


def _build_relation_lines(relation: Mapping[str, Any]) -> tuple[str, ...]:
    lines: list[str] = []
    relation_bias_strength = _as_float(relation.get("relation_bias_strength"))
    recent_strain = _as_float(relation.get("recent_strain"))
    trust_memory = _as_float(relation.get("trust_memory"))
    familiarity = _as_float(relation.get("familiarity"))
    attachment = _as_float(relation.get("attachment"))
    partner_timing_hint = str(relation.get("partner_timing_hint") or "").strip()
    partner_stance_hint = str(relation.get("partner_stance_hint") or "").strip()
    partner_social_interpretation = str(relation.get("partner_social_interpretation") or "").strip()

    if recent_strain >= 0.45:
        lines.append("system は、この相手とのあいだに最近の引っかかりが残っていると見ています。")
    if (
        relation_bias_strength >= 0.28
        or familiarity >= 0.55
        or trust_memory >= 0.55
        or attachment >= 0.55
    ):
        lines.append("system は、この相手とのこれまでのつながりを今回の判断に強く使っています。")

    if partner_timing_hint == "delayed":
        lines.append("system は、この相手には少し間を置いて入る方がよいと見ています。")
    elif partner_timing_hint == "open":
        lines.append("system は、この相手には比較的そのまま入ってもよいと見ています。")

    if partner_stance_hint == "respectful":
        lines.append("system は、この相手には丁寧さを強めた方がよいと見ています。")
    elif partner_stance_hint == "familiar":
        lines.append("system は、この相手には親しさを少し出してよいと見ています。")

    if "repair" in partner_social_interpretation:
        lines.append("system は、この相手とのいまの関係には修復を意識した接し方が必要だと見ています。")
    elif partner_social_interpretation:
        lines.append(
            f"system は、この相手との関係を「{partner_social_interpretation}」として読んでいます。"
        )

    if not lines:
        lines.append("system は、この相手との関係を見ながら、距離の取り方を決めようとしています。")

    return tuple(dict.fromkeys(line for line in lines if line))


def _build_memory_lines(memory: Mapping[str, Any]) -> tuple[str, ...]:
    lines: list[str] = []
    relation_seed_summary = str(memory.get("relation_seed_summary") or "").strip()
    long_term_theme_summary = str(memory.get("long_term_theme_summary") or "").strip()
    conscious_residue_summary = str(memory.get("conscious_residue_summary") or "").strip()
    memory_anchor = str(memory.get("memory_anchor") or "").strip()

    if relation_seed_summary:
        lines.append(f"system は、関係の手がかりとして「{relation_seed_summary}」を参照しています。")
    if long_term_theme_summary:
        lines.append(f"system は、長く残っている流れとして「{long_term_theme_summary}」を参照しています。")
    if conscious_residue_summary:
        lines.append(f"system は、前回から残っているものとして「{conscious_residue_summary}」をまだ持っています。")
    if memory_anchor:
        lines.append(f"system は、「{memory_anchor}」を今回の記憶の足場にしています。")

    return tuple(dict.fromkeys(line for line in lines if line))


def _build_integration_lines(
    *,
    estimated_other_person_state: Mapping[str, Any],
    scene: Mapping[str, Any],
    relation: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> tuple[str, ...]:
    lines: list[str] = []
    detail_room_level = str(estimated_other_person_state.get("detail_room_level") or "").strip()
    acknowledgement_need_level = str(estimated_other_person_state.get("acknowledgement_need_level") or "").strip()
    pressure_sensitivity_level = str(estimated_other_person_state.get("pressure_sensitivity_level") or "").strip()
    next_step_room_level = str(estimated_other_person_state.get("next_step_room_level") or "").strip()

    if detail_room_level == "low" and pressure_sensitivity_level == "high":
        lines.append("そのため system は、いまは詳しく聞くより、相手に話す余地を残す方がよいと見ています。")
    elif acknowledgement_need_level == "high":
        lines.append("そのため system は、いまは説明を求めるより、まず受け止める方が大事だと見ています。")
    elif next_step_room_level == "high":
        lines.append("そのため system は、いまは無理のない小さい次の一歩なら扱えると見ています。")

    if str(scene.get("scene_family") or "").strip() == "repair_window" and _as_float(relation.get("recent_strain")) >= 0.45:
        lines.append("場面と関係の両方を踏まえて、system は関係の引っかかりを増やさない返し方を優先します。")

    if str(memory.get("conscious_residue_summary") or "").strip() and detail_room_level == "low":
        lines.append("前回からの残りもあるため、system は今すぐ話を広げない方がよいと見ています。")

    if not lines:
        lines.append("system は、場面・関係・記憶を合わせて、いま無理のない関わり方を選ぼうとしています。")

    return tuple(dict.fromkeys(line for line in lines if line))


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
