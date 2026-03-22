from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class InteractionJudgementSummary:
    observed_lines: tuple[str, ...] = ()
    inferred_lines: tuple[str, ...] = ()
    selected_object_lines: tuple[str, ...] = ()
    deferred_object_lines: tuple[str, ...] = ()
    operation_lines: tuple[str, ...] = ()
    intended_effect_lines: tuple[str, ...] = ()
    compact_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_interaction_judgement_summary(
    *,
    interaction_judgement_view: Mapping[str, Any] | None = None,
    conversational_objects: Mapping[str, Any] | None = None,
    object_operations: Mapping[str, Any] | None = None,
    interaction_effects: Mapping[str, Any] | None = None,
) -> InteractionJudgementSummary:
    judgement_view = dict(interaction_judgement_view or {})
    conversational_objects = dict(conversational_objects or {})
    object_operations = dict(object_operations or {})
    interaction_effects = dict(interaction_effects or {})

    observed_lines: list[str] = []
    inferred_lines: list[str] = []
    selected_object_lines: list[str] = []
    deferred_object_lines: list[str] = []
    operation_lines: list[str] = []
    intended_effect_lines: list[str] = []

    for item in judgement_view.get("observed_signals") or []:
        if not isinstance(item, Mapping):
            continue
        line = _format_observed_signal(item)
        if line:
            observed_lines.append(line)

    for item in judgement_view.get("inferred_signals") or []:
        if not isinstance(item, Mapping):
            continue
        line = _format_inferred_signal(item)
        if line:
            inferred_lines.append(line)

    for label in judgement_view.get("selected_object_labels") or []:
        clean_label = str(label or "").strip()
        if clean_label:
            selected_object_lines.append(
                f"今回 system が扱う対象は「{clean_label}」です。"
            )

    for label in judgement_view.get("deferred_object_labels") or []:
        clean_label = str(label or "").strip()
        if clean_label:
            deferred_object_lines.append(
                f"今回 system は「{clean_label}」にはまだ深く触れません。"
            )

    operation_items = [
        dict(item)
        for item in object_operations.get("operations") or []
        if isinstance(item, Mapping)
    ]
    for item in operation_items:
        line = _format_operation_line(item)
        if line:
            operation_lines.append(line)

    effect_items = [
        dict(item)
        for item in interaction_effects.get("effects") or []
        if isinstance(item, Mapping)
    ]
    for item in effect_items:
        line = _format_effect_line(item)
        if line:
            intended_effect_lines.append(line)

    if not selected_object_lines:
        for item in conversational_objects.get("objects") or []:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or "").strip()
            if label:
                selected_object_lines.append(
                    f"今回 system が扱う対象は「{label}」です。"
                )
                break

    compact_lines = tuple(
        line
        for group in (
            tuple(observed_lines),
            tuple(inferred_lines),
            tuple(selected_object_lines),
            tuple(deferred_object_lines),
            tuple(operation_lines),
            tuple(intended_effect_lines),
        )
        for line in group
    )

    return InteractionJudgementSummary(
        observed_lines=tuple(observed_lines),
        inferred_lines=tuple(inferred_lines),
        selected_object_lines=tuple(selected_object_lines),
        deferred_object_lines=tuple(deferred_object_lines),
        operation_lines=tuple(operation_lines),
        intended_effect_lines=tuple(intended_effect_lines),
        compact_lines=compact_lines,
    )


def _format_observed_signal(item: Mapping[str, Any]) -> str:
    signal_kind = str(item.get("signal_kind") or "").strip()
    text = str(item.get("text") or "").strip()
    if signal_kind == "user_text" and text:
        return f"相手は「{text}」と言いました。"
    if signal_kind == "reportable_fact" and text:
        return f"system は「{text}」を、相手が実際に出した事実として受け取りました。"
    if text:
        return f"system は「{text}」を、今回の観測内容として受け取りました。"
    return ""


def _format_inferred_signal(item: Mapping[str, Any]) -> str:
    signal_kind = str(item.get("signal_kind") or "").strip()
    strength = _clamp01(float(item.get("strength", 0.0) or 0.0))
    score_suffix = f"（強さ {strength:.2f}）"
    if signal_kind == "detail_room":
        return (
            "system は、相手は今詳しく話す余裕が少ないかもしれないと見ています。"
            f"{score_suffix}"
        )
    if signal_kind == "pressure_sensitivity":
        return (
            "system は、今ここで詳しく聞くと相手の負担が増えるかもしれないと見ています。"
            f"{score_suffix}"
        )
    if signal_kind == "acknowledgement_need":
        return (
            "system は、相手はまず受け止めてもらう必要が高いかもしれないと見ています。"
            f"{score_suffix}"
        )
    if signal_kind == "question_pressure":
        return (
            "system は、このターンでは質問を減らす方向が強いと見ています。"
            f"{score_suffix}"
        )
    if signal_kind == "defer_dominance":
        return (
            "system は、このターンでは話を広げるより保留する方向が強いと見ています。"
            f"{score_suffix}"
        )
    statement = str(item.get("statement") or "").strip()
    if statement:
        return f"system の推測: {statement} {score_suffix}"
    return ""


def _format_operation_line(item: Mapping[str, Any]) -> str:
    operation_kind = str(item.get("operation_kind") or "").strip()
    target_label = str(item.get("target_label") or "").strip() or "今回の対象"
    strength = _clamp01(float(item.get("operation_strength", 0.0) or 0.0))
    burden_risk = _clamp01(float(item.get("burden_risk", 0.0) or 0.0))
    suffix = f"（強さ {strength:.2f} / 相手の負担見積り {burden_risk:.2f}）"
    if operation_kind == "hold_without_probe":
        return (
            f"system は「{target_label}」について、相手に詳しい説明を求めずに受け止めます。"
            f"{suffix}"
        )
    if operation_kind == "acknowledge":
        return f"system は「{target_label}」について、まず受け止めます。{suffix}"
    if operation_kind == "narrow_clarify":
        return (
            f"system は「{target_label}」について、範囲を狭めて一つだけ確かめます。"
            f"{suffix}"
        )
    if operation_kind == "offer_small_next_step":
        return (
            f"system は「{target_label}」について、負担が増えない小さい次の一歩として扱います。"
            f"{suffix}"
        )
    if operation_kind == "defer_detail":
        return f"system は「{target_label}」を、いまは詳しく触れずに保留します。{suffix}"
    if operation_kind == "keep_return_point":
        return (
            f"system は「{target_label}」について、あとで戻れるように開いたままにします。"
            f"{suffix}"
        )
    if operation_kind:
        return f"system は「{target_label}」に対して {operation_kind} を行います。{suffix}"
    return ""


def _format_effect_line(item: Mapping[str, Any]) -> str:
    effect_kind = str(item.get("effect_kind") or "").strip()
    intensity = _clamp01(float(item.get("intensity", 0.0) or 0.0))
    suffix = f"（強さ {intensity:.2f}）"
    if effect_kind == "feel_received":
        return (
            "system は、相手に「そのまま受け止められた」と感じてほしいと考えています。"
            f"{suffix}"
        )
    if effect_kind == "preserve_self_pacing":
        return (
            "system は、相手が自分のペースで話すかどうかを選べる状態を保ちたいと考えています。"
            f"{suffix}"
        )
    if effect_kind == "keep_connection_open":
        return (
            "system は、相手があとでこの話に戻れる余地を残したいと考えています。"
            f"{suffix}"
        )
    if effect_kind == "enable_small_next_step":
        return (
            "system は、相手が無理なく次の小さい一歩を見つけられる状態をつくりたいと考えています。"
            f"{suffix}"
        )
    if effect_kind == "protect_boundary":
        return (
            "system は、相手がまだ触れたくない領域に話が広がらないようにしたいと考えています。"
            f"{suffix}"
        )
    if effect_kind == "reduce_pressure":
        return (
            "system は、相手が急かされていると感じにくい状態をつくりたいと考えています。"
            f"{suffix}"
        )
    if effect_kind:
        return f"system は、相手に {effect_kind} が起きやすい状態を目指しています。{suffix}"
    return ""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
