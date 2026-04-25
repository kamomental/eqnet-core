from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

try:
    from .surface_policy import compile_surface_policy
except ImportError:  # pragma: no cover - supports script-level dynamic loading.
    from inner_os.expression.surface_policy import compile_surface_policy


@dataclass(frozen=True)
class LLMExpressionBridgePolicy:
    """LLM を表出器として使うための境界設定。"""

    language: str = "ja"
    max_state_fields: int = 32
    include_raw_observation: bool = False
    system_identity: str = (
        "あなたは EQNet の表出層です。内部状態を推定し直さず、"
        "渡された状態契約に従って自然な日本語の一言だけを生成します。"
    )


@dataclass(frozen=True)
class LLMExpressionRequest:
    """状態主導の LLM 呼び出しリクエスト。"""

    should_call_llm: bool
    action_channel: str
    system_prompt: str
    user_prompt: str
    contract: dict[str, Any]
    surface_policy: dict[str, Any]
    state_summary: dict[str, Any]
    blocked_reason: str = ""
    fallback_action: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_call_llm": self.should_call_llm,
            "action_channel": self.action_channel,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "contract": dict(self.contract),
            "surface_policy": dict(self.surface_policy),
            "state_summary": dict(self.state_summary),
            "blocked_reason": self.blocked_reason,
            "fallback_action": dict(self.fallback_action),
        }


def build_llm_expression_request(
    *,
    input_text: str,
    reaction_contract: Mapping[str, Any],
    joint_state: Mapping[str, Any] | None = None,
    shared_presence: Mapping[str, Any] | None = None,
    subjective_scene: Mapping[str, Any] | None = None,
    self_other_attribution: Mapping[str, Any] | None = None,
    audit_projection: Mapping[str, Any] | None = None,
    policy: LLMExpressionBridgePolicy | None = None,
) -> LLMExpressionRequest:
    """LLM へ渡す最小 foreground を、状態契約から構成する。"""

    active_policy = policy or LLMExpressionBridgePolicy()
    contract = _compact_contract(reaction_contract)
    surface_policy = compile_surface_policy(contract).to_dict()
    channel = str(contract.get("response_channel") or "speak")
    state_summary = _build_state_summary(
        joint_state=joint_state,
        shared_presence=shared_presence,
        subjective_scene=subjective_scene,
        self_other_attribution=self_other_attribution,
        audit_projection=audit_projection,
        max_fields=active_policy.max_state_fields,
    )
    fallback_action = _build_fallback_action(contract)

    if channel != "speak":
        return LLMExpressionRequest(
            should_call_llm=False,
            action_channel=channel,
            system_prompt="",
            user_prompt="",
            contract=contract,
            surface_policy=surface_policy,
            state_summary=state_summary,
            blocked_reason="reaction_contract.response_channel is not speak",
            fallback_action=fallback_action,
        )

    system_prompt = active_policy.system_identity
    user_prompt = _render_user_prompt(
        input_text=input_text,
        contract=contract,
        surface_policy=surface_policy,
        state_summary=state_summary,
        include_raw_observation=active_policy.include_raw_observation,
    )
    return LLMExpressionRequest(
        should_call_llm=True,
        action_channel=channel,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        contract=contract,
        surface_policy=surface_policy,
        state_summary=state_summary,
        fallback_action=fallback_action,
    )


def _compact_contract(reaction_contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stance",
        "scale",
        "initiative",
        "question_budget",
        "interpretation_budget",
        "response_channel",
        "timing_mode",
        "continuity_mode",
        "distance_mode",
        "closure_mode",
        "shape_id",
        "strategy",
        "execution_mode",
    )
    return {
        key: reaction_contract[key]
        for key in keys
        if key in reaction_contract and reaction_contract[key] not in {None, ""}
    }


def _build_state_summary(
    *,
    joint_state: Mapping[str, Any] | None,
    shared_presence: Mapping[str, Any] | None,
    subjective_scene: Mapping[str, Any] | None,
    self_other_attribution: Mapping[str, Any] | None,
    audit_projection: Mapping[str, Any] | None,
    max_fields: int,
) -> dict[str, Any]:
    candidates: list[tuple[str, Any]] = []
    candidates.extend(_audit_projection_items(audit_projection))
    candidates.extend(_prefix_items("joint", joint_state))
    candidates.extend(_prefix_items("shared_presence", shared_presence))
    candidates.extend(_prefix_items("subjective_scene", subjective_scene))
    candidates.extend(_prefix_items("self_other", self_other_attribution))
    compact: dict[str, Any] = {}
    for key, value in candidates:
        if len(compact) >= max_fields:
            break
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            compact[key] = value
    return compact


def _audit_projection_items(
    audit_projection: Mapping[str, Any] | None,
) -> list[tuple[str, Any]]:
    if not isinstance(audit_projection, Mapping):
        return []
    items: list[tuple[str, Any]] = []
    audit_axes = audit_projection.get("audit_axes")
    if isinstance(audit_axes, Mapping):
        items.extend(
            _prefix_items(
                "audit",
                {
                    key: value
                    for key, value in audit_axes.items()
                    if _is_expression_safe_audit_key(str(key))
                },
            )
        )
    source_state = audit_projection.get("surface_context_source_state")
    if isinstance(source_state, Mapping):
        items.extend(_prefix_items("surface_source", source_state))
    return items


def _prefix_items(
    prefix: str,
    mapping: Mapping[str, Any] | None,
) -> list[tuple[str, Any]]:
    if not isinstance(mapping, Mapping):
        return []
    return [(f"{prefix}.{key}", value) for key, value in mapping.items()]


def _is_expression_safe_audit_key(key: str) -> bool:
    hidden_prefixes = (
        "protective_trace_",
    )
    return not key.startswith(hidden_prefixes)


def _render_user_prompt(
    *,
    input_text: str,
    contract: Mapping[str, Any],
    surface_policy: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    include_raw_observation: bool,
) -> str:
    question_budget = int(contract.get("question_budget") or 0)
    interpretation_budget = str(contract.get("interpretation_budget") or "")
    scale = str(contract.get("scale") or "small")
    max_sentences = int(surface_policy.get("max_sentences") or 0)

    constraints = [
        "出力は日本語の自然な会話文だけにする。",
        "内部状態、スコア、JSON、分析文、注釈は出さない。",
        f"反応の大きさは {scale} に保つ。",
        f"最大文数は {max_sentences} 文。",
    ]
    constraints.extend(_surface_policy_constraint_texts(surface_policy))
    if question_budget <= 0:
        constraints.append("質問で終えない。聞き出しに行かない。")
    else:
        constraints.append(f"質問は最大 {question_budget} 個まで。")
    if interpretation_budget in {"none", "low"}:
        constraints.append("相手の気持ちや出来事を断定解釈しない。")

    payload = {
        "user_input": input_text,
        "reaction_contract": dict(contract),
        "surface_policy": dict(surface_policy),
        "state_summary": dict(state_summary),
        "constraints": constraints,
    }
    if not include_raw_observation:
        payload["raw_observation_policy"] = "raw observation は渡さない。状態要約だけを使う。"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _surface_policy_constraint_texts(surface_policy: Mapping[str, Any]) -> list[str]:
    allowed = {
        str(item)
        for item in surface_policy.get("allowed_acts") or []
        if str(item).strip()
    }
    prohibited = {
        str(item)
        for item in surface_policy.get("prohibited_acts") or []
        if str(item).strip()
    }
    constraints: list[str] = []
    if {"minimal_acknowledgement", "surface_mirror"}.issubset(allowed):
        constraints.append("短い受け止め、または相手の表面表現の最小反映だけにする。")
    if "ask_question" in prohibited:
        constraints.append("質問しない。確認しない。")
    if prohibited.intersection(
        {"infer_hidden_feeling", "explain_meaning", "attribute_motive", "summarize_as_truth"}
    ):
        constraints.append("相手の内面、理由、意味、動機を推測しない。")
    if prohibited.intersection({"offer_advice", "normalize_or_advise"}):
        constraints.append("助言、方針化、正しさの提示をしない。")
    if prohibited.intersection({"bright_reframe", "positive_spin"}):
        constraints.append("明るく言い換えない。前向きに整えない。")
    if "assistant_attractor" in prohibited:
        constraints.append("支援者、相談員、カウンセラーの口調に寄せない。")
    if "continue_conversation" in prohibited:
        constraints.append("会話を広げない。次の話題を促さない。")
    if "close_or_conclude" in prohibited:
        constraints.append("結論づけない。終わらせない。")
    return constraints


def _build_fallback_action(contract: Mapping[str, Any]) -> dict[str, Any]:
    channel = str(contract.get("response_channel") or "speak")
    if channel == "hold":
        return {
            "type": "nonverbal",
            "name": "presence_hold",
            "description": "発話せず、少し間を保って相手の継続余地を残す。",
        }
    return {
        "type": "review_gate",
        "name": "retry_or_minimal_surface",
        "description": "LLM出力が契約違反なら、同じ契約で再生成する。",
    }
