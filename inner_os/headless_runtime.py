from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class HeadlessTurnResult:
    execution_mode: str
    primary_action: str
    action_queue: list[str] = field(default_factory=list)
    reply_permission: str = "speak"
    wait_before_action: str = "brief"
    repair_window_commitment: str = "soft"
    outcome_goal: str = ""
    boundary_mode: str = ""
    attention_target: str = ""
    memory_write_priority: str = ""
    do_not_cross: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "primary_action": self.primary_action,
            "action_queue": list(self.action_queue),
            "reply_permission": self.reply_permission,
            "wait_before_action": self.wait_before_action,
            "repair_window_commitment": self.repair_window_commitment,
            "outcome_goal": self.outcome_goal,
            "boundary_mode": self.boundary_mode,
            "attention_target": self.attention_target,
            "memory_write_priority": self.memory_write_priority,
            "do_not_cross": list(self.do_not_cross),
        }


class HeadlessInnerOSRuntime:
    """LLM を使わずに actuation plan を実行計画へ写す最小ランタイム。"""

    def step(self, *, actuation_plan: Mapping[str, Any] | None) -> HeadlessTurnResult:
        plan = dict(actuation_plan or {})
        return HeadlessTurnResult(
            execution_mode=str(plan.get("execution_mode") or "attuned_contact"),
            primary_action=str(plan.get("primary_action") or "hold_presence"),
            action_queue=[str(item) for item in plan.get("action_queue") or [] if str(item).strip()],
            reply_permission=str(plan.get("reply_permission") or "speak"),
            wait_before_action=str(plan.get("wait_before_action") or "brief"),
            repair_window_commitment=str(plan.get("repair_window_commitment") or "soft"),
            outcome_goal=str(plan.get("outcome_goal") or ""),
            boundary_mode=str(plan.get("boundary_mode") or ""),
            attention_target=str(plan.get("attention_target") or ""),
            memory_write_priority=str(plan.get("memory_write_priority") or ""),
            do_not_cross=[str(item) for item in plan.get("do_not_cross") or [] if str(item).strip()],
        )
