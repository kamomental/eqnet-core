from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ConversationContractExpectation:
    scenario_name: str
    stance: str | None = None
    scale: str | None = None
    response_channel: str | None = None
    question_budget: int | None = None
    interpretation_budget: str | None = None
    continuity_mode: str | None = None
    timing_mode: str | None = None
    distance_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "stance": self.stance,
            "scale": self.scale,
            "response_channel": self.response_channel,
            "question_budget": self.question_budget,
            "interpretation_budget": self.interpretation_budget,
            "continuity_mode": self.continuity_mode,
            "timing_mode": self.timing_mode,
            "distance_mode": self.distance_mode,
        }


@dataclass(frozen=True)
class ConversationContractViolation:
    code: str
    expected: Any
    actual: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConversationContractEvalResult:
    scenario_name: str
    passed: bool
    score: float
    violations: tuple[ConversationContractViolation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "score": self.score,
            "violations": [violation.to_dict() for violation in self.violations],
        }


CORE_QUICKSTART_EXPECTATIONS: dict[str, ConversationContractExpectation] = {
    "small_shared_moment": ConversationContractExpectation(
        scenario_name="small_shared_moment",
        stance="join",
        scale="small",
        response_channel="speak",
        question_budget=0,
        interpretation_budget="none",
        continuity_mode="continue",
        distance_mode="near",
    ),
    "guarded_uncertainty": ConversationContractExpectation(
        scenario_name="guarded_uncertainty",
        stance="hold",
        response_channel="hold",
        question_budget=0,
        continuity_mode="reopen",
        timing_mode="held_open",
        distance_mode="guarded",
    ),
}


def evaluate_reaction_contract_against_expectation(
    *,
    reaction_contract: Mapping[str, Any],
    expectation: ConversationContractExpectation,
) -> ConversationContractEvalResult:
    checks: tuple[tuple[str, str, Any], ...] = (
        ("stance", "stance_mismatch", expectation.stance),
        ("scale", "scale_mismatch", expectation.scale),
        ("response_channel", "response_channel_mismatch", expectation.response_channel),
        ("question_budget", "question_budget_violation", expectation.question_budget),
        (
            "interpretation_budget",
            "interpretation_budget_violation",
            expectation.interpretation_budget,
        ),
        ("continuity_mode", "continuity_mode_mismatch", expectation.continuity_mode),
        ("timing_mode", "timing_mode_mismatch", expectation.timing_mode),
        ("distance_mode", "distance_mode_mismatch", expectation.distance_mode),
    )
    violations: list[ConversationContractViolation] = []
    for field_name, code, expected in checks:
        if expected is None:
            continue
        actual = reaction_contract.get(field_name)
        if actual == expected:
            continue
        violations.append(
            ConversationContractViolation(
                code=code,
                expected=expected,
                actual=actual,
                detail=f"{field_name} は {expected!r} を期待したが {actual!r} だった。",
            )
        )

    total_checks = sum(1 for _, _, expected in checks if expected is not None)
    if total_checks <= 0:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (len(violations) / float(total_checks)))
    return ConversationContractEvalResult(
        scenario_name=expectation.scenario_name,
        passed=not violations,
        score=round(score, 3),
        violations=tuple(violations),
    )
