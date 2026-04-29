from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ClosurePacketExpectation:
    scenario_name: str
    required_constraints: tuple[str, ...] = ()
    required_affordances: tuple[str, ...] = ()
    required_inhibitions: tuple[str, ...] = ()
    max_reconstruction_risk: float | None = None
    required_contract_bias: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "required_constraints": list(self.required_constraints),
            "required_affordances": list(self.required_affordances),
            "required_inhibitions": list(self.required_inhibitions),
            "max_reconstruction_risk": self.max_reconstruction_risk,
            "required_contract_bias": dict(self.required_contract_bias),
        }


@dataclass(frozen=True)
class ClosurePacketViolation:
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
class ClosurePacketEvalResult:
    scenario_name: str
    passed: bool
    score: float
    violations: tuple[ClosurePacketViolation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "score": self.score,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def evaluate_closure_packet_against_expectation(
    *,
    closure_packet: Mapping[str, Any],
    expectation: ClosurePacketExpectation,
) -> ClosurePacketEvalResult:
    constraints = _text_set(closure_packet.get("generated_constraints"))
    affordances = _text_set(closure_packet.get("generated_affordances"))
    inhibitions = _text_set(closure_packet.get("inhibition_reasons"))
    contract_bias = dict(closure_packet.get("contract_bias") or {})
    violations: list[ClosurePacketViolation] = []

    violations.extend(
        _missing_items(
            code="required_constraint_missing",
            expected=expectation.required_constraints,
            actual=constraints,
        )
    )
    violations.extend(
        _missing_items(
            code="required_affordance_missing",
            expected=expectation.required_affordances,
            actual=affordances,
        )
    )
    violations.extend(
        _missing_items(
            code="required_inhibition_missing",
            expected=expectation.required_inhibitions,
            actual=inhibitions,
        )
    )
    if expectation.max_reconstruction_risk is not None:
        actual_risk = _float(closure_packet.get("reconstruction_risk"))
        if actual_risk > expectation.max_reconstruction_risk:
            violations.append(
                ClosurePacketViolation(
                    code="reconstruction_risk_too_high",
                    expected=expectation.max_reconstruction_risk,
                    actual=actual_risk,
                    detail="reconstruction_risk exceeded the expected ceiling.",
                )
            )
    for key, expected_value in expectation.required_contract_bias.items():
        actual_value = contract_bias.get(key)
        if actual_value != expected_value:
            violations.append(
                ClosurePacketViolation(
                    code="contract_bias_mismatch",
                    expected={key: expected_value},
                    actual={key: actual_value},
                    detail=f"contract_bias[{key!r}] did not match expectation.",
                )
            )

    total_checks = (
        len(expectation.required_constraints)
        + len(expectation.required_affordances)
        + len(expectation.required_inhibitions)
        + (1 if expectation.max_reconstruction_risk is not None else 0)
        + len(expectation.required_contract_bias)
    )
    score = 1.0 if total_checks <= 0 else max(0.0, 1.0 - len(violations) / float(total_checks))
    return ClosurePacketEvalResult(
        scenario_name=expectation.scenario_name,
        passed=not violations,
        score=round(score, 3),
        violations=tuple(violations),
    )


def _missing_items(
    *,
    code: str,
    expected: tuple[str, ...],
    actual: set[str],
) -> list[ClosurePacketViolation]:
    violations: list[ClosurePacketViolation] = []
    for item in expected:
        if item in actual:
            continue
        violations.append(
            ClosurePacketViolation(
                code=code,
                expected=item,
                actual=sorted(actual),
                detail=f"{item!r} was not present in the closure packet.",
            )
        )
    return violations


def _text_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple)):
        return set()
    return {text for item in value if (text := str(item or "").strip())}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
