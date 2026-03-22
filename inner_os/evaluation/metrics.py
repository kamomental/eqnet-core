from __future__ import annotations

from typing import Mapping, Any

from .contracts import EvalReport


def evaluate_run(trace: Mapping[str, Any]) -> EvalReport:
    findings = []
    if not trace.get("foreground"):
        findings.append("foreground_missing")
    if trace.get("raw_observation_passed_to_llm"):
        findings.append("llm_raw_observation_violation")
    score = 0.0 if findings else 1.0
    return EvalReport(score=score, findings=findings, uncertainty=0.2 if not findings else 0.7)
