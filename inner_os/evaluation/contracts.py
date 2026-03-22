from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalReport:
    score: float
    findings: list[str] = field(default_factory=list)
    uncertainty: float = 1.0
