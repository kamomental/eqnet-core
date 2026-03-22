from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

KERNEL_UPDATE_ORDER: Tuple[str, ...] = (
    "observe",
    "estimate",
    "memory",
    "qualia",
    "conscious",
    "protect",
    "act",
    "plant",
)


@dataclass(frozen=True)
class KernelStepContract:
    update_order: Tuple[str, ...] = KERNEL_UPDATE_ORDER
    fail_on_observation_contract_break: bool = True
    fail_on_estimator_overconfidence: bool = True
    fail_on_low_observability: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
