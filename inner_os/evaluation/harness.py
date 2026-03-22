from __future__ import annotations

from typing import Any


def smoke_trace(**payload: Any) -> dict[str, Any]:
    return dict(payload)
