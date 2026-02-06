from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Protocol


class HubRuntime(Protocol):
    """Runtime delegation contract used by EQNetHub facade."""

    def log_moment(
        self,
        raw_event: Any,
        raw_text: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> None:
        ...

    def run_nightly(
        self,
        date_obj: Optional[date] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> None:
        ...

    def query_state(
        self,
        *,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...
