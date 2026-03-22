from __future__ import annotations

from typing import Protocol

from .models import ResponsePlan


class ExpressionAdapter(Protocol):
    def render(self, response_plan: ResponsePlan) -> str:
        """Foreground から表出を生成する adapter。"""
