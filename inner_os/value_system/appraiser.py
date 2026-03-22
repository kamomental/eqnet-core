from __future__ import annotations

from .models import ValueState


def summarize_value_axes(value_state: ValueState) -> list[str]:
    return [
        f"{name}:{value:.2f}"
        for name, value in sorted(value_state.value_axes.items())
    ]
