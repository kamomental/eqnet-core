from __future__ import annotations

from collections.abc import Iterator, MutableMapping as MutableMappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping


def _export_expression_hint_value(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _export_expression_hint_value(to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _export_expression_hint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_export_expression_hint_value(item) for item in value]
    if isinstance(value, tuple):
        return [_export_expression_hint_value(item) for item in value]
    return value


@dataclass
class ExpressionHintsContract(MutableMappingABC[str, object]):
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            str(key): _export_expression_hint_value(value)
            for key, value in self.payload.items()
        }

    def copy(self) -> "ExpressionHintsContract":
        return ExpressionHintsContract(dict(self.payload))

    def __getitem__(self, key: str) -> object:
        return self.payload[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.payload[str(key)] = value

    def __delitem__(self, key: str) -> None:
        del self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


def coerce_expression_hints_contract(
    value: Mapping[str, Any] | ExpressionHintsContract | None,
) -> ExpressionHintsContract:
    if isinstance(value, ExpressionHintsContract):
        return value
    return ExpressionHintsContract(
        {
            str(key): item
            for key, item in dict(value or {}).items()
        }
    )
