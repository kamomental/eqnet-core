# Typed Integration Gate Export (2026-04-04)

## 目的

`integration_hooks.ResponseGateResult` の内部では typed contract を保ちつつ、
外へ出す `to_dict()` だけを plain dict export 境界にする。

対象:

- `interaction_policy_packet`
- `action_posture`
- `actuation_plan`

## 変更点

- [integration_hooks.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/integration_hooks.py)
  の `ResponseGateResult` で
  - `conscious_access`
  - `expression_hints`
  を `Mapping` / `MutableMapping` 前提に寄せた
- `ResponseGateResult.to_dict()` は shallow な `dict(...)` ではなく、
  `to_dict()` を持つ contract を再帰的に plain dict へ落とす export helper を使う
- 内部の `gate.expression_hints["interaction_policy_packet"]` などは、
  引き続き contract のまま保持する

## 意味

これで `integration_hooks` は、

- 内部: typed contract
- export: plain dict

という境界を明示できる。

`state-first` を維持したまま、既存の payload 互換も壊さない。

## 回帰

- `tests/test_inner_os_integration_hooks.py`
  - gate 内部では contract instance を保持すること
  - `gate.to_dict()` では plain dict に直列化されること
- 実行:
  - `pytest tests\test_inner_os_integration_hooks.py tests\test_inner_os_bootstrap.py tests\test_runtime_process_turn_hooks.py -q`
  - `153 passed, 1 warning`
