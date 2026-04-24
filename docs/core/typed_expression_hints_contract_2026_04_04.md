# Typed Expression Hints Contract (2026-04-04)

## 目的

`expression_hints` は runtime / integration / bridge の各層をまたいで使われる一方で、
実体は巨大な可変 `dict` のままでした。これが packet-first への再回収点になっていたため、
flat key 互換を残したまま、内部正本だけを typed mutable contract に寄せました。

## 今回の変更

- `inner_os/expression_hints_contract.py` を追加
  - `ExpressionHintsContract`
  - `coerce_expression_hints_contract(...)`
- `inner_os/expression/hint_bridge.py`
  - `build_expression_hints_from_gate_result(...)`
  - `ensure_qualia_planner_view(...)`
  を `ExpressionHintsContract` 返しに変更
- `inner_os/integration_hooks.py`
  - `_expression_hints(...)` は contract を返す
  - `ResponseGateResult.expression_hints` は contract を内部正本にする
  - `ResponseGateResult.to_dict()` は export 境界で plain dict に落とす

## ねらい

- flat key 群との後方互換は壊さない
- `gate.expression_hints["..."]` や `.update(...)` はそのまま使える
- export 直前だけ `to_dict()` で plain dict に落とす

## まだ残るもの

- key 自体はまだ flat で多い
- `expression_hints` の中身はまだ scene / qualia / audit などが同居している
- 次は `expression_hints` の大項目を小さい sub-contract に分けていく必要がある

## 確認

- `pytest tests\test_inner_os_integration_hooks.py tests\test_inner_os_bootstrap.py tests\test_runtime_process_turn_hooks.py -q`
  - `154 passed, 1 warning`
