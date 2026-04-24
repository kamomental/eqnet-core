# Typed Surface Context Contract (2026-04-03)

## 目的

`surface_context_packet` は expression bridge の中心にあるが、
実装上は dataclass を作ってもすぐ `.to_dict()` に落として使っていた。

そのため、

- planner の内部では contract が正本にならない
- bridge 境界が再び packet-first に戻る
- `dict[str, Any]` のまま取り回す箇所が増える

という問題が残っていた。

今回はこの packet を、**Mapping 互換の typed contract として保持する** 方向へ一段進めた。

## 変更内容

### 1. `SurfaceContextPacket` を Mapping 互換にした

`inner_os/expression/surface_context_packet.py` の `SurfaceContextPacket` に

- `__getitem__`
- `__iter__`
- `__len__`

を追加し、`Mapping[str, Any]` としてそのまま扱えるようにした。

### 2. coercion を追加した

以下を追加した。

- `coerce_surface_context_packet(...)`

これにより、既存の `dict` fixture や runtime 側の raw packet からも、
必要なら typed contract に戻せる。

### 3. planner 側で即 `.to_dict()` しないようにした

`inner_os/expression/response_planner.py` では、
`build_surface_context_packet(...)` の返り値をいったん contract のまま保持するようにした。

つまり、

- `derive_discourse_shape(...)`
- `ResponsePlan.surface_context_packet` 相当の内部利用

は contract を正本にし、
`llm_payload` へ入れるときだけ `.to_dict()` に落としている。

## 設計意図

今回の意図は、

- `surface_context_packet` をまた一つの巨大 dict として扱うのをやめる
- bridge の手前までは contract を正本にする
- 外向き payload だけ dict にする

ことにある。

これは `timing contract`、`action contract` と同じ方針で、
**外部互換を壊さずに、内部の正本だけを typed 化する** 段階的圧縮である。

## まだ残っている課題

- `policy_packet` は依然として巨大 packet
- `llm_payload` は依然として広い dict
- runtime 側では `surface_context_packet` を defensive に `dict(...)` へ戻す箇所が多い

つまり今回は、
**surface context の生成元と planner 内部の正本だけを typed 化した**
第一段である。

## 確認

- `py_compile`
  - `inner_os/expression/surface_context_packet.py`
  - `inner_os/expression/__init__.py`
  - `inner_os/expression/response_planner.py`
  - `tests/test_inner_os_surface_context_packet.py`
- `pytest tests/test_inner_os_surface_context_packet.py tests/test_inner_os_surface_context_packet_reasoning.py tests/test_inner_os_bootstrap.py tests/test_llm_hub_reason_chain.py -q`
  - `19 passed, 1 warning`

warning は既存の `python_multipart` のみ。
