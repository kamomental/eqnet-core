# Typed Timing Contract (2026-04-03)

## 目的

`response_channel` と turn timing を `dict` の寄せ集めではなく、型付き contract として扱う。

今回の対象は次の 3 つです。

- `TurnTimingHint`
- `EmitTimingContract`
- `TimingGuardState`

## 背景

これまで timing 系は、

- headless actuation
- runtime
- response meta
- probe
- tests

の各所で `dict[str, Any]` として横流しされていました。

この形だと、

- どの key が正本か分かりにくい
- runtime 内で `dict(...)` の再構成が増える
- 末端の if と整形処理が肥大化する
- fixture と本番経路の差で壊れやすい

という問題がありました。

## 今回の変更

### 1. headless runtime の contract を型付き化

`inner_os/headless_runtime.py` に次を追加しました。

- `TurnTimingHint`
- `EmitTimingContract`
- `TimingGuardState`
- `coerce_turn_timing_hint(...)`
- `coerce_emit_timing_contract(...)`
- `coerce_timing_guard_state(...)`

`HeadlessTurnResult.turn_timing_hint` も typed contract を正本にしています。

### 2. runtime の内部保持を typed contract に変更

`emot_terrain_lab/hub/runtime.py` では、

- `_last_inner_os_timing_guard`
- `_apply_inner_os_emit_timing(...)`
- `_serialize_response_meta(...)`

が typed contract を使うようになりました。

外部へ出すときだけ `to_dict()` で落としています。

### 3. fixture 互換は coercion で吸収

test や既存 call site が `dict` を渡しても壊れないように、

- `HeadlessTurnResult.to_dict()`
- `process_turn(...)` 内の headless result 正規化

で coercion を入れています。

つまり、

- 内部の正本は typed
- 入出力境界では dict 互換を維持

という形です。

## 設計上の意味

これは if を増やす変更ではありません。

やっていることは、

- 判断を branch の追加で持つ

のではなく、

- timing の意味単位を contract に昇格する

ことです。

今後は同じやり方で、

- `policy_packet`
- `action_posture`
- `actuation_plan`

の巨大 `dict` 経路も縮めていくのが本筋です。

## 確認

- `py_compile`
  - `inner_os/headless_runtime.py`
  - `inner_os/__init__.py`
  - `emot_terrain_lab/hub/runtime.py`
  - `tests/test_inner_os_headless_runtime.py`
  - `tests/test_runtime_process_turn_hooks.py`
- `pytest tests/test_inner_os_headless_runtime.py tests/test_runtime_process_turn_hooks.py tests/test_lmstudio_pipeline_probe.py tests/test_lmstudio_pipeline_probe_controls.py -q`
  - `85 passed, 1 warning`

warning は既存の `python_multipart` のみです。
