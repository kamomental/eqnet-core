# Typed Runtime Guidance Bridge (2026-04-04)

## 目的

`response_planner` 側で導入した typed contract を、`runtime.py` の guidance / qualia-gate / live loop 側でも正本として扱う。

今回の対象は次の 4 本。

- `InteractionPolicyPacketContract`
- `ActionPostureContract`
- `ActuationPlanContract`
- `SurfaceContextPacket`

## 何を変えたか

### 1. live loop の persistent hints を contract 起点にした

`runtime._run_inner_os_live_response_loop(...)` の `persistent_planning_hints` で、

- `interaction_policy_packet`
- `action_posture`
- `actuation_plan`
- `surface_context_packet`

を raw `dict` ではなく coercion helper で保持するようにした。

外へ出すときだけ `to_dict()` を通す。

### 2. bright override / persistent stamp でも contract を保つ

`_bright_guidance_override(...)` と `_refresh_persistent_planning_hints(...)` で、
`surface_context_packet` は `SurfaceContextPacket` として持ち回し、
`controls_used` に同期するときだけ dict 化するようにした。

`discourse_shape` は現状 helper 群が `Mapping.get(...)` 前提なので、
内部保持は dict serialization を維持している。

### 3. qualia gate の current guidance を contract 起点にした

`_apply_qualia_gate(...)` では、

- `inner_os_discourse_shape`
- `inner_os_surface_context_packet`

を coercion で読み直し、
current guidance を優先して minimal response を組み立てる経路を維持した。

### 4. `_build_inner_os_llm_guidance(...)` の内側を contract 起点にした

`_build_inner_os_llm_guidance(...)` の内部では、

- `interaction_policy_packet`
- `surface_context_packet`
- `action_posture`
- `actuation_plan`

を contract として読み、
内部の enrich は contract から dict を起こして再度 `SurfaceContextPacket` に戻す形へ寄せた。

返り値は従来互換のため dict のまま維持している。

## 注意点

`SurfaceContextPacket` は Mapping 互換だが、空 coercion しても key 自体は存在する。
そのため `if not packet:` ではなく、

- `conversation_phase`
- `shared_core`
- `response_role`
- `constraints`
- `surface_profile`
- `source_state`

のどれかが実際に埋まっているかで builder 実行を判断する必要がある。

## 回帰確認

- `py_compile`
  - `emot_terrain_lab/hub/runtime.py`
  - `tests/test_runtime_process_turn_hooks.py`
- `pytest tests\test_runtime_process_turn_hooks.py tests\test_inner_os_surface_context_packet.py tests\test_inner_os_discourse_shape.py tests\test_human_output_examples.py -q`
  - `95 passed, 1 warning`
- `pytest tests\test_runtime_process_turn_hooks.py tests\test_inner_os_integration_hooks.py tests\test_inner_os_surface_context_packet_reasoning.py tests\test_runtime_route_prompts.py -q`
  - `150 passed, 1 warning`
- `pytest tests\test_inner_os_bootstrap.py tests\test_llm_hub_reason_chain.py -q`
  - `13 passed, 1 warning`

warning は既存の `python_multipart` のみ。

## 次

次の圧縮対象は `runtime.py` と `integration_hooks.py` に残る、

- metrics 用の defensive `dict(...)`
- payload export の前段での再回収
- `surface_context_packet` を受ける helper 群の dict 前提

を薄くしていくこと。

## 追加メモ

- `_apply_inner_os_surface_profile(...)` でも `surface_context_packet` を contract 起点にした
- `_compact_inner_os_sequence_text(...)` でも bright cue 判定時の packet 参照を contract 起点にした
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_human_output_examples.py tests\\test_lmstudio_pipeline_probe.py -q`
    - `83 passed, 1 warning`
