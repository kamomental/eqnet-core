# Typed Surface Runtime Bridge (2026-04-04)

## 目的

`surface_context_packet` を planner で typed contract 化したあとも、
runtime 側で再び巨大 `dict` に戻してしまう経路を減らす。

同時に、repair 文脈で `light_playful` や `light_bounce` が混ざっただけで
`bright_bounce` に吸われる regression を防ぐ。

## 今回の変更

### 1. `surface_context_packet` に repair / strategy 情報を正式に載せる

`build_surface_context_packet(...)` に次を追加した。

- `interaction_policy_packet`
- `action_posture`
- `actuation_plan`

これにより `source_state` から次が読める。

- `interaction_policy_strategy`
- `interaction_policy_opening_move`
- `interaction_policy_followup_move`
- `interaction_policy_closing_move`
- `scene_family`
- `action_posture_mode`
- `actuation_execution_mode`
- `actuation_primary_action`
- `actuation_response_channel`

### 2. `derive_discourse_shape(...)` の bright 判定を tightening

repair / guarded 文脈では、単に

- `voice_texture = light_playful`
- `content_sequence` に `light_bounce`

があるだけでは `bright_bounce` にしない。

抑制条件は少なくとも次。

- `interaction_policy_strategy in {"repair_then_attune", "respectful_wait"}`
- `scene_family == "repair_window"`
- `interaction_policy_opening_move in {"name_overreach_and_reduce_force", "reduce_force_and_secure_boundary"}`
- `actuation_primary_action == "soft_repair"`

ただし、次の明示的 bright 信号は別扱いで維持する。

- `turn_delta.kind == "bright_continuity"`
- `conversation_phase == "bright_continuity"`
- `reason_driven_bright`

つまり、

- repair 文脈の accidental bright

は抑え、

- 本当に共有された small bright moment

は残す。

### 3. runtime の内部保持を typed に寄せる

`runtime.py` では、外向きの `controls_used` は dict のまま残しつつ、
内部保持は次を typed contract に寄せた。

- `self._last_surface_context_packet`
- `current_state["surface_context_packet"]`

ここで使うのは `coerce_surface_context_packet(...)`。

## 意味

今回の変更で、

- planner で作った state を runtime が再び平坦化しすぎる問題
- repair 文脈が surface fallback で bright 化する問題

の 2 点を同時に少し締めた。

これは wording hack ではなく、
`surface_context_packet` が読む state を正規化した結果としての修正である。

## 確認

- `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py -q`
  - `10 passed, 1 warning`
- `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py tests\\test_inner_os_surface_context_packet_reasoning.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_runtime_route_prompts.py -q`
  - `159 passed, 1 warning`
- `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
  - `149 passed, 1 warning`

warning は既存の `python_multipart` のみ。

## 次

次の本命は、`surface_context_packet` だけでなく
runtime 内部の `current_state / _last_gate_context` に残る
他の flat payload も、同じく typed boundary 前提で薄くすること。
