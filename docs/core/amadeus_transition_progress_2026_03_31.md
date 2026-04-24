# Amadeus Transition Progress (2026-03-31)

## 2026-04-01 bridge reviewer / probe 追記

- `llm_bridge_contract` を phrase match 中心から **sentence-form review** へ拡張
  - `assistant lead`
  - `interpretive prose`
  - `elicitation / counselor move`
  - `too_many_sentences`
  を文単位で判定するようにした
- `small shared moment` 判定に current fallback も使うようにして、
  packet が薄いケースでも `bright_bounce` final から reviewer が文脈を再構成できるようにした
- `lmstudio_pipeline_probe` に post-review を追加
  - `llm_raw_contract_ok=false`
  - `raw == original`
  のとき、current `response_text + discourse_shape` から `effective raw` を再導出する
- live の `qwen3.5-4b` bright continuity でも、
  probe 上は `assistant_attractor_violation / interpretive_bright_violation / question_block_violation / too_many_sentences`
  まで見えるようになった

## 意味

- raw モデル自体の assistant attractor はまだ残っている
- ただし diagnostics は、`uncertainty_meta` だけに見える状態から抜けた
- これで次に進むべき論点が
  - prompt-only でどこまで削れるか
  - validator / contract を runtime 本体にどう昇格させるか
  にかなり絞れた

## 2026-04-01 llm_hub raw attractor 追記
- `llm_hub.py` に `reason_chain_language_guard` を追加
  - `utterance_reason`
  - `joint_state`
  - `memory anchor`
  - `organism_state`
  - `external_field`
  - `terrain_dynamics`
  を raw system prompt の主理由として前景化
- bright continuity では、assistant / counselor / analyst 側へ戻る文体をさらに抑制
  - shared な小出来事を外から解釈しない
  - 既知の anchor を新しい問題として再説明しない
  - `question_policy = none` のとき follow-up question を立てない
- `test_llm_hub_reason_chain.py` と `test_llm_hub_uncertainty.py` で
  - `organism / field / terrain` が policy prompt に載ること
  - raw system prompt が reason-chain を前面化すること
  を固定

## 2026-04-01 posture / actuation 追記

- `joint_state` を `action_posture / actuation_plan` に接続
- `joint_reentry_room` があるときだけ `share_small_shift / shared_progress` に入りやすくした
- `joint_guard_signal` が高いときは `respectful_wait / defer_with_presence` 側へ寄せるようにした

## 2026-04-01 追記

- `joint_state` を `live_engagement_state` に接続
  - `shared_delight / common_ground / joint_attention / mutual_room / coupling_strength`
    が `pickup_comment / riff_with_comment / seed_topic` の room に直接効くようにした
- `joint_state` を `bright_strategy_override / field_strategy_override` に接続
  - `joint_mode` と `joint_reentry_room` があるときだけ
    `shared_world_next_step` や `field_reentry_progression` を選びやすくした
  - 逆に `strained_jointness` と高い `shared_tension` があるときは
    `respectful_wait` 側へ戻りやすくした
- これにより、`shared moment` だけでなく
  `self-other coupling` を含んだ canonical state から
  upstream strategy が決まる段階に入った

## この round で進めたこと

- `organism_state` を `surface_context_packet` に正式に射影
  - `organism_posture`
  - `organism_attunement`
  - `organism_grounding`
  - `organism_protective_tension`
  - `organism_expressive_readiness`
  - `organism_play_window`
  - `organism_relation_pull`
- `discourse_shape` が `organism_state` を読むように変更
  - `play` posture では `bright_bounce` の playfulness / tempo を底上げ
  - `attune` posture では `reflect_hold` を優先
  - `protect / recover / verify` posture では follow-up question を抑制
- `response_planner` と `runtime` の両方で同じ `organism_state` を packet に通すようにし、
  runtime 側だけの補修に依存しない形へ寄せた

## 意味

- `mode だからこの shape` ではなく、
  `organism_state + utterance_reason` から shape を決める方向へ一段進んだ
- `organism_state` はまだ route の主決定因ではないが、
  少なくとも `surface_context_packet -> discourse_shape -> final shaping`
  の経路には入った

## 確認

- `py_compile`
  - green
- `pytest tests\\test_inner_os_surface_context_packet_reasoning.py tests\\test_inner_os_discourse_shape.py tests\\test_runtime_process_turn_hooks.py -q`
  - `73 passed, 1 warning`
- `pytest tests\\test_inner_os_surface_context_packet.py tests\\test_runtime_bright_short_sequence.py -q`
  - `7 passed, 1 warning`

## 次の本命

- `organism_state` を `route / response_strategy / live_engagement` にも効かせる
- `joint_state` を作って self-only の organism から self-other coupling へ広げる
- `memory_dynamics` と `utterance_reason` を route 側の判断にも接続する

## この round で進めたこと

- `external_field_state` を canonical state として追加
  - `environmental_load`
  - `social_pressure`
  - `continuity_pull`
  - `ambiguity_load`
  - `safety_envelope`
  - `novelty`
- `terrain_dynamics` を canonical dynamics として追加
  - `dominant_basin`
  - `dominant_flow`
  - `terrain_energy`
  - `entropy`
  - `ignition_pressure`
  - `barrier_height`
  - `recovery_gradient`
  - `basin_pull`
- `integration_hooks / transfer_package / continuity_summary / runtime / surface_context_packet` まで配線
- `external_field_state` と `terrain_dynamics` の contract test を追加

## 確認

- `py_compile`
  - green
- `pytest tests\\test_inner_os_external_field_state.py tests\\test_inner_os_terrain_dynamics.py tests\\test_inner_os_transfer_package.py tests\\test_inner_os_continuity_summary.py tests\\test_inner_os_surface_context_packet.py tests\\test_inner_os_integration_hooks.py tests\\test_runtime_process_turn_hooks.py -q`
  - `148 passed, 1 warning`

## 2026-04-02 relation field reconstruction summary

- 外部事例の ensemble を整理
  - GraphRAG: static structure / hierarchy
  - HippoRAG: co-activation / traversal
  - REMem: episodic time-aware memory
  - common ground: jointness / shared state
  - feed-forward internal geometry: internal structure だけでは独自性にならない圧力
- `memory_dynamics` の relation field を
  - `appraisal`
  - `meaning_update`
  - `utterance_reason`
  - `joint_state`
  の主語に昇格
- 追加 summary:
  - `docs/core/external_case_ensemble_2026_04_02.md`
  - `docs/core/relation_reason_chain_2026_04_02.md`
  - `docs/core/relation_field_reconstruction_summary_2026_04_02.md`
- 今回の立場:
  - memory は保存物ではなく relation field
  - response は mode ではなく current relation から決まる
  - 次は `terrain_dynamics` と `route / strategy` を relation-aware にする

## 2026-04-02 force-LLM-bridge clarify opening sticky

- `policy_packet` の opening-request cue を拡張し、
  - `どこから話せば`
  - `何から話せば`
  系を `opening_request_hint` で拾えるようにした
- `force_llm_bridge` の clarify 経路では、`offer_small_opening_line / offer_small_opening_frame` を含む sequence を runtime 側で `opening_support` shape に固定
  - これで `bright_bounce` が後段で上書きせず、
    - `切り出すなら…`
    - `まだうまく整理できないなら…`
    を final まで保持できる
- targeted regression:
  - `pytest tests\\test_inner_os_policy_packet_opening_request.py tests\\test_runtime_force_llm_bridge.py -q`
    - `8 passed, 1 warning`
- nearby regression:
  - `pytest tests\\test_runtime_force_llm_bridge.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
    - `135 passed, 1 warning`

## 2026-04-03 terrain relation strategy -> posture -> actuation

- `memory_dynamics` の relation / causal field を受けた `terrain_dynamics` を、
  `response_strategy / route` だけでなく
  `action_posture / actuation_plan` にまで接続
- `cross_context_bridge / reframing_cause / name_distant_link`
  が continuity / recovery terrain 上で立つと、
  shared moment がなくても
  - `engagement_mode = co_move`
  - `execution_mode = shared_progression`
  に上がる
- `unfinished_link / unfinished_thread_cause / keep_unfinished_link_near`
  が protective / diffuse terrain 上で立つと、
  social pressure field がなくても
  - `engagement_mode = wait`
  - `execution_mode = defer_with_presence`
  に落ちる
- これで chain は
  - `memory -> utterance_reason -> terrain -> strategy/route -> posture -> actuation`
  まで一本化

- targeted regression:
  - `pytest tests\\test_inner_os_live_engagement_state.py -q`
    - `19 passed, 1 warning`
- nearby regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_route_prompts.py -q`
    - `78 passed, 1 warning`

## 2026-04-03 relation-aware reaction-first channel

- `presence_hold_state`
  - relation reentry / guarded relation を `utterance_reason_packet` から直接読むようにした
  - relation reentry では `reentry_open` へ、guarded relation では `hold_room` 優勢へ寄せる
- `nonverbal_response_state`
  - relation reentry 用の `bridge_ack_presence`
  - guarded relation 用の `guarded_hold_presence`
  を canonical state として追加し、generic ack より優先するようにした
- `response_selection_state`
  - `utterance_reason_packet` を読み、relation reentry は `backchannel`、guarded relation は `hold` を選びやすくした
- `actuation_plan`
  - `utterance_reason_packet` を `presence_hold / nonverbal / response_selection` まで通し、
    `memory -> reason -> terrain -> response_channel`
    を一本化した
- targeted regression:
  - `pytest tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_live_engagement_state.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py -q`
    - `96 passed, 1 warning`

## 2026-04-03 response_channel runtime shaping

- `runtime._shape_inner_os_content_sequence(...)` に `response_channel` の final shaping 分岐を追加
  - `backchannel` は listener token / fast ack を優先
  - `hold` は breath 側の minimal fallback を優先
- これで `response_channel` は state / actuation の表示項目ではなく、text-only runtime でも実際の reaction を変えるようになった
- `tests/test_runtime_bright_short_sequence.py`
  - 長い sequence があっても `backchannel / hold` の channel fallback を優先することを固定
- `tests/test_runtime_process_turn_hooks.py`
  - harness を runtime の新 helper に追従させた
- regression:
  - `pytest tests\\test_runtime_bright_short_sequence.py tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py -q`
    - `83 passed, 1 warning`

## 2026-04-03 response_channel timing profile

- `runtime._apply_inner_os_actuation_timing_profile(...)` を追加し、
  `wait_before_action / nonverbal_response_state.timing_bias / response_channel`
  から `opening_pace_windowed / pause_insertion / response_length` を更新するようにした
- 対象は `backchannel / hold` に限定し、既存の speak/defer clarify 経路は汚さない
- `bright_bounce` の cleanup は維持しつつ、`backchannel` の entry timing は profile 上に残す
- `tests/test_runtime_process_turn_hooks.py` に timing 回帰を追加し、
  `backchannel -> ready/none/short` と `hold -> held/soft_pause/short` を固定した
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_runtime_bright_short_sequence.py tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_headless_runtime.py -q`
    - `85 passed, 1 warning`

## 2026-04-04 typed surface runtime bridge

- `surface_context_packet` に次の runtime / discourse 判定用 state を正式に通した
  - `interaction_policy_strategy`
  - `interaction_policy_opening_move`
  - `scene_family`
  - `action_posture_mode`
  - `actuation_primary_action`
  - `actuation_response_channel`
- `derive_discourse_shape(...)` は repair 文脈で
  - `light_playful`
  - `light_bounce`
  だけを理由に `bright_bounce` へ落ちないように tightening
- ただし次の明示的 bright は維持
  - `bright_continuity`
  - `reason_driven_bright`
- `runtime.py` の内部保持も少し圧縮
  - `self._last_surface_context_packet`
  - `current_state["surface_context_packet"]`
  を `coerce_surface_context_packet(...)` で typed contract に寄せた
- regression:
  - `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py -q`
    - `10 passed, 1 warning`
  - `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py tests\\test_inner_os_surface_context_packet_reasoning.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_runtime_route_prompts.py -q`
    - `159 passed, 1 warning`
  - `pytest tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
    - `149 passed, 1 warning`

## 2026-04-03 turn_timing_hint canonical化

- `inner_os/headless_runtime.py`
  - `TurnTimingHint` と `derive_turn_timing_hint(...)` を追加
  - `response_channel / wait_before_action / nonverbal_response_state.timing_bias`
    から `entry_window / pause_profile / overlap_policy / interruptibility / minimum_wait_ms / interrupt_guard_ms`
    を導出するようにした
  - `HeadlessTurnResult` に `turn_timing_hint` を追加し、actuation 正本から timing hint を一緒に carry するようにした
- `emot_terrain_lab/hub/runtime.py`
  - `metrics` に `inner_os/headless_wait_ms` と `inner_os/headless_interrupt_guard_ms` を追加
  - `persona_meta["inner_os"]` と `current_state` に `actuation_turn_timing_hint` を追加
- `emot_terrain_lab/hub/lmstudio_pipeline_probe.py`
  - `actuation_wait_before_action` と `actuation_turn_timing_hint` を probe に通し、render にも表示するようにした
- `tests`
  - `tests/test_inner_os_headless_runtime.py`
  - `tests/test_runtime_process_turn_hooks.py`
  - `tests/test_lmstudio_pipeline_probe.py`
  に timing hint の contract を追加した
- regression:
  - `pytest tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`
    - `76 passed, 1 warning`

## 2026-04-03 emit timing contract

- `runtime._apply_inner_os_emit_timing(...)` を追加
  - `turn_timing_hint` から runtime 側の `inner_os_emit_timing` を導出
  - `response_channel / entry_window / overlap_policy / minimum_wait_ms / interrupt_guard_ms`
    に加えて
    `effective_emit_delay_ms / effective_latency_ms`
    を確定する
- `backchannel / hold / defer` のときは
  `response.latency_ms` を `minimum_wait_ms` の floor に合わせる
  - まだ `sleep` は入れていない
  - 先に observer / live / probe が同じ timing contract を読む段階
- `response.controls_used` と `response.controls` の両方に
  `inner_os_emit_timing` を載せ、
  `persona_meta["inner_os"]` / `current_state` にも
  `actuation_emit_timing` を同期
- `lmstudio_pipeline_probe.py`
  - `emit_timing` を表示可能にした
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`
    - `76 passed, 1 warning`
  - absolute deadline と guard window を追加
    - `emit_not_before_ms`
    - `interrupt_guard_until_ms`
    - `wait_applied`
    - `wait_applied_ms`
  - `backchannel / hold / defer` では opt-in の runtime wait を許可
    - `effective_emit_delay_ms` が正のときだけ `sleep`
    - helper 単体で wait の有無を検証
  - rerun:
    - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`
      - `78 passed, 1 warning`

## 2026-04-03 turn timing guard

- 前ターンの `inner_os_emit_timing` を runtime に保持するようにした
  - `emit_not_before_ms`
  - `interrupt_guard_until_ms`
  - `response_channel`
  - `overlap_policy`
- 次ターンの gate 作成直後に `runtime._apply_inner_os_turn_timing_guard(...)` を通す
  - `hold / backchannel / defer` でまだ emit window の中なら `force_listen=True`
  - `interrupt_guard` が生きていて、かつ `allow_soft_overlap` でない場合は、相手の発話継続を優先する
- これで `まだ出さない` と `今は被せない` が metadata ではなく `talk_mode=watch` 側の判断に効き始めた
- `last_gate_context` にも `inner_os_timing_guard` を残すようにした
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`
    - `81 passed, 1 warning`

## 2026-04-03 timing guard visibility

- `timing_guard` を `persona_meta["inner_os"]` に載せた
  - `active`
  - `reason`
  - `response_channel`
  - `overlap_policy`
  - `emit_not_before_ms`
  - `interrupt_guard_until_ms`
- `gate_force_listen` も `persona_meta["inner_os"]` と probe で読めるようにした
- `lmstudio_pipeline_probe.py` に
  - `gate_force_listen`
  - `timing_guard`
  を追加し、render でも表示するようにした
- これで live 実走前でも
  - 前ターンの guard が立っていたか
  - 今回 `watch / force_listen` に寄ったか
  を probe から確認できる
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py tests\\test_inner_os_headless_runtime.py -q`
    - `81 passed, 1 warning`

## 2026-04-03 response_meta timing contract

- `runtime._serialize_response_meta(...)` に `actuation_emit_timing / timing_guard / gate_force_listen` を追加
- `response.controls_used` に `inner_os_timing_guard / inner_os_gate_force_listen` を同期
- distillation / transfer / live consumer でも probe と同じ timing contract を読めるようにした
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`

## 2026-04-03 streaming emit wait

- `runtime._inner_os_emit_wait_enabled()` を整理
  - instance override があればそれを優先
  - 通常は `config.latency.enable_loose` かつ `surface_mode=streaming` のとき guarded channel wait を有効化
- これで live/streaming では `emit_not_before_ms` に沿った短い wait が runtime 自体で入りやすくなった
- `tests/test_runtime_process_turn_hooks.py` に process_turn integration を追加し、
  guarded channel の headless actuation で `sleep / wait_applied / emit_not_before_ms` が実際に動くことを固定
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py -q`
    - `85 passed, 1 warning`

## 2026-04-04 typed runtime guidance bridge

- `runtime._run_inner_os_live_response_loop(...)` の `persistent_planning_hints` を
  - `interaction_policy_packet`
  - `action_posture`
  - `actuation_plan`
  - `surface_context_packet`
  の contract 起点へ寄せた
- `_bright_guidance_override(...)` / `_refresh_persistent_planning_hints(...)` / `_stamp_persistent_inner_os_controls(...)`
  でも `surface_context_packet` は contract で持ち、`controls_used` に同期するときだけ dict 化するようにした
- `_apply_qualia_gate(...)` の current guidance 経路でも
  - `inner_os_discourse_shape`
  - `inner_os_surface_context_packet`
  を coercion して current plan を優先する流れを維持した
- `_build_inner_os_llm_guidance(...)` の内部も
  - `interaction_policy_packet`
  - `surface_context_packet`
  - `action_posture`
  - `actuation_plan`
  を contract 起点へ寄せ、返り値だけ従来互換の dict にした
- `SurfaceContextPacket` は空 coercion でも Mapping として truthy になるので、
  builder 実行判定は `if not packet` でなく populated fields を見るようにした
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_surface_context_packet.py tests\\test_inner_os_discourse_shape.py tests\\test_human_output_examples.py -q`
    - `95 passed, 1 warning`
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_surface_context_packet_reasoning.py tests\\test_runtime_route_prompts.py -q`
    - `150 passed, 1 warning`
  - `pytest tests\\test_inner_os_bootstrap.py tests\\test_llm_hub_reason_chain.py -q`
    - `13 passed, 1 warning`
  - `surface_context_packet` の contract 起点化を
    - `_apply_inner_os_surface_profile(...)`
    - runtime 内 raw review helper
    にも広げた
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_human_output_examples.py tests\\test_inner_os_surface_context_packet.py tests\\test_lmstudio_pipeline_probe.py -q`
    - `88 passed, 1 warning`
  - `_compact_inner_os_sequence_text(...)` の bright cue packet 参照も contract 起点に変更
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_human_output_examples.py tests\\test_lmstudio_pipeline_probe.py -q`
    - `83 passed, 1 warning`

## 2026-04-04 typed integration gate export

- `integration_hooks.ResponseGateResult` の export 境界を見直した
- `interaction_policy_packet / action_posture / actuation_plan` は内部では contract のまま保持する
- `ResponseGateResult.to_dict()` だけが contract を plain dict へ再帰的に直列化する
- これで `integration_hooks` でも
  - 内部: typed contract
  - export: plain dict
  の境界が明示された
- regression:
  - `pytest tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_bootstrap.py tests\\test_runtime_process_turn_hooks.py -q`
    - `153 passed, 1 warning`

## 2026-04-04 typed runtime export boundary

- `runtime.py` に export helper を追加した
  - `_export_runtime_value(...)`
  - `_export_runtime_mapping(...)`
- `_serialize_response_meta(...)` の
  - `safety`
  - `perception_summary`
  - `retrieval_summary`
  を export helper 経由へ寄せた
- `_build_inner_os_llm_guidance(...)` の返り値でも
  - `conversation_contract`
  - `conversational_objects`
  - `object_operations`
  - `interaction_effects`
  - `interaction_judgement_view`
  - `interaction_judgement_summary`
  - `interaction_condition_report`
  - `interaction_inspection_report`
  - `interaction_audit_bundle`
  - `interaction_audit_casebook`
  - `interaction_audit_report`
  - `interaction_audit_reference_case_meta`
  を shallow `dict(...)` から export helper 経由へ置き換えた
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_lmstudio_pipeline_probe.py -q`
    - `143 passed, 1 warning`
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_bootstrap.py tests\\test_lmstudio_pipeline_probe.py -q`
    - `155 passed, 1 warning`
- `controls_used / persona_meta["inner_os"] / current_state.setdefault(...)` 側でも
  shallow `dict(...)` を export helper 経由へ寄せた
## 2026-04-04 typed expression hint bundle summaries

- `runtime.py` で
  - `scene`
  - `workspace`
  - `interaction_reasoning`
  - `interaction_audit`
  の bundle から persona summary を作る helper を追加
- `current_state` の復元も `_last_gate_context` の flat key を直接読むのでなく、bundle contract から派生する形へ寄せた
- `qualia_hint_bundle` も `_last_gate_context` の内部正本へ追加
- regression:
  - `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_bootstrap.py -q`
    - `154 passed, 1 warning`
## 2026-04-04 AI4Animation reference bridge

- AI4Animation の
  - PFNN
  - MANN
  - Neural State Machine
  - Local Motion Phases
  - DeepPhase
  - Codebook Matching
  を、`Inner OS` の設計言語へ翻訳する文書を追加
- 使う要素を
  - 連続 phase
  - 状態条件付き mixer
  - scene interaction controller
  - sparse cue から valid manifold への投影
  に限定
- 次の導入候補を `interaction_phase_state.py` と `response_mode_mixer` に固定
## 2026-04-04 policy packet orchestration bridge

- `interaction_policy_packet` を、flat key の材料ではなく orchestration の正本へ寄せた
- `inner_os/integration_hooks.py`
  - `_apply_interaction_policy_packet_views(...)` を追加
  - `interaction_policy_*` と `surface_*` の派生規則を 1 か所に集約
- `emot_terrain_lab/hub/runtime.py`
  - `_apply_interaction_policy_packet_to_current_state(...)`
  - `_apply_interaction_policy_packet_to_gate_context(...)`
  - `_interaction_policy_packet_summary(...)`
  を追加
  - current_state / gate_context / persona summary が packet 起点で揃うように変更
- direct regression を追加
  - `tests/test_runtime_process_turn_hooks.py`
  - `tests/test_inner_os_integration_hooks.py`
- regression:
  - `pytest tests\\test_inner_os_integration_hooks.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_bootstrap.py -q`
    - `156 passed, 1 warning`

## 2026-04-05 bright continuity stale shape override

- 課題
  - `bright_continuity` と `brief_shared_smile` が立っていても、既存の `inner_os_discourse_shape=reflect_step` が無条件で優先されると、最終応答が反射的な長文へ戻ることがあった
  - その結果
    - `strategy=shared_world_next_step`
    - `actuation=riff_current_comment`
    でも `discourse_shape=reflect_step` と `guarded_narrative_bridge` が残る経路があった
- 対応
  - `emot_terrain_lab/hub/runtime.py`
    - `_should_prefer_runtime_derived_discourse_shape(...)` を追加
    - `turn_delta / conversation_phase / utterance_reason_offer / shared_moment_kind` から bright の構造信号が明確なときは、stale な `reflect_step` より derived `bright_bounce` を優先
  - `tests/test_runtime_process_turn_hooks.py`
    - stale `reflect_step` を bright signal で上書きする回帰
    - `_derive_runtime_discourse_shape(...)` の direct regression
- regression
  - `pytest tests\\test_runtime_process_turn_hooks.py -q`
    - `84 passed, 1 warning`
  - `pytest tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_bootstrap.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py tests\\test_runtime_process_turn_hooks.py -q`
    - `163 passed, 1 warning`
- LM Studio live check
  - `gpt-oss-20b`
    - final: `ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。`
    - `discourse_shape=bright_bounce`
    - `allow_guarded_narrative_bridge=false`
  - `qwen3.5-4b`
    - final: `ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。`
    - `discourse_shape=bright_bounce`
    - `allow_guarded_narrative_bridge=false`
## 2026-04-05 reaction contract to probe

- `reaction_contract` を planner / llm_hub / runtime に続いて probe まで通した
- `emot_terrain_lab/hub/lmstudio_pipeline_probe.py`
  - `LMStudioPipelineProbe.reaction_contract` を追加
  - `to_dict()` と render に反映
  - post-review 時の `review_llm_bridge_text(...)` に `reaction_contract` を渡すよう変更
- `tests/test_lmstudio_pipeline_probe.py`
  - probe 表示に `reaction_contract` が出ることを固定
- `tests/test_lmstudio_pipeline_probe_controls.py`
  - controls 由来の contract を優先すること
  - bright small moment で `question_budget=0` と `interpretation_budget=none` が見えることを固定
- regression
  - `pytest tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
    - `151 passed, 1 warning`
  - `python -m py_compile emot_terrain_lab\\hub\\lmstudio_pipeline_probe.py emot_terrain_lab\\hub\\runtime.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py`
    - 通過
## 2026-04-05 reaction contract live alignment

- `reaction_contract` を probe まで通した後、live で整合を確認
- `qwen3.5-4b`
  - final: `ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。`
  - contract:
    - `stance=join`
    - `scale=small`
    - `question_budget=0`
    - `interpretation_budget=none`
    - `continuity_mode=continue`
  - raw original はまだ解釈寄りで `interpretive_bright_violation / too_many_sentences`
- `gpt-oss-20b`
  - final は同じ
  - contract も qwen と同じ軸へ整合
  - raw original はまだ質問寄りで `question_block_violation / too_many_sentences`
- contract 導出側の修正
  - `recent_dialogue_state` が dict のまま文字列化されないよう `_state_name(...)` を追加
  - `question_budget` は `discourse_shape.question_budget` を最優先し、`0` を `or` で落とさないよう修正
- regression
  - `pytest tests\\test_inner_os_reaction_contract.py tests\\test_lmstudio_pipeline_probe.py tests\\test_lmstudio_pipeline_probe_controls.py tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
    - `153 passed, 1 warning`
- 2026-04-07
  - `SubjectiveSceneState` / `SelfOtherAttributionState` / `SharedPresenceState` を追加し、`camera_observation -> subjective scene -> shared presence -> joint_state` の最小導線を実装
  - `derive_joint_state(...)` に self-view 系 state の入力を追加し、`common_ground / joint_attention / mutual_room / coupling_strength` に寄与させた
  - unit test と docs を追加し、state core としての self-view 実装を開始
