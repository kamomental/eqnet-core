# Amadeus Transition Progress 2026-03-29

## 現在地
`Inner OS` を「アマデウス級の継続人格対話体」へ近づけるため、既存断片の canonical 化を進めています。
現時点で core として揃ってきたものは次です。

- `growth_state`
- `development_transition_policy`
- `epistemic_state`
- `qualia_structure_state`
- `heartbeat_structure_state`
- `utterance_reason_packet`
- `discourse_shape`
- `memory_dynamics`

## この round までに通ったもの

### 1. slow-state の canonical contract
- `growth_state`
  - `development / forgetting / sleep_consolidation / transfer` を束ねる slow-state
- `epistemic_state`
  - freshness / verification / stale risk を持つ知識側 state
- `qualia_structure_state`
  - 静的ラベルではなく `phase / emergence / stability / drift / trace` を持つ時系列状態
- `heartbeat_structure_state`
  - `pulse / attunement / recovery / tempo / reaction` を持つ身体寄り状態

### 2. 発話理由の chain
- `appraisal_state`
- `meaning_update_state`
- `utterance_reason_packet`

これで「mode だからそう返す」ではなく、「何が少し変わったからこう返す」の下流を持てるようになりました。

### 3. 表出の canonical 化
- `discourse_shape`
  - `anchor_reopen / reflect_hold / bright_bounce / reflect_step`
- runtime / probe / qualia gate suppression に `discourse_shape` を配線
- bright/live の stale containment を削り、`bright_bounce` の final まで通る経路を確保

### 4. `memory_dynamics` の canonical 化
- `palace / monument / ignition / reconsolidation / forgetting`
  を 1 つの typed state として統合
- `integration_hooks / transfer_package / continuity_summary / runtime`
  にはすでに接続済み
- この round でさらに
  - `sleep_consolidation_core`
  - `emot_terrain_lab/sleep/inner_os_bridge.py`
  - `ops/nightly.py`
  - `daily_carry_summary.py`
  まで通し、nightly と carry にも同じ state が残るようにした

## いま評価で見えていること

### deep / reopen
- final は raw よりかなり自然
- reopen は anchor を知っている感じが戻っている

### bright / live
- `shared_delight / light_bounce / warm_laugh_ack`
  の final は live で成立し始めた
- ただし upstream の `route / response_strategy / live_engagement`
  にはまだ containment 寄りの判定が残る

### nightly / carry
- `growth_state` と `memory_dynamics` は same-turn だけでなく overnight carry にも入った
- これで翌日に残る summary が、単なる memory class だけでなく
  `monument / ignition / consolidation / tension`
  を含めて読めるようになった

## 次の open issue

### `AMA-11` `organism_state`
- `growth / qualia / heartbeat / epistemic / relation`
  を 1 つの organism dynamics に束ねる

### `AMA-12` `joint_state`
- 会話を joint action として扱うため、
  `shared_tension / shared_delight / repair_readiness / common_ground`
  を canonical state にする

### `AMA-13` `memory_dynamics`
- contract 自体は立った
- 次は `organism_state` との接続が本丸

### `AMA-14` explorer temporalization
- explorer はまだ static distance 表示が中心
- backend にある `qualia_structure_state / heartbeat_structure_state / trace`
  を UI 側へ通す必要がある

### `AMA-15` closed loop
- VLM / TTS を含む closed loop を canonical state に接続する

## 次の優先順位
1. `AMA-11` `organism_state`
2. `AMA-12` `joint_state`
3. `AMA-14` explorer temporalization
4. `AMA-15` VLM / TTS closed loop

## 評価方針
- `pytest`
- 未使用確認ケース
- LM Studio live

この 3 本で確認する。
文面だけでなく、

- `state`
- `packet`
- `discourse_shape`
- `final`

の因果が説明できる変更だけを採る。
