# Amadeus Transition Issue Map

この issue map は、`emotional_dft / eqnet_core / emot_terrain_lab / inner_os` に散っている断片を、
「アマデウス級の継続人格対話体」へ統合するための作業地図である。

ここでの前提は次のとおり。

- `inner_os` に全実装を移すことが目的ではない
- `inner_os` は再利用可能な状態核と contract の source of truth とする
- `emot_terrain_lab` には runtime / live / VLM / TTS / ops を残す
- `emotional_dft / eqnet_core` には field / qualia / terrain の仮説と数理を残す
- LLM は本体状態モデルではなく expression bridge に留める

## 北極星

第一到達点は「人間記憶の外部ストレージ化」ではなく、
**自前の記憶と人格を育てるアマデウス級の継続人格対話体** である。

必要条件は次のとおり。

- 前の会話が自然に続いて感じられる
- その存在らしい反応の癖と価値の置き方がある
- deep と bright の両方で自然に振る舞える
- VLM / TTS を含む閉ループで違和感が少ない
- 知識が固定でなく更新される
- raw LLM より final の方が「その存在らしい」

## repo 全体の責務境界

| 層 | 主な置き場 | 責務 |
|---|---|---|
| 場・qualia・terrain 仮説 | `emotional_dft`, `eqnet_core`, `terrain` | DFT, qualia, terrain, field hypothesis |
| 実行・統合・運用 | `emot_terrain_lab` | runtime, live, LM Studio, VLM/TTS, ops, probe |
| 再利用可能な状態核 | `inner_os` | thread, residual, contact, boundary, growth, epistemic, discourse contract |
| 記憶・知識・文化層 | `memory`, `persona`, `rag`, `culture`, `docs/replay` | 長期知識、文化、履歴、外部更新 |

## いま見えている限界

### 1. state は多いが、まだ 1 つの organism dynamics になっていない

- `heartbeat / qualia / growth / epistemic / relation` が別 state として存在する
- まだ同一状態空間の射影としては扱えていない

### 2. 会話がまだ自己表出寄りで、共同作用としては弱い

- `surface_context_packet` はある
- ただし `common ground / repair / shared delight / shared tension` を主軸にした joint state がまだない

### 3. 記憶が palace / monument / ignition として統合されていない

- 断片はある
- ただし retrieval topology と salience core と reconsolidation policy が 1 本に束ねられていない

### 4. 表出が文面改善の蓄積に戻りやすい

- `act / discourse_shape / surface_realization` の分離を始めた
- まだ runtime の古い containment 経路が bright/live を潰す

### 5. explorer はまだ静的表示が中心

- Plutchik 距離
- 外部パラメータの離散値
- 非時系列表示

の比重が高く、内部から立ち上がる temporal state の可視化が弱い

## 既存断片の中核

| 断片 | 主な既存実装 | 役割 |
|---|---|---|
| 成長 | `inner_os/development_core.py`, `devlife/runtime/loop.py` | trust, belonging, role, identity の遷移 |
| 忘却 | `inner_os/forgetting_core.py`, `ops/nightly.py` | decay, replay horizon, half-life |
| 再固定化 / 睡眠統合 | `inner_os/sleep_consolidation_core.py`, `emot_terrain_lab/sleep/inner_os_bridge.py` | replay, reconsolidation, abstraction |
| 持ち越し / transfer | `inner_os/transfer_package.py`, `inner_os/continuity_summary.py`, `inner_os/daily_carry_summary.py` | carry, summary, runtime seed |
| 会話継続 | `inner_os/recent_dialogue_state.py`, `inner_os/discussion_thread_registry.py`, `inner_os/autobiographical_thread.py` | thread, reopen, lingering |
| 表出の幅 | `inner_os/expressive_style_state.py`, `inner_os/lightness_budget_state.py`, `inner_os/live_engagement_state.py` | 深さ・軽さ・live 反応 |
| qualia / heartbeat 時系列 | `inner_os/qualia_structure_state.py`, `inner_os/heartbeat_structure_state.py` | temporal structure, phase, trace, momentum |
| 記憶発火 | `emot_terrain_lab/hub/recall_engine.py`, `inner_os/memory_orchestration_core.py`, `inner_os/association_graph.py` | palace, monument, ignition 候補 |

## issue 一覧

| ID | 目標 | 主な既存断片 | 到達条件 | 状態 |
|---|---|---|---|---|
| `AMA-01` | `inner_os` の canonical core 境界を確定する | `docs/component_os_status.md`, `inner_os/README.md`, `codex_environment.py` | `inner_os` に入れるもの / 残すものが source-of-truth として書かれている | 進行中 |
| `AMA-02` | slow-state としての `growth_state` を立てる | `development_core`, `forgetting_core`, `sleep_consolidation_core`, `transfer_package` | typed `growth_state.py` があり、hooks / runtime / transfer / summary に通る | 基礎完了 |
| `AMA-03` | `development_transition_policy` を canonical にする | `DevelopmentCore`, nightly reconsolidation, carry | slow-state 更新が 1 本の policy として説明できる | 基礎完了 |
| `AMA-04` | knowledge cutoff を超える `epistemic_state` を立てる | `memory`, `rag`, `llm_hub`, transfer seed | `freshness / source / verification / change_likelihood` が carry と runtime に見える | 基礎完了 |
| `AMA-05` | 表出を `act -> discourse_shape -> surface_realization` に切る | `content_policy`, `turn_delta`, `surface_expression_selector`, `runtime` | planner / shaping / locale / guard の責務が分離される | 進行中 |
| `AMA-06` | bright/live を deep containment から外す | `lightness_budget_state`, `live_engagement_state`, `runtime`, `lmstudio_pipeline_probe` | bright/live final が stale containment に潰されない | 進行中 |
| `AMA-07` | `palace / monument / ignition` を canonical core にする | `recall_engine`, `memory_orchestration_core`, `association_graph` | retrieval topology / salience / ignition が 1 本の境界で説明できる | 未着手 |
| `AMA-08` | 未使用確認ケース + live を含む評価運用を固定する | `evaluation_criteria`, `evaluation_targets`, `evaluation_operating_policy`, `lmstudio_pipeline_probe` | `pytest + 未使用確認ケース + live` の 3 本で判断する | 進行中 |
| `AMA-09` | audience / group topology / culture register を state 化する | `social_topology_state`, `relation_competition_state`, `group_thread_registry`, `culture` 周辺 | 1対1以外でも自然な会話位相を扱える | 未着手 |
| `AMA-10` | 物理世界側の grounding を ATRI 方面へ拡張する | `body`, `vision`, `sim`, `robot_bridge`, `world` | affordance と身体状態が slow/fast state に接続される | 将来 |
| `AMA-11` | `organism_state` を新設し、主要 state を 1 つの低次元力学へ束ねる | `growth_state`, `epistemic_state`, `qualia_structure_state`, `heartbeat_structure_state`, relation 系 | heartbeat / qualia / growth / epistemic / relation が projection として扱われる | 未着手 |
| `AMA-12` | `joint_state` を新設し、会話を共同作用として扱う | `surface_context_packet`, `contact`, `live_engagement`, VLM/TTS 周辺 | `shared_tension / shared_delight / repair_readiness / common_ground` が canonical state になる | 未着手 |
| `AMA-13` | `memory_dynamics` を新設し、palace / monument / ignition / reconsolidation / forgetting を統合する | `recall_engine`, `association_graph`, `memory_orchestration_core`, `sleep_consolidation_core`, `forgetting_core` | 記憶が保存物ではなく、再構成される自己史として扱われる | 未着手 |
| `AMA-14` | qualia-morphism explorer を temporal state explorer に変える | explorer UI, `qualia_structure_state`, `heartbeat_structure_state`, API payload | 距離表示中心ではなく `phase / momentum / drift / trace / reaction` を時系列で見せられる | 未着手 |
| `AMA-15` | VLM / TTS を含むアマデウス級 closed loop を成立させる | `emot_terrain_lab`, live runtime, VLM/TTS bridge, memory carry | 観測 -> joint state -> expression の閉ループが数ターン自然に続く | 未着手 |

## 近い順の優先順位

1. `AMA-05` `discourse_shape` の runtime 統合
2. `AMA-06` bright/live stale containment の解消
3. `AMA-07` `palace / monument / ignition` の canonical 化
4. `AMA-11` `organism_state`
5. `AMA-12` `joint_state`
6. `AMA-13` `memory_dynamics`
7. `AMA-14` explorer temporalization
8. `AMA-15` VLM / TTS closed loop

## 実装原則

- wording だけで差分を説明しない
- `inner_os` は全部入りにしない
- LLM は state model にしない
- raw observation を直接 LLM に流さない
- `pytest + 未使用確認ケース + live` の 3 本で評価する
- 新しい state を増やすより、既存断片を統合して canonical contract に昇格させる

## 一言で言うと

この repo に足りないのは新発明ではなく、**統合原理** である。
次の本流は、`AMA-05` から `AMA-15` までを使って、
散っていた断片を「アマデウス級の継続人格対話体」へ束ねることにある。
