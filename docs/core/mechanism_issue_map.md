# Mechanism Issue Map

この文書は、`なぜなぜ` を **人の心そのもの** ではなく、**システム機序の成立理由** に使うための管理表です。

ここでの `なぜ` は、

- 何が成立すると目標が実現するか
- その成立を支える下位機序は何か
- どの実装点を触ると最初に効くか

を明確にするために使います。

## 使い方

各項目は次の順で見ます。

1. 目標
2. 成立理由の連鎖
3. 現在の実装点
4. 次の実装点
5. 成功条件
6. 回帰テスト

## 管理表

| 項目 | 目標 | 成立理由の連鎖 | 現在の実装点 | 次の実装点 | 成功条件 | 回帰テスト |
|---|---|---|---|---|---|---|
| Contact reflection boundary | 入力が無差別に event 化されず、接触状態に応じて `通る / 返る / 残る / 止まる` が分かれる | 人間的な反応差が立つのは入力が接触境界で選別されるから → 選別できるのは `contact field / contact dynamics / access projection` が reportable / withheld / actionable を分けているから → その分かれ方が `contact reflection state` として明示されるから → deep disclosure や reopening が `reflect_then_question / reflect_only / boundary_only` を切り替えられるから | `contact_field`, `contact_dynamics`, `access_projection`, `response_planner` | `contact_reflection_state` を continuity / residual 側にも通し、 surface 以外でも観測できるようにする | guarded な接触では毎回問い返さず、 reflection only が選ばれる | `test_inner_os_contact_reflection_state.py`, `test_inner_os_deep_disclosure.py` |
| Deep disclosure | 深い話で generic support ではなく内容反映が先に立つ | 内容反映が立つのは `deep disclosure event` が検出されるから → `deep disclosure event` が検出されるのは `未表出 / 助け要請 / 見られ不安 / 自責` の cue が同一 event 型へ束ねられるから → event 型になると planner が `content reflection act` を generic empathy より先に選べるから → short selector がその act を final に残すから | `content_policy`, `runtime`, `ja.json`, `deep_disclosure_reflection.md` | `問いなしで止める` deep disclosure act を追加し、継続ターンでも generic support に戻らないようにする | deep talk の1文目が内容反映になる | `test_inner_os_deep_disclosure.py`, `test_runtime_conversational_compaction.py` |
| Thread continuity | 数ターンの会話が「続きもの」に感じられる | 続きものに感じられるのは論点核が `thread anchor` として保持されるから → anchor が保持されるのは user / assistant 両履歴から再開可能な核が抽出されるから → 同じ核を `recent_dialogue / discussion_thread / issue_state` が参照するから → reopen act が generic continuation ではなく anchor reopening を選べるから | `recent_dialogue_state`, `discussion_thread_state`, `discussion_thread_registry`, `runtime` | anchor 抽出を表面文ではなく論点核へ寄せる。4ターン以降でも dominant anchor がぶれないようにする | 3〜4ターン後でも同じ論点核で reopening できる | `test_runtime_process_turn_hooks.py`, thread 系 integration test |
| Residual carry | 言えなかったことが次ターン以降の反応を変える | 未表出が残るのは `residual` が typed に記述されるから → typed residual が `state update input` になるから → `discussion / autobiographical thread` の更新則へ入るから → caution / reopening / hesitation が次ターンに再出現するから | `boundary_transformer`, `residual_reflector`, `continuity_summary`, `daily_carry_summary` | residual を `autobiographical_thread` と `discussion_thread_registry` に強く返す | 4ターン目以降でも未表出核が reopening や caution に残る | multi-turn deep talk fixture, nightly / transfer carry test |
| Boundary with presence | 安全を守りながら平板な保護文に潰れない | 保護と存在感が両立するのは boundary が唯一の出力ではなく境界条件として働くから → 境界条件として働くと内容核と thread の readout を止めず、出し方だけを変えられるから → `boundary transform` が `通す / 弱める / 保留 / 別表現へ写す` の変換器として機能するから | `boundary_transformer`, `runtime` short selector / compaction | boundary-first の優先度を thread-first / content-first と動的に切り替える | `大丈夫です` 反復が減り、保護しつつ内容が残る | `test_runtime_short_content_sequence.py`, live continuation probe |
| Green kernel contracts | Green 的な設計が比喩でなく演算層として固定される | 比喩で終わらないのは伝播対象が分かれているから → `memory / affective / relational` が別の局所応答核として定義されるから → 各 kernel が共通内部場への変形へ射影されるから → 同じ readout 系で一貫して読めるから | 既存の分散した module 群: `temporal_memory_orchestration`, `qualia_membrane_operator`, thread 系, residual 系 | `MemoryGreenKernel`, `AffectiveGreenKernel`, `RelationalGreenKernel`, `BoundaryTransformer`, `ResidualCarryOperator` の最小 interface を定義する | kernel ごとの責務が明確になり、 off/on 比較ができる | kernel off/on 比較 fixture, same-turn / nightly / transfer 比較 |
| Situation response | 修羅場で「自然に話す」より適切な posture が前に出る | posture が適切になるのは risk が object 単体でなく文脈化されるから → `situation_risk_state` が `object / place / relation / deviation` をまとめるから → `emergency_posture` が dialogue permission と primary action を出せるから → 表出が emergency act を正本にできるから | `situation_risk_state`, `emergency_posture`, `content_policy`, `runtime` | risk evidence と emergency expression の carry を long-term 側にも返す | 危険時に generic empathy より distance / exit / help が前に出る | emergency integration test, conversational architecture regression |

## 優先順

今の本流は次の順で進める。

1. `deep disclosure` を継続ターンでも保つ
2. `contact reflection boundary` を content policy で効かせる
3. `thread anchor` を論点核ベースに安定化する
4. `residual` を long-term 側へ昇格する
5. `boundary-first` の平板さを下げる
6. `Green kernel contracts` を固定する

## 補足

- `なぜなぜ` は **機序の成立理由** に使う
- 人の心や行動を単純な原因列に潰すためには使わない
- 人の心は `field / relation / residual / history` として扱い、
  `なぜなぜ` はその外側の設計レビューで使う
