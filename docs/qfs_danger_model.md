# QFS Danger Model v1

EQNet の危険学習を「世界モデル」「性格」「行動評価」の三層で整理した設計ノート。危険の検知は全個体共通、危険への価値付けは PolicyPrior の係数で差異化する。

## 1. 危険を構成する 4 トラック
危険は 1bit ではなく、以下の情報源を複合して扱う。

| トラック | 具体例 | QFS での保持先 |
|---|---|---|
| **direct_risk_accum** (積み上げ) | 軽いストレス・ヒヤリハットの蓄積 | `h(potential, visit_count_direct)` 例: `potential * log(1 + visit_count_direct)` |
| **direct_risk_acute** (一撃) | 致命的・人生観が変わる事故 | `event_severity ∈ [0,1]` を MomentLog から最大値で保持、必要なら年単位で decay |
| **social_risk_score** | 他者の失敗・文化に刻まれた危険情報 | self_story / Monument / culture layer から抽出し `membrane_state` へ集約 |
| **fog_level** | 未知・予測誤差 | まだ踏破していないゾーンの不確実性指標 |

`membrane_state` 例:
```
membrane_state.update({
    "potential": potential,
    "fog_level": fog,
    "visit_count_direct": visit_count_direct,
    "visit_count_vicarious": visit_count_vicarious,
    "social_risk_score": social_risk,
    "direct_risk_accum": direct_risk_accum,
    "direct_risk_acute": direct_risk_acute,
})
```

### danger_score の合成
```
danger_score =
    w_direct_accum * direct_risk_accum +
    w_direct_acute * direct_risk_acute +
    w_social       * social_risk_score
```
- 積み上げ型は「軽微×多数」をじわじわ効かせる。
- 一撃型は `event_severity` で即座に危険領域化（必要に応じて長期減衰）。
- social リスクで「行ったことはないが危ないと聞いている」を反映。

## 2. 危険検知と性格の分離
- **danger_score** … 世界側の事実（全個体共通の危険マップ）。
- **PolicyPrior** … 個体側の性格係数。以下の 3 つを追加する。
  - `risk_aversion`: 危険をどれだけ嫌うか（損失回避の強さ）。
  - `thrill_gain`: fog や danger そのものから得る快感の強さ（刺激追求）。
  - `discount_rate`: 長期コストをどれだけ割り引くか（今>未来）。

### 未来リプレイの評価関数
```
future_reward =
    - risk_aversion * danger_score
    + thrill_gain   * fog
    - discount_rate * long_term_cost
    + base_reward
```
- 同じ danger_score を見ても、PolicyPrior の係数で「行く／行かない」が分岐。
- `risk_aversion` 高 → 保守的に回避。低 → 危険を取る傾向。
- `thrill_gain` 高 → 不確実性を報酬として捉え、スリル探索に向かう。
- `discount_rate` 高 → 目先を優先し、将来の危険を軽視。

## 3. レポートと挙動
- **危険マップの可視化**: terrain 上に `danger_score` を塗り、`direct/acute/social` の内訳も別レイヤ表示。
- **無自覚 vs 自覚的回避**: `danger_score` 高・`life_indicator.meta` 低 → 無自覚な回避、meta 高 → 自覚的に理由を把握している。
- **PolicyPrior 更新**: imagery replay 後、`danger_score` の内訳に応じて calmness / directness 等を微調整。`direct_risk_acute` が高い領域では慎重さを優先する等。

## 4. 今後の拡張メモ
1. MomentLog に `event_severity` と `risk_type`（near_miss / minor / major / fatal）を記録。
2. Nightly で self_story や culture_log から危険語を抽出し `social_risk_score` を更新。
3. `danger_score` を future replay の reward, PolicyPrior の学習率、life_indicator.meta のラベルに統合。
4. 個体差 (`risk_aversion` など) を persona / config から注入し、性格的な危険志向を表現。

## 5. Healing Layer Integration (QFS Healing Model v0)
危険学習（Danger Model v1）は「どこが危険か」を強化学習的に学ぶ層だが、単体では反芻や自己否定に陥りやすい。危険を学びつつ自己を壊さないため、以下の Healing/CBT レイヤを統合する。

### 5.1 Event→Thought→Emotion→Behavior 分解
Nightly で MomentLog を再処理し、危険イベントを ETEB の4要素に分解する。
- Event: 起きた事実
- Thought: 自動思考（例:「自分はダメだ」）
- Emotion: 恐怖・恥・怒り・悲しみ
- Behavior: 回避、固まる、誰かに助けを求めた etc
Nightly が埋めるフィールド例: `moment.cog_thoughts`, `moment.cog_emotion_tags`, `moment.cog_behavior_tags`

### 5.2 認知の歪みタグ付け
ETEB 情報から LLM が認知パターン（全か無か思考／破局視／自己批判の過剰 etc）を推定し、`moment.cog_distortion_tags` に記録する。これで danger_score と「どう解釈したか」を分離できる。

### 5.3 Reframing & Healing Script
危険シーンを再生したあと、Nightly は以下を生成して `self_story_reframe` / `healing_script` に保存する。
- `disputation`: 自動思考にあった歪み
- `alternative_thought`: 別の捉え方
- `self_compassion_message`: 自分への優しい声がけ
- `next_time_plan`: 次に同じ状況が来たときの健全な対処案
これにより危険学習の日にも心理的バッファを持てる。

### 5.4 Healing Future Replay
`run_healing_replay` では、self-compassion や support-seeking に寄せた `healing_intent_vec` を使って `simulate_future(..., mode=IMAGERY)` を実行し、避けるだけでなく「優しい対処ルート」を練習する。危険を理解しつつ、支えを求めながら進む未来をリハーサルできる。

### 5.5 Rumination Guard
過度な反芻を防ぐため `life_indicator` に `rumination_level` を追加。
- 高 danger + distortion 多発 + ネガティブ replay 多 → rumination_level を上げる
- healing_script や reframe が活用された日 → rumination_level を下げる
- しきい値を超えたら acute danger replay の回数を制限し、healing_script を優先表示

### 5.6 レポート統合
Nightly レポートで Danger/Healing を並列表示。
- Danger: `danger_score` heatmap、direct/acute/social 内訳、potential/fog の分布
- Healing: distortion tag 件数、reframe/healing_script 生成数、rumination_level、healing replay 実施可否
「危険を学ぶスピード」と「心の耐久度」を同時に監視できる。
