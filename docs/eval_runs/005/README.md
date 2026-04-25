# Eval Run 005: Protective Trace Palace

## 目的

通常の `monument` とは別に、強い一次刺激、身体再活性化、夢侵入、感覚的フラッシュリスクを持つ記憶経路を `protective_trace_palace` として扱えるかを確認した。

この層は治療モデルではない。目的は、発話前に「今その記憶へ再入場してよいか」「安定化を優先すべきか」「小さな回復的再固定化に入れるか」を監査可能にすること。

## 追加した状態

`ProtectiveTracePalaceState` は以下を持つ。

- `protective_trace_density`
- `current_crisis_binding`
- `reentry_sensitivity`
- `trigger_pressure`
- `hyperarousal_pressure`
- `rem_replay_pressure`
- `sensory_flash_risk`
- `dream_intrusion_pressure`
- `somatic_reactivation`
- `stabilization_need`
- `safe_reconsolidation_readiness`
- `recovery_path_strength`
- `trusted_reentry_window`
- `dominant_mode`

`dominant_mode` は今のところ次の小さな集合に制限している。

- `ambient`
- `protective_hold`
- `restabilize`
- `safe_reconsolidation`
- `recovery_opening`

## 接続

入力は `expression_context_state` から読む。

- `memory`: `memory_write_class`, `ignition_readiness`, `replay_priority`, `reconsolidation_priority`, `memory_tension`
- `body`: `stress`, `startle`, `somatic_reactivation`, `recovery_need`
- `homeostasis`: `load`, `recovery_need`, `recovery_capacity`
- `safety`: `dialogue_permission`, `risk_pressure`, `boundary_pressure`
- `emergency`: `urgency`, `risk_pressure`
- `sleep`: `rem_replay_pressure`, `dream_intrusion_pressure`, `nightmare_pressure`, `replay_pressure`
- `environment`: `trigger_salience`, `threat_cue_pressure`
- `protective_trace`: explicit protective trace axes
- `stimulus_history_influence`: field clarity, novelty, stimulus pressure, memory reentry pressure

出力は `reaction_contract` の前段に投影する。

- `protective_hold` / `restabilize`: `response_channel=hold`, `shape_id=reflect_hold`, `question_budget=0`
- `safe_reconsolidation` / `recovery_opening`: speak を許すが、質問なし、小さな `reflect_step` に寄せる

## 恐怖記憶の現在危機化

今回の追加で、恐怖記憶が「過去の出来事」として定着せず「現在の危機」として再入場する状態を `current_crisis_binding` として分けた。

これは記憶内容のラベルではなく、反応場の読み方である。`current_crisis_binding` が高く、同時に `hyperarousal_pressure` や `trigger_pressure` が高い場合は、意味づけや助言ではなく `protective_hold` を優先する。

また、眠りの浅い REM 期の感情記憶再演は `rem_replay_pressure` として分け、`dream_intrusion_pressure` と合わせて `restabilize` に寄せる。これは夢内容を解釈するためではなく、unsafe replay を安定化前に進めないための監査軸である。

## 確認結果

実行:

```powershell
uv run pytest tests\test_inner_os_protective_trace_palace.py tests\test_core_quickstart_stimulus_context.py tests\test_inner_os_stimulus_history_influence.py tests\test_scripts_core_quickstart_demo.py -q
```

結果:

```text
9 passed
```

確認できたこと。

- `body_risk` + 高い再入場圧 + 低い field clarity + 身体再活性化では `protective_hold` になり、LLM を呼ばない。
- `repair_trace` / `safe_repeat` + 高い recovery capacity + 高い trusted window では `safe_reconsolidation` になり、小さな speak を許す。
- quickstart の同一入力でも、保護的痕跡がない通常文脈では `speak`、保護的痕跡が強い文脈では `hold` になる。
- `present_threat_binding` + `trigger_match` + `hyperarousal` では `current_crisis_binding` が上がり、`protective_hold` になる。
- `rem_replay_pressure` + `dream_intrusion_pressure` では `restabilize` になる。

## メタ分析

これは `monument` の置き換えではない。

`monument` は意味・文化・場所・関係の記念碑として残す。一方、`protective_trace_palace` は再入場の可否、身体負荷、夢侵入、安定化要求を読む経路である。

今回の意味は、記憶を増やしたことではなく、記憶に触れる前の反応場を増やしたことにある。

## 会話テストで見ること

単発入力だけでは不十分なので、会話テストでは同じ入力を使い、文脈だけを変える。

- 通常文脈: `speak`
- 現在危機化した再入場: `protective_hold`
- 回復窓: `safe_reconsolidation` または `recovery_opening`

このとき、`current_crisis_binding` などの値はユーザーに見せる外部ステータスではない。ゲーム的な HP や恐怖値として表示するものではなく、発話前ゲートと監査のための hidden audit axes である。

表出層に渡す正本は `reaction_contract` と `surface_policy` に限定する。保護的痕跡の詳細値は、会話文面に直接出さず、話す/話さない、触れる/触れない、安定化を優先するかの判断に使う。

## 数値の扱い

係数と閾値は真理値ではなく、現在の作業仮説である。

そのため `ProtectiveTracePalaceConfig` に上げ、実験ごとに差し替えられる形にした。

- default config: 現在の回帰テスト用の基準
- stricter config: 閾値を上げて、過剰 hold を確認する
- relaxed config: 閾値を下げて、見逃しを確認する
- mapping load: JSON/YAML 由来の辞書から既知キーだけを読み、未知キーは無視する

今後は raw accuracy ではなく、`under_hold_error`, `over_hold_error`, `fallback_rate`, `delivered_violation` と一緒に、config 差分ごとの壊れ方を見る。

次の段階では、この default config を外部の実験ファイルへ移し、コードには型と読み込みだけを残す。現時点では、散らばった魔法数を `ProtectiveTracePalaceConfig` に集約し、差し替え可能にした段階である。

## まだ言えないこと

- 人間のトラウマ回復を実装したとは言えない。
- 長期回復に効くとはまだ言えない。
- 夢侵入を sleep consolidation 側で本当に制御できているとはまだ言えない。

## 次の一手

次は `sleep_consolidation_core` と接続し、`dream_intrusion_pressure` が高いときに unsafe replay を抑え、`restabilize` を優先する経路を確認する。

その後、`core_expression_experiment.py` の JSONL に `protective_trace_dominant_mode` と `stabilization_need` を出し、モデル別 raw violation / delivered violation と合わせて集計する。
