# Eval Run 004: Stimulus History Influence Probe

## 目的

ContextInfluence だけでなく、クオリア膜の刺激状態、履歴、正規化信頼度が reaction contract に効くかを確認した。

今回の焦点は、同じ入力でも以下が変わるかである。

- 刺激圧
- 初体験/新奇性スパイク
- 慣れ
- field clarity
- gradient confidence
- LLM を呼ぶかどうか

## 実装接続

追加した接続は `StimulusHistoryInfluence`。

入力として主に以下を見る。

- `qualia_structure_state.emergence`
- `qualia_structure_state.drift`
- `qualia_structure_state.intensity`
- `qualia_structure_state.stability`
- `qualia_state.habituation`
- `qualia_state.normalization_stats.*.range_trust`
- `qualia_state.normalization_stats.*.gradient_confidence`
- `environment.fog_density`
- `memory.ignition_readiness`
- `memory.replay_priority`
- `memory.reconsolidation_priority`

## プローブ結果

入力は同一。

```text
今日は少し疲れた。
```

### A: 慣れていて明瞭な場

```json
{
  "stimulus_pressure": 0.1552,
  "novelty_pressure": 0.182,
  "habituation_pressure": 0.6488,
  "field_clarity": 0.8,
  "gradient_confidence": 0.78,
  "response_bias": "habituated_small_response",
  "response_channel": "speak",
  "should_call_llm": true
}
```

反応は発話可能。
ただし `question_budget=0`、`interpretation_budget=none` のままなので、軽い一言に制限される。

### B: 初体験スパイク + 霧

```json
{
  "stimulus_pressure": 0.6792,
  "novelty_pressure": 0.8204,
  "habituation_pressure": 0.0893,
  "field_clarity": 0.14,
  "gradient_confidence": 0.12,
  "response_bias": "hold_for_clarity",
  "response_channel": "hold",
  "should_call_llm": false
}
```

反応は非発話。
`presence_hold` に落ち、LLM は呼ばれない。

## 解釈

この結果は、単なる surface policy や ContextInfluence ではなく、刺激場の読み取り状態が発話ゲートへ接続されたことを示す。

重要なのは、刺激が強いから話すのではない点である。
刺激が強く、かつ field clarity / gradient confidence が低い場合は、むしろ反応を抑える。

逆に、刺激が弱く、慣れと明瞭さがある場合は、短い発話を許可する。

## まだ未証明の範囲

- 実モデル出力がこの制約内で自然になるか
- 長期履歴から `global_range` を更新できるか
- 記憶再発火が過剰 hold と過少 hold の境界を改善するか
- 文化・安全・身体・環境の軸と刺激履歴軸の競合解決が十分か

次はこの `StimulusHistoryInfluence` を、実モデル評価 JSONL の集計軸に入れる。
