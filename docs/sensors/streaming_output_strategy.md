# Streaming Sensor 出力戦略（議論用）

この文書は、現行コードを確認したうえで「Gradio / Observer / Hub / Replay をつなぐとき、最終的にどんな出力を正とするか」を整理するための設計メモです。

- Path表記メモ: 本資料では説明簡略化のため `persona/<id>.yaml` と表記する場合があるが、実運用では `personas/*.yaml`（複数）を前提とする。両者は同一概念（人格定義YAML）を指す。

## 1. 現状確認（コードベース）

- センサ正規化の中核は `StreamingSensorState.from_raw` にある。  
  参照: `eqnet/hub/streaming_sensor.py`
- ランタイムへの入口は `RuntimeSensors.tick`。最新スナップショットを保持し、任意で生フレームを JSONL に記録できる。  
  参照: `eqnet/hub/runtime_sensors.py`
- Runtime 側は `on_sensor_tick(raw_frame)` で取り込み、`_merge_sensor_metrics` で数値メトリクスを会話ターンに合流している。  
  参照: `emot_terrain_lab/hub/runtime.py:822`, `emot_terrain_lab/hub/runtime.py:1745`
- 合流後メトリクスは `MomentLogEntry.metrics` として保存される。  
  参照: `emot_terrain_lab/hub/runtime.py:2227`, `eqnet/logs/moment_log.py`
- Gradio ライブ橋渡しは別系統で `logs/live_observations.jsonl`, `logs/live_telemetry.jsonl`, `logs/live_diary.jsonl` を出している。  
  参照: `emot_terrain_lab/live/gradio_bridge.py`

## 2. いまの主課題

- `gradio_bridge` の出力と `StreamingSensorState` 契約が分かれており、再利用しづらい。
- 出力が「監査向け」「運用向け」「演出向け」で分離されていないため、用途ごとの責務が曖昧。
- 既存 `docs/streaming_sensor_pipeline.md` は文字化けがあり、現状運用の参照先として不安定。

## 3. 推奨アウトプット（3層）

### 3.1 Layer A: Sensor Ingress Log（取得監査）

- 目的: 「何を観測したか」の監査
- 推奨ファイル: `logs/sensor_ingress.jsonl`
- 単位: フレーム（高頻度）
- 必須項目:
  - `ts_ms`
  - `source` (`gradio|observer|mock|replay`)
  - `raw_frame_schema_version`
  - `raw_frame`（同意ポリシーで許可された項目のみ）
  - `consent_flags`
- 注意:
  - 画像/音声の生データは原則保存しない
  - 平文テキストは避け、必要ならハッシュ化または redact

### 3.2 Layer B: Turn Core Log（心の実行ログ）

- 目的: 「内部で何が起き、どう決定したか」の一次ソース
- 推奨ファイル: `logs/moment_log.jsonl`（既存を正とする）
- 単位: ターン（中頻度）
- 必須項目:
  - `turn_id`, `session_id`, `talk_mode`
  - `metrics`（`StreamingSensorState.metrics` 合流済み）
  - `gate_context`
  - `gate_context.interaction_decision`（v0互換: default=`IGNORE`）
  - `gate_context.interaction_reason_tags`（v0互換: default=`[]`）
  - `mood`, `prospective`, `qualia_vec`
  - `trace_observations`（必要時）
- 補助:
  - trace 側は `trace_v1` に同一ターンキーで連結
- 真実源（source of truth）:
  - `diary` は Layer B（`moment_log`）を元に生成される可読ビュー（派生成果物）であり、真実源は常に Layer B（`moment_log`）とする。
  - `interaction_reason_tags` は平文理由ではなく分類タグのみを保持する（監査・プライバシー要件）。
- Phase 2: Safety Policy Overrides（上位安全ポリシー）
  - `interaction_gate` は人格層として通常時の反応強度（`IGNORE|ACK_ONLY|SHORT_REPLY|ENGAGE`）を決定する。
  - 緊急時は上位の安全ポリシー層が `IGNORE` のみを禁止し、「介入」を強制せず無視しないこと（`ACK_ONLY` 以上）だけを保証する。
  - 最小ルール（実装準拠）:
    - `context_urgency >= threshold` の場合は `decision_min = ACK_ONLY` とし、`URGENT_CONTEXT` タグを追加する。
    - `safety_event == true` の場合は `decision_min = SHORT_REPLY` とし、`SAFETY_RISK` タグを追加する。
    - 最終決定は `interaction_decision = max(normal_decision, decision_min)` とし、強度は下げない。
    - `interaction_reason_tags` は分類タグのみを保持し、平文理由は保持しない。

### 3.3 Layer C: Replay / UI Payload（可視化・演出）

- 目的: RPG風可視化、比較表示、オブザーバUI
- 推奨ファイル: `logs/replay_payload.jsonl`（派生生成）
- 単位: ターンまたはステップ
- 必須項目:
  - `turn_id`
  - `world_profile` / `world_state`
  - `hero_state`（内部要約）
  - `expression_diff`（表情・姿勢・音声トーン）
  - `decision`, `subtitle`, `markers`
- 原則:
  - Layer B から生成する（UIが生ログに直接依存しない）

## 4. 推奨データフロー

1. `Gradio/Observer` が `raw_frame` を生成  
2. `RuntimeSensors.tick(raw_frame)` で `StreamingSensorState` 化  
3. Runtime `step()` 中で `_merge_sensor_metrics` によりターン統合  
4. `MomentLogEntry` に保存（Layer B）  
5. 同ターンを `trace_v1` に反映  
6. 後段ジョブで `replay_payload` を派生生成（Layer C）

## 5. 接続実装の最小方針

- 方針A（推奨）:
  - `gradio_bridge` は「取得専用」に寄せる
  - `MiniWorldStep` 直書きを減らし、`raw_frame -> RuntimeSensors` へ接続
- 方針B（移行期）:
  - 現行 `live_observations.jsonl` は維持
  - 追加で `sensor_ingress.jsonl` を吐き、後段でマージ

## 6. スキーマ運用ルール

- `schema_version` を各層に持たせる
- 破壊的変更はバージョンを上げる
- `metrics` は数値中心（文字列は補助）
- ハードコード値を避け、閾値は `config/*.yaml` 管理

## 7. セキュリティ / プライバシー

- 同意フラグがない項目は保存しない（deny-by-default）
- 音声文字起こしは必要最小限
- 個人識別可能情報は匿名化または参照ID化
- 出力先は用途別に分離し、アクセス制御を分ける

## 8. 段階導入計画

1. Phase 1: Layer A を追加（既存処理は維持）  
2. Phase 2: `gradio_bridge` を `RuntimeSensors` 契約へ寄せる  
3. Phase 3: Layer C を Layer B から自動生成  
4. Phase 4: Nightly/Audit が Layer B + trace を基準に評価

- Nightlyの成長更新:
  - Nightly による growth/habit の日次確定は段階導入中であり、現時点では一部がダミー/未接続の状態を含む（仕様は維持しつつ実装を順次充足する）。

## 9. 議論ポイント

- `raw_frame` の必須キー最小集合をどこまで固定するか
- `expression_diff` の仕様（顔/姿勢/声）をどの粒度で標準化するか
- Observer UI は Layer C のみ参照に統一するか
- 既存 `live_telemetry.jsonl` を Layer C へ寄せるか、別用途のまま残すか

## 10. Growth / Expression / Reaction 表示契約（v0）

### 10.1 目的

- Layer B（真実ログ）で積み上げた状態を、Layer C（物語/UI）へ安全に翻訳する
- ユーザーが「今どう感じ、どう決め、どう育ったか」を一目で理解できる状態を作る
- 方針:
  - Layer B は一次ソース（監査可能・再現可能）
  - Layer C は派生（再生成可能・演出可能）
  - UI は Layer C のみ参照（生ログ直結を禁止）

### 10.2 Layer C 必須フィールド（v0）

#### 10.2.1 reaction_line（因果の一行）

- 1ターンにつき1行で「内面 -> 判断 -> 行動」を表示する
- `text_redacted` はテンプレ文とタグ差し込みを基本とし、自由入力の平文は避ける
- 個人情報や会話本文は含めない

```json
{
  "reaction_line": {
    "v": 0,
    "text_redacted": "<template-based>",
    "tokens": [
      "STATE_TENSE_HIGH",
      "TRUST_UP",
      "DECISION_SLOW_DOWN",
      "ACTION_SPEAK_SOFT"
    ],
    "refs": {
      "turn_id": "turn-0001",
      "trace_id": "trace-0001"
    }
  }
}
```

#### 10.2.2 reason_tags（意思決定の理由タグ）

- `decision` の背後理由を安定タグで表す
- 破壊的変更は語彙の `version` 更新で扱う

```json
{
  "reason_tags": {
    "v": 0,
    "tags": ["SAFETY_GENTLE", "CURIOUS_PROBE", "LOW_CONFIDENCE", "BOND_PROTECT"],
    "weights": {
      "SAFETY_GENTLE": 0.7,
      "CURIOUS_PROBE": 0.4
    }
  }
}
```

#### 10.2.3 growth_state（育ってる実感の固定UI）

- 常時表示する成長メーター
- 日次または重要イベント後に微小変化を反映する
- 平文メモは避け、トークンとハッシュを利用する

```json
{
  "growth_state": {
    "v": 0,
    "axes": {
      "bond": {"value": 0.53, "delta": 0.02},
      "stability": {"value": 0.41, "delta": -0.01},
      "curiosity": {"value": 0.68, "delta": 0.00}
    },
    "event_note": {
      "note_tokens": ["LEARNED_BOUNDARY", "LEARNED_TRUST_CUE"],
      "note_hash": "hmac_sha256:..."
    }
  }
}
```

#### 10.2.4 expression_diff（表情・姿勢・声の差分）

- v0 は共通カテゴリ + 強度だけを固定する
- モデル/プロバイダ依存の詳細値は v1 以降で `detail.*` に分離する
- 世界観上の重要表現として `face.id=jitome` を正式語彙に含める

```json
{
  "expression_diff": {
    "v": 0,
    "face": {"id": "neutral|smile|surprise|tired|jitome", "intensity": 0.0},
    "pose": {"id": "still|lean_in|bounce|guarded", "intensity": 0.0},
    "voice": {"id": "soft|normal|excited|whisper", "intensity": 0.0}
  }
}
```

### 10.3 Layer B -> Layer C マッピング規約（v0）

- 入力ソース:
  - `MomentLogEntry.metrics`（数値中心）
  - `gate_context`（カテゴリ/理由/スイッチ）
  - `trace_v1`（必要時）
- 変換原則:
  - ハードコード禁止（閾値/重みは `config/*.yaml` 管理）
  - deny-by-default（同意のない要素はUI出力しない）
  - 平文回避（テンプレ + タグ差し込み、またはトークンのみで表示可能にする）

### 10.4 設定ファイル例（`config/ui_telemetry_map.yaml`）

```yaml
version: 0

state_tokens:
  tense:
    metric: tension_index
    bands:
      - {max: 0.3, token: STATE_TENSE_LOW}
      - {max: 0.7, token: STATE_TENSE_MID}
      - {max: 1.0, token: STATE_TENSE_HIGH}

  trust:
    metric: trust_index
    bands:
      - {max: 0.3, token: STATE_TRUST_LOW}
      - {max: 0.7, token: STATE_TRUST_MID}
      - {max: 1.0, token: STATE_TRUST_HIGH}

expression_rules:
  - when: {token: STATE_TENSE_HIGH}
    set:
      face: {id: jitome, intensity: 0.6}
      pose: {id: guarded, intensity: 0.5}
      voice: {id: soft, intensity: 0.3}

  - when: {tag: CURIOUS_PROBE}
    set:
      pose: {id: lean_in, intensity: 0.4}

growth_axes:
  bond:
    clamp: [0.0, 1.0]
    max_daily_delta: 0.03
  stability:
    clamp: [0.0, 1.0]
    max_daily_delta: 0.03
  curiosity:
    clamp: [0.0, 1.0]
    max_daily_delta: 0.03
```

### 10.5 生成責務

- `reason_tags`: Runtime の decision / gate 結果から生成（Layer B -> C）
- `reaction_line`: `tokens` をもとに UI 側で文生成（多言語・演出差し替え対応）
- `expression_diff`: `tokens`/`tags` からルール適用で生成（config駆動）
- `growth_state`: Nightly または後段ジョブで更新（Layer B + trace を材料に生成）

### 10.6 v0 最低保証チェック

- 毎ターン必須:
  - `reaction_line.tokens`
  - `reason_tags.tags`
  - `expression_diff`
- 毎日または重要イベント後必須:
  - `growth_state.axes.*.delta`
  - `event_note.note_tokens`
- 再生成性:
  - Layer C は Layer B + trace から再生成可能であること（Layer C のみで正としない）

### 10.7 演出レイヤーの但し書き（必須）

- Layer C の感情表示は **演出用ビュー** であり、診断・確定ラベルではない。
- Layer C は Layer B / trace の数値・タグからの **推定結果** である。
- UI には常時またはヘルプで次の注記を表示する:
  - 「この感情表示は実ログからの推定に基づく演出です」
  - 「内部状態の厳密値は監査ログ（Layer B / trace）を参照してください」
- 同意・権限のない情報は推定表示にも使わない（deny-by-default）。

### 10.8 感情表示プリセット（v0）

- 伝達優先で、v0 は人間が直感的に理解しやすい表示を標準とする。
- UI 標準セット:
  - `basic_emotion_5`: `喜 / 怒 / 哀 / 楽 / 驚`
  - `basic_emotion_plus`: `normal / flat / jitome` を追加した拡張セット
  - `plutchik_wheel_8`: `joy, trust, fear, surprise, sadness, disgust, anger, anticipation`
- 実装ルール:
  - 内部の `reason_tags` / `state_tokens` / `metrics` から感情カテゴリへ写像する
  - 表示カテゴリは固定語彙（`emotion_vocab_version`）で管理する
  - 不確実性が高い場合は単一断定せず `mixed` または `unknown` を許可する
  - `normal` は「ベースライン状態」、`flat` は「感情起伏が低い状態」、`jitome` は「軽い蔑み/警戒の演出状態」として扱う
  - `jitome` は診断ラベルではなく演出ラベルであることを明記する

```json
{
  "emotion_view": {
    "v": 0,
    "mode": "basic_emotion_plus",
    "primary": "驚",
    "secondary": "喜",
    "stability": 0.62,
    "source": "estimated_from_layer_b",
    "evidence": {
      "tokens": ["STATE_TENSE_MID", "STATE_TRUST_UP"],
      "reason_tags": ["CURIOUS_PROBE", "BOND_PROTECT"]
    }
  }
}
```

### 10.9 感情ビューの誤解防止ルール（v0）

- UI では `emotion` ではなく `emotion_view`（感情ビュー）という名称を使う。
- `confidence` という語は使わず、`stability`（表示安定度）を使う。
- `stability` は「当たり確率」ではなく「直近Nターンで表示カテゴリがどれだけ一貫しているか」を示す。
- `emotion_view.evidence` は最小限の根拠タグのみ保持し、平文や個人情報は保持しない。
- `stability` が低い場合は `mixed` または `unknown` を優先表示し、単一断定を避ける。

### 10.10 固定語彙ファイル（`ui_emotion_vocab.yaml`）

- `emotion_vocab_version` を実体化するため、語彙は設定ファイルで管理する。
- 少なくとも以下を定義する:
  - `basic_emotion_5`
  - `basic_emotion_plus`
  - `plutchik_wheel_8`
  - `aliases`（表示名差し替え）
  - `non_diagnostic_labels`（例: `jitome`）
- UI は語彙ファイルの注記を読み、`non_diagnostic_labels` には常に「演出ラベル」注記を添える。

### 10.11 emotion_view 写像ファイル（`ui_emotion_map.yaml`）

- `emotion_view` の生成規則は単一設定に集約する。
- 入力:
  - `state_tokens`
  - `reason_tags`
  - 必要最小限のメトリクス帯域
- 出力:
  - `emotion_view.primary`
  - `emotion_view.secondary`
  - `emotion_view.mode`
- `stability` の算出パラメータ（例: `window_turns`, `decay`）も設定に置く。
- 低 `stability` 時の `mixed/unknown` 優先は実装のハードルールとし、閾値は設定で管理する。

### 10.12 persona_style（jitome を含む表現バイアス）

- `jitome` を状態そのものではなく、表現スタイル（性格バイアス）として扱う。
- Layer B は汚さず、Layer C 生成時にのみ適用する。
- 推奨パラメータ:
  - `jitome_bias`（0..1）
  - `jitome_sensitivity`（0..1）
  - `jitome_cooldown_turns`（>=0）
- 生成原則:
  - 根拠タグは Layer B 由来を維持する
  - 最後の表情選択で性格バイアスをかける
  - 単発トリガーより複合条件を優先し、誤解を抑える

## 11. 可観測性・互換性・決定性契約（v0）

### 11.1 互換性レベル（Compatibility Level）

- 各層（Layer A/B/C）は `schema_version` に加えて `compatibility_level` を持つ。
- `compatibility_level` は次を使う:
  - `PATCH`: フィールド追加・説明追加のみ（既存 UI / 既存ジョブを壊さない）
  - `MINOR`: 語彙追加・新モード追加（デフォルト挙動は維持）
  - `MAJOR`: 意味変更・削除を含む破壊的変更
- 同一キーの意味変更は必ず `MAJOR` とする。
- 互換性判定は Layer C 生成ジョブと UI 起動時チェックの両方で実施する。

```json
{
  "schema_version": "replay_payload.v0",
  "compatibility_level": "PATCH"
}
```

### 11.2 決定性契約（Determinism Contract）

- 原則: 同じ Layer B + trace 入力からは同じ Layer C を再生成できること。
- 演出ゆらぎを使う場合は `render_seed` を必須にし、seed 駆動でのみ乱択する。
- `render_seed` は `HMAC-SHA256(session_id + ":" + turn_id + ":" + renderer_version, secret_key)` の先頭 N 桁を使う。
- `secret_key` は `.env` または秘密ストアで管理し、平文ログへ出力しない。
- 乱択が許可されるのは非本質表現（字幕の細部、微小モーション、効果音タイミング）に限定する。
- 乱択可能領域は `render_meta.allowed_variants` で列挙し、列挙外の差分は不正とみなす。

```json
{
  "render_meta": {
    "renderer_version": "emotion_renderer_v0",
    "render_seed": "d5c7c7e9a4f0d2a1",
    "deterministic": true,
    "allowed_variants": [
      "subtitle_microphrase",
      "sfx_timing",
      "micro_motion"
    ]
  }
}
```

### 11.3 最低テスト要件（v0）

- `same_input_same_output`: 同一入力で `replay_payload` が一致する。
- `seed_changes_variant`: `render_seed` を変えたとき、許可された非本質表現のみが変化する。
- `compat_guard`: `compatibility_level=MAJOR` の差分は、明示的な移行フラグなしで起動失敗させる。

### 11.4 可観測性契約（Layer C 共通）

- Layer C の各サブビュー（`emotion_view` / `expression_diff` / `growth_state` / `reaction_line` / `fog_state` / `anomaly_flags` など）は `evidence` を持つか、`evidence=null` と `reason` を必ず持つ。
- `evidence` は参照キーのみを保持し、平文テキストや個人識別情報（PII）は保持しない。

## 12. 研究根拠マップ（一次ソース, 2023-2026）

本章は、`interaction_gate` / `safety override` / `monitorability` 設計の根拠を一次ソースで固定するための参照マップである。
運用ルール:
- 仕様本文で参照する研究は、原則として査読付き論文または公式研究公開（一次）を優先する。
- arXiv は未査読であるため、仕様根拠として使う場合はその旨を注記する。
- 題名の類似資料（二次要約・ニュース・ブログ転載）は根拠ソースとして扱わない。

### 12.1 Uncertainty / Entropy（反応様式の調整）

- Ling et al., 2024, NAACL: *Uncertainty Quantification for In-Context Learning of Large Language Models*  
  DOI: `10.18653/v1/2024.naacl-long.184`  
  URL: `https://aclanthology.org/2024.naacl-long.184/`
- Xia et al., 2025, Findings ACL: *A Survey of Uncertainty Estimation Methods on Large Language Models*  
  DOI: `10.18653/v1/2025.findings-acl.1101`  
  URL: `https://aclanthology.org/2025.findings-acl.1101/`
- Shen et al., 2024, ICML: *Thermometer: Towards Universal Calibration for Large Language Models*  
  URL: `https://research.ibm.com/publications/thermometer-towards-universal-calibration-for-large-language-models`
- Shorinwa et al., 2024/2025, arXiv（未査読）: *A Survey on Uncertainty Quantification of Large Language Models*  
  DOI: `10.48550/arXiv.2412.05563`  
  URL: `https://arxiv.org/abs/2412.05563`
- Liu et al., 2025, arXiv（未査読）: *Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey*  
  DOI: `10.48550/arXiv.2503.15850`  
  URL: `https://arxiv.org/abs/2503.15850`

### 12.2 Safety Overrides（下限保証を別層で保持）

- Wachi et al., 2024, IJCAI Survey Track: *A Survey of Constraint Formulations in Safe Reinforcement Learning*  
  DOI: `10.24963/ijcai.2024/913`  
  URL: `https://www.ijcai.org/proceedings/2024/913`

設計対応:
- `normal_decision`（人格層）と `decision_min`（安全下限）を分離する。
- 最終決定を `max(normal_decision, decision_min)` で合成し、緊急時に下限を割らない。

### 12.3 Monitorability / Interpretability（監査可能性）

- OpenAI, 2024-06-06: *Extracting concepts from GPT-4*（公式研究公開）  
  URL: `https://openai.com/index/extracting-concepts-from-gpt-4/`
- OpenAI, 2025-12-18: *Evaluating chain-of-thought monitorability*（公式研究公開）  
  URL: `https://openai.com/index/evaluating-chain-of-thought-monitorability/`

設計対応:
- `gate_context` に分離ログ（例: `normal_decision_before_entropy`, `normal_decision_after_entropy`, `decision_min`, `final_decision`）を残し、判断経路の監査可能性を担保する。

### 12.4 WEP（Words of Estimative Probability）

- Tang, Shen, Kejriwal, 2026, npj Complexity: *An evaluation of estimative uncertainty in large language models*  
  DOI: `10.1038/s44260-026-00070-6`  
  URL: `https://www.nature.com/articles/s44260-026-00070-6`

設計対応:
- 曖昧な確率語（WEP）ほど解釈ずれが生じうるため、`entropy` は反応様式の調整に使う。
- ただし、`safety override` の最低義務（`IGNORE` 禁止など）は WEP や entropy で解除しない。

## 13. Epistemic Guardrails（認識ガードレール）

### 13.1 目的

- 人は常に一次ソースを精査できず、解釈は認知バイアスの影響を受けうる。
- 本仕様は誤りを前提にしつつ、致命的な逸脱を防ぐための最小ガードレールを定義する。

### 13.2 Layer B 必須フィールド

- `source_tier`（必須）: `PRIMARY | REVIEW | SUMMARY | HEARSAY`
- `confidence`（必須）: 主観的確信度（自己評価）
- `evidence_strength`（必須）: 根拠の強さ（観測/引用/再現性など）
- `contrary_evidence_present`（必須）: 反証・不一致情報の存在フラグ（`true|false`）
- `evidence_strength` は「信頼の距離」を表す指標であり、`confidence`（確信）とは独立に記録する。

### 13.3 安全下限の独立性

- `decision_min` は常に独立であり、`entropy` / 認知バイアス / WEP などによって解除・減衰させない（`IGNORE` 禁止などの安全下限は不変）。

```yaml
epistemic_guardrails:
  require_source_tier: true
  require_confidence_split: true
  require_contrary_evidence_flag: true
  safety_floor_independent_from_entropy: true
```

## 13. External Reference (Social SoT)

- 社会制度レベルの原則（法は床、運用は複線、創造は余白）は実装仕様から分離し、正本を `docs/social_resilience_principles.md` とする。
- 本書では当該原則を参照のみ行い、同一原則文の重複管理を行わない。
