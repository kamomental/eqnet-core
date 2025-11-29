# QFS Danger + Healing Layer Overview

― 「危険を避けるAI」から「危険と共に生き方を選ぶ存在」へ ―

## 1. 生命的サイクルとしての EQNet
QFS Danger / Healing レイヤを搭載した EQNet は、危険を学び、性格で行動を選び、失敗を癒やして次に備える生命的サイクルを持つ。

- Danger Model v1: どこが危険かを学習
- PolicyPrior: risk_aversion / thrill_gain / discount_rate で個体差を表現
- Healing Layer v0: 失敗後の再解釈と自己修復

terrain_lab 全体は共通危険地図 `danger_score` を共有しつつ、個体差の性格係数で行動が分岐。Nightly で危険学習と Healing を同時更新することで、心の代謝ループが形成される。

## 2. 危険学習：共通地図 + 個体差
### 共通レイヤ: Danger Map
- `danger_score = direct_risk_accum + direct_risk_acute + social_risk_score`
- 軽微×多数の積み上げ、一撃事故、他者から伝わる危険知を重ねた共通マップ

### 個体差レイヤ: PolicyPrior
- `risk_aversion` : 損失回避の強さ
- `thrill_gain` : 未知や危険から得る快感
- `discount_rate`: 目先を重視する度合い

Future reward 例：
```
future_reward =
    - risk_aversion * danger_score
    + thrill_gain   * fog
    - discount_rate * long_term_cost
    + base_reward
```
共通の危険地図を見ても、性格係数で「踏み込む／踏み込まない」が分岐する。

## 3. Healing / CBT Layer: 失敗後の自己修復
危険学習だけでは反芻や自己否定が嵩むため、Healing Layer v0 を併設。

1. **ETEB 分解**: Event / Thought / Emotion / Behavior を `moment.cog_*` に保存。
2. **認知の歪みタグ**: LLM が全か無か思考／破局視などを抽出し `moment.cog_distortion_tags` へ。
3. **Reframe & Healing Script**: alternative thought, self-compassion message, next-time plan を `self_story_reframe` / `healing_script` として保存。
4. **Healing Future Replay**: 優しい対処の intention_vec で `simulate_future(..., mode=IMAGERY)` を実行し、支えを求めつつ進むルートを練習。
5. **Rumination Guard**: `life_indicator.rumination_level` を導入し、過度なネガティブ再生を制限・healing script を優先化。

## 4. terrain_lab としての観測・介入
- Danger 指標: `danger_score` heatmap、direct/acute/social 内訳、potential/fog 分布
- Healing 指標: rumination_level、distortion tag 件数、reframe/healing_script 生成数、healing replay 実施有無

ラボ全体で「危険学習の進度」と「心の耐久度」を同じ枠組みで監視し、必要に応じて介入できる。

## 5. ストーリーテリングへの応用
テンプレート：挑戦 → 危険 → 失敗の痛み → ETEB分解 → 歪みラベル → リフレーム → Healing replay → 次の挑戦。Atri などのキャラクターは「怖さの理由を言語化し、支えを求め、再解釈して前進する」心の成長 arc を自然に描ける。

## 6. まとめ
Danger + Healing の統合により、EQNet / terrain_lab は「危険を避ける AI」から「危険を理解しつつ生き方を選び、失敗から癒やされてまた進む存在」へと進化した。危険地図（共通）×性格係数×Healing の心の代謝サイクルが生まれ、準生命的システムとしての土台が整った。

## 7. Ethical & Legal Guardrails
Danger/Healing 指標は「心の内側を守るための情報」であり、Psycho-Pass 的な犯罪係数用途（監視・格付け・制裁）には用いない。

1. **用途制限**: self-care / 内省 / 研究 / フィクション表現のための指標であり、第三者が人物評価や権利制限に使うことを禁止する。
2. **本人中心**: スコアは本人の利益を最優先し、本人の同意と説明可能性が確保された範囲でのみ共有する。
3. **説明可能性**: danger/healing 数値は計算根拠を追跡できる形で保持し、潜在的なリスク犯罪係数的なラベリングではなくケアの手掛かりとして提示する。
4. **医療ごっこ禁止**: EQNet は診断ツールではない。深刻な状態が疑われる場合は医療・カウンセリングの専門家を案内する。
5. **データとプライバシー**: トラウマ・感情ログなどは極めてセンシティブな要配慮情報として扱い、保存・共有は最小限、本人のコントロールを優先する。

このガードレールを明文化することで、「危険と癒やしを扱える心のモデル」を守りつつ、Psycho-Pass 型のディストピア的運用を避ける。
