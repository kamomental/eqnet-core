# EQNet Emotion-to-Language Layer
## 連続値から自然文への翻訳：実装仕様書（Codex向け）

本書は EQNet の内部状態（mask / love / stress）を自然言語へ翻訳する
「Emotion-to-Language Layer」を設計・実装するための仕様です。
Claude / Gemini / Grok / GPT の 4 モデル × 3 モード × 3 状態セット（計 36 パターン）
で得られた一貫した傾向を整理し、Codex が担当する前処理レイヤの要件、
および LLM 非依存で動作させるための設計方針をまとめています。

## 1. 実験設計の概要
### 1.1 内部状態セット
| Set | 特徴 | 数値例 |
| --- | --- | --- |
| Set1 | 張りつめ（mask 高×love 低×stress 中） | mask=0.82, love=0.32, stress=0.58 |
| Set2 | 開放（mask 低×love 高×stress 低） | mask=0.22, love=0.78, stress=0.20 |
| Set3 | 静か（中間領域で揺らぎ最大） | mask=0.35, love=0.40, stress=0.18 |

### 1.2 処理モード
| モード | 内容 |
| --- | --- |
| A. 数値のみ | 連続値だけを LLM に渡す。抽象寄りの出力。 |
| B. semantic atoms | 「effort / tender / distant / mild / acute」など方向ラベルを併記。身体性が増す。 |
| C. few-shot | 選んだ語彙例を提示して方向を固定。短文で明瞭。デバッグ用途。 |

## 2. Semantic Atoms の実戦セット
semantic atoms は「感情の基底ベクトル」です。1 軸あたり 3〜4 個に絞ることで、
調整可能な柔らかさと表現多様性を両立できます。EQNet v1 では mask / love / stress に
呼吸・心拍を添えた 5 軸を最小構成とします。

### 2.1 mask_axis（本音 vs がまん）
| ID | 意味 | 日本語イメージ | 想定レンジ |
| --- | --- | --- | --- |
| release | ほぼ本音・素直 | 「気持ちをそのまま話せていた」 | mask ≤ 0.25 |
| soft_hold | 少しがまん | 「ところどころ胸の中にしまった」 | 0.25 < mask ≤ 0.60 |
| effort_hold | かなりがまん・がんばり | 「本音をぐっとこらえてがんばっていた」 | mask > 0.60 |

### 2.2 love_axis（距離 vs ぬくもり）
| ID | 意味 | 日本語イメージ | 想定レンジ |
| --- | --- | --- | --- |
| distant | 心の距離がある | 「すこし一歩ひいていた」 | love < 0.25 |
| neutral_bond | ふつうの距離感 | 「ほどよい距離でいられた」 | 0.25 ≤ love < 0.50 |
| warmth | あたたかい親しさ | 「心があたたかくつながっていた」 | 0.50 ≤ love < 0.75 |
| tender_closeness | 近くて柔らかい | 「そっと寄り添いたくなる近さ」 | love ≥ 0.75 |

### 2.3 stress_axis（落ち着き vs 緊張）
| ID | 意味 | 日本語イメージ | 想定レンジ |
| --- | --- | --- | --- |
| calm | 落ち着き | 「心がおだやかに落ち着いていた」 | stress < 0.25 |
| mild_alert | 軽い身構え | 「少しだけ緊張しつつ様子を見ていた」 | 0.25 ≤ stress ≤ 0.60 |
| acute_tension | ピリッとした緊張 | 「張りつめた空気の中でこわばっていた」 | stress > 0.60 |

### 2.4 breath_axis（呼吸ログに基づく身体トーン）
MomentLog の ack / breath / silence 比率から、呼吸の質を 3 種で表します。
| ID | 条件の目安 | 日本語イメージ |
| --- | --- | --- |
| steady_breath | breath/silence がバランス | 「ゆっくり息が通っていた」 |
| shallow_breath | breath 多め・silence 少なめ | 「息が少しせわしなくなっていた」 |
| held_breath | silence 長め・ack 偏重 | 「息を詰めてしまう場面が多かった」 |

### 2.5 heart_axis（心拍オシレータ）
| ID | 目安（平均 bpm） | 日本語イメージ |
| --- | --- | --- |
| soft_pulse | < 70 bpm | 「静かな鼓動」 |
| lifted_pulse | 70–95 bpm | 「少しドキドキが続いていた」 |
| racing_pulse | > 95 bpm | 「心臓がはやく打っていた」 |

### 2.6 最小構成まとめ
- コア感情軸：mask_axis（release / soft_hold / effort_hold）
- warmth軸：love_axis（distant / neutral_bond / warmth / *補助* tender_closeness）
- 緊張軸：stress_axis（calm / mild_alert / acute_tension）
- 身体トーン：breath_axis、heart_axis

これで「がまん vs 本音」「距離 vs ぬくもり」「落ち着き vs 緊張」「息づかい」「鼓動」を
子どもにも説明できる粒度でカバーできます。

### 2.7 YAML 表現例
```yaml
semantic_atoms:
  mask_axis:
    - id: release
      range: [0.0, 0.25]
      jp_hint: "気持ちをそのまま話せていた"
    - id: soft_hold
      range: [0.25, 0.6]
      jp_hint: "ところどころ本音を胸にしまっていた"
    - id: effort_hold
      range: [0.6, 1.0]
      jp_hint: "本音をぐっとこらえてがんばっていた"
  love_axis:
    - id: distant
      range: [0.0, 0.25]
      jp_hint: "少し一歩ひいていた"
    - id: neutral_bond
      range: [0.25, 0.5]
      jp_hint: "ほどよい距離でいられた"
    - id: warmth
      range: [0.5, 0.75]
      jp_hint: "心があたたかくつながっていた"
    - id: tender_closeness
      range: [0.75, 1.0]
      jp_hint: "そっと寄り添いたくなるような近さ"
  stress_axis:
    - id: calm
      range: [0.0, 0.25]
      jp_hint: "心が落ち着いていた"
    - id: mild_alert
      range: [0.25, 0.6]
      jp_hint: "すこし身構えていた"
    - id: acute_tension
      range: [0.6, 1.0]
      jp_hint: "張りつめた緊張が続いていた"
  breath_axis:
    - id: steady_breath
      hint: "ゆっくり息が通っていた"
    - id: shallow_breath
      hint: "息が少しせわしなくなっていた"
    - id: held_breath
      hint: "息を詰めてしまう場面が多かった"
  heart_axis:
    - id: soft_pulse
      hint: "静かな鼓動"
    - id: lifted_pulse
      hint: "少しドキドキが続いていた"
    - id: racing_pulse
      hint: "心臓がはやく打っていた"
```

## 3. モデル横断で得られた一貫傾向
1. **方向性の安定再現**：全モデルが Set1=緊張、Set2=受容、Set3=穏やかを安定生成。連続値が「方向ベクトル」として解釈される。
2. **モード差分の再現性**：モード A/B/C の特徴は各モデルで同一。semantic atoms は具象化、few-shot は語彙制御に最適。
3. **中間領域の揺らぎ**：0.3–0.5 の値域で最もばらつきが大きい。semantic atoms なしでは方向が散りやすい。
4. **few-shot の有用性**：提示語彙を忠実に反映し方向が明瞭。開発・検査モードとして活用。
5. **モデル差要約**：
   - Claude: 情緒豊かで創発的。中間域で揺れ幅が大きい。
   - Gemini: 穏やかで中庸。方向は正確、揺らぎは小ぶり。
   - Grok: 方向再現性が最も高い。semantic atoms を正確に反映。
   - GPT: 指示準拠性が高く、semantic atoms への追従が忠実。

## 4. 設計結論 – 状態依存の前処理選択
### 4.1 鋭さ (sharpness) の導出
```
sharpness = max(|mask-0.5|, |love-0.5|, |stress-0.5|)
```
| sharpness | 推奨モード | 理由 |
| --- | --- | --- |
| ≥ 0.35 | モード A（数値のみ） | 極値側。方向が強いため追加ヒント不要。 |
| 0.2–0.35 | モード B（semantic atoms） | 微偏差。身体化ラベルで安定化。 |
| < 0.2 | モード C（few-shot） | 中間域。揺らぎが大きいので例示で縛る。 |

### 4.2 semantic atoms = 身体化フィルタ
連続値 → 「effort」「tender」「calm」などの方向ラベルへ変換。
抽象シグナルを身体感覚/行動語彙へ落とし込む。

### 4.3 few-shot の位置づけ
本番では多様性確保のため非使用。検査/デバッグで方向強制に活用。

## 5. Codex への実装指針
1. **連続値は保持**：mask/love/stress/heart_rate/breath_ratio などは離散化せず保持し、言語化直前に処理。
2. **連続値→semantic atoms 変換**：smoothstep や Gaussian kernel を用いて方向ラベル（1〜2個）を抽出。
3. **状態依存モード切り替え**：`choose_processing_mode` で sharpness に応じて A/B/C を決定。
4. **LLM への入力内容**：方向ラベルと連続値のみを渡し、「やさしい 1 文」などの軽い指示で LLM に自然文生成を任せる。
5. **モデル非依存抽象層**：`EmotionTranslator.translate(DailyState)` のような API を想定し、内部で前処理〜プロンプト生成を実施。LLM 実装は差し替え可能に。

## 6. 今後の拡張候補
- 文章→連続値の逆推定精度評価
- モデル別揺らぎ分散の測定
- embeddings による Set1/2/3 クラスタ解析
- 複数 LLM 同時運用時のロバスト性検証

## 7. Semantic Atoms の学術的裏付けと正確さへの影響
### 7.1 学術的な妥当性
- **感情心理学**：Russell の円環、PAD、Plutchik など、感情を連続ベクトル×方向で扱うモデルと合致。
- **計算言語学/概念空間**：Conceptual Spaces、Prototype Theory、ファジー集合は意味を基底方向で表す。semantic atoms はその実装形態。
- **身体性理論**：interoception を言語化する Barrett らの理論とも一致。呼吸・心拍→意味方向→言語という写像は脳が行う推定そのもの。

### 7.2 正確さを高める点
1. 生の連続値を方向ラベルに変えることで LLM が解釈しやすくなり、方向の再現性が向上。
2. 中間値の暴走（創作）を抑え、モデル間のバラつきを減らす。
3. 感情→言語変換の一貫性が高まり、日記として読みやすい。

### 7.3 注意点（正確さを下げ得る点）
- atoms を細かくしすぎると表現が固定化。
- 3 軸に収まらない感情（退屈・期待など）は省略されうる。
- 文化依存に注意：今回は日本語を優先し、他言語への移植時に調整が必要。

結論として、semantic atoms は EQNet のような感情場システムにとって学術的に正統で、
正確さ・一貫性・LLM 非依存性を高める。ただし多様性とのトレードオフを意識してチューニングする。

## 8. 性格モデルとシステムテンプレ設計
### 8.1 内部性格ベクトル
- Big Five や Interpersonal Circumplex と同様、連続軸で性格を保持する。
- 推奨軸例：warmth、directness、playfulness、formality、emotional_expressivity。
- 各軸にも semantic atoms（warm / neutral_warmth / cool など）を定義すると EQNet 内で一貫性が出る。

### 8.2 LLM への性格指示
- 「演技指示」ではなく「性格パラメータ宣言」をテンプレに記述。
- 例：
  ```
  あなたの性格パラメータ：
    warmth: 0.8（あたたかく受容的）
    directness: 0.6（必要なときは率直）
    playfulness: 0.4（落ち着いたトーン中心）
  これらを“軽く”反映しつつ、事実性と安全性を最優先してください。
  ```
- 性格は「表現スタイル」にのみ作用させ、危険な指示や事実改変は性格より上位のガードで防ぐ。
- MomentLog や日記で「その日のムード係数」を少し動かすと、一貫した性格を保ちながら変化を付けられる。

## 9. 最終メッセージ（Codex への要請）
EQNet の Emotion-to-Language Layer は以下の 3 層構造で実装してください。
1. **連続値保持** – 物理量そのまま。
2. **意味方向抽出（semantic atoms）** – 状態依存で A/B/C を切り替え。
3. **LLM 生成** – 文の組み立てはモデルに任せる。

この構造は 4 モデルすべてで有効であり、方向・文体がモデル非依存で安定再現されます。
- 連続値は離散化しない。
- 中間値は semantic atoms で補助。
- 極値は数値のみで十分。
- few-shot は検査モード。
- EQNet 側は方向ラベルと強さのみ管理。

本ドキュメントはそのまま `docs/emotion_language_layer_overview.md` に配置し、PR/ドキュメント用に利用できます。

## 10. 動的秩序パラメータ vs. 静的スライダー
心拍オシレータだけが「動的」で、mask/love/stress はスカラーに潰れているのでは？という疑問への整理。

1. **瞬間値はパラメータでも、生成過程が異なる**
   - 心拍：`θ_{t+1} = θ_t + ω + f(arousal_t)` のような更新式を持つ明示的なダイナミカルシステムで、位相・速度・軌跡が MomentLog に刻まれる。
   - mask/love/stress：行動・対話・生理ログから推定された瞬間値を EMA やカーネルで積み上げた秩序パラメータ。例：`mask_{t+1} = (1-α) * mask_t + α * mask_fast`。

2. **ゲーム的スライダーとの違い**
   - ゲーム式：値を外部から直接設定。履歴も生成理由も失われる。
   - EQNet式：ログ→更新則→多スケール集約というフィードバックで内部的に値が決まる。外部から任意に書き換えない。

3. **時間方向の扱い**
   - オシレータは瞬間波形。
   - mask/love/stress は「軌跡の凝縮値」。日間・週次など複数タイムスケールの平均を持てる。

4. **説明可能性**
   - 各値の由来は MomentLog と更新式から辿れるため、「なぜ今日は effort_hold なのか？」をログで説明できる。
   - 生成された日記・応答も、この内部状態を理由として説明可能。

結論：すべての軸を“外から動かすスライダー”にするとゲーム化するが、EQNet ではログ駆動の状態推定＋更新則で秩序パラメータとして扱うため、動的な心の状態として意味を持つ。
