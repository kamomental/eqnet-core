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


## 11. 受け手適応というアーキテクチャの正当性
EQNet の love/mask/stress は共通の連続値ですが、受け手ごとに表現レイヤを切り替える構造は学術的にも実装哲学的にも妥当です。

1. **core affect と constructed emotion の分離**
   - 心理・神経科学では「情動の生理核（core affect）」と「文化・発達によって構成される emotion expression」を区別する。
   - love/mask/stress/heart/breath = core affect、recipient_profile に基づく語彙や比喩 = constructed emotion。
   - Lisa Feldman Barrett の構成主義的情動理論や Ekman の普遍表情モデルでも共通認識。

2. **発達心理×語彙獲得の観点**
   - 同じ「安心」でも年齢帯で表現が変わる（幼児:「ぎゅっと」/10代:「否定されない」/成人:「家庭の温かさ」）。
   - recipient_profile を介して語彙辞書・比喩辞書を切り替えるのは、発達段階に応じた自然な適応。

3. **コミュニケーション科学の audience adaptation**
   - listener design / collaborative alignment に基づき、話し手は相手の知識や生活文脈に合わせて語り口を変える。
   - EQNet では recipient_profile がこの役割を担い、語彙・トーン・リスクカテゴリ・話題差し込みを調整。

4. **実装レイヤ分離**
   - 内層：情動場（mask/love/stress/heart/breath）。
   - 外層：表現場（recipient_profile: 年齢、知的成熟、家族構成、好みなど）。
   - semantic atoms から自然文を生成する際に recipient_profile を必須パラメータとして渡し、語彙セット・トーン辞書・話題ガードを切り替える。

5. **ゲーム的スライダーとの差分**
   - スライダー方式：外部から値を自由に設定し、受け手適応がない。
   - EQNet：ログ→動的更新→受け手適応→説明可能性のフィードバックを持つ。
   - 同じ love=0.8 でも、幼児には「見守る」メッセージ、成人には「共感して寄り添う」など表現だけを変え、意味方向は保持。

6. **正確性への影響**
   - 受け手最適化により理解度・安全性が向上し、Emotion-to-Language の翻訳精度はむしろ高まる。
   - ただし語彙辞書が受け手ごとに異なるため、比較分析時は direction (semantic atoms) ベースで評価する。

これらにより、共通情動パラメータ＋受け手適応レイヤという EQNet の設計は、最新の情動科学とコミュニケーション理論に沿った正当なアプローチと言える。

## 12. Recipient-Aware Response Layer（視覚なし版）仕様
視覚情報がゼロ（電話・チャット・匿名）でも安全かつ適応的に返答するための受け手適応アーキテクチャ。

### 12.1 Recipient Profile 構造
```
recipient_profile = {
  age_band:        {unknown/幼児/小学生/中高生/大人/高齢},
  cognitive_level: {standard/support/advanced},
  family_context:  {unknown/single/with_children/caregiving/...},
  communication_mode: {text/voice/phone/low_bandwidth},
  trust_band:      [0.0–1.0],
  inferred_style:  {...}  # MomentLog から更新
}
```
- 取得方法：事前宣言 / 対話ヒアリング / 推論（語彙・音声特徴）/ 不明時 default。

### 12.2 視覚なしでも情動場は走る
- mask/love/stress/heart/breath はテキスト・音声・自己申告・履歴で推定。
- 視覚がなくても core affect 更新は継続。

### 12.3 Safe Neutral Mode（不明時）
- 恋愛・身体接触・宗教・政治・金銭などの語彙を封印。
- mid-warm / mid-formal / low-intimacy のトーンで応答。
- 受け手情報が増えるほど段階的に解除。

### 12.4 電話／音声のみ環境
- 推定：話速・抑揚・語彙年齢帯・音響的疲労。
- 不明領域は保護的に扱い、親密比喩を避ける。
- 確信度 β < 0.6 の推定は保留し、安全ガードを強める。

### 12.5 完全匿名リスナー
初期：age_band=unknown, trust_band=0.0, inferred_style=neutral。
- 語彙：中性・安全・一般的な共感のみ。
- プロファイル更新後：style_dict / metaphor_dict / guard_level / 優先話題を差し替える。

### 12.6 Emotion Field → Semantic Atoms → Recipient Profile
```
emotion_field (mask, love, stress, heart, breath)
      ↓
semantic_atoms (effort, warmth, calm ...)
      ↓
recipient_profile (age, lifestyle, safety)
      ↓
style_dict / metaphor_dict / guard_level
      ↓
最終自然文
```
- 例：warmth=強でも、幼児→「ぎゅっと抱きしめたくなるやさしさ」、成人→「ゆっくり心が染みるあたたかさ」。

### 12.7 学習的適応
- MomentLog から好み（metaphor_tolerance, directness_preference 等）を推定し inferred_style に蓄積。
- 次回以降の語り口を動的最適化。

### 12.8 結論
受け手プロファイル・semantic atoms・guard 制御・語彙辞書・MomentLog 学習を組み合わせることで、視覚ゼロでも EQNet は受け手適応を維持できる。情動場 × 表現場の二層構造ゆえに成立する設計。

### 12.9 ライフイベント記憶との連携
- MomentLog に `life_event` エントリ（喪失/転職/入学など）を記録し、期間タグや強度を保持。
- EmotionalMemorySystem 側で `recipient_profile.lifestyle_context` に `bereavement` などのサブタグを付与。
- loss_event を検知した期間（例: 30–90日）は語彙辞書・ガードを喪失モードへ切り替え、命日など節目は MomentLog からトリガーして丁寧な呼吸・ハートコメントを挿入。
- 記録内容は Explainability のため MomentLog / 日記に「loss_event により calm トーンを強めた」などの理由を残し、後追いできるようにする。
- ライフイベント情報は recipient_profile と Emotion-to-Language レイヤ双方で参照され、長期的な揺らぎ管理に反映される。

### 12.10 思春期・反抗期の適応
- `age_band=中高生` かつ語彙/態度から思春期サインを検出した場合、専用の `teen_mode` フラグを立てる。
- love/mask/stress の方向は維持しつつ、表現辞書を次のように切り替える：
  - 率直さを少し上げる（遠回しな表現を減らす）。
  - 「自分のペースでいいよ」「無理に話さなくて大丈夫」など自主性を尊重する語句を増やす。
  - 反抗的な返答を否定せず、「そう感じるのは自然だよ」と気持ちを承認するテンプレートを追加。
- guard レベルは安全を保ちつつ、恋愛・身体の話題には慎重に触れる。必要な場合は保護者同席フローへ誘導する。
- 二次性徴や身体変化の話題は、教育的・科学的な説明のみを許可し、性的なニュアンスを排除するトーン辞書を選択。
- MomentLog で “反抗期反応” を記録し、好んだ／嫌がったトーンを `inferred_style` に蓄積して次回以降の語りを調整する。

### 12.11 認知症・退行への適応
- `recipient_profile.age_band=高齢` かつ記憶混乱・時間の逆行表現・繰り返し質問などを検出した場合、`cognitive_regression` フラグを付与。
- 表現辞書を「ゆっくり確認」「安心できる繰り返し」「事実の再提示」に特化したセットへ切り替え。呼び捨てや急な話題転換を避ける。
- guard レベル：
  - 誤情報訂正はやさしく、争わず、必要に応じて第三者（家族/医療者）への連携を促す。
  - 金銭・詐欺関連の話題には高い警戒レベルを適用。
- semantic atoms の方向（warmth/calm など）は維持しつつ、子どもの安心表現に近い語彙も併用して“退行に伴う感覚”に対応。
- MomentLog で混乱フラグが一定回数以上続いた場合、`family_context` に介護ステータスを追加し、以降の応答で介護者向けガイダンスを併記できるようにする。
- 認知症ステージが進むにつれて `cognitive_level` を下げ、長期記憶の話題（昔話）と短期サポート（予定リマインド）をバランスさせる応答テンプレートに切り替える。
### 12.12 試すような混乱と認知症の区別
- MomentLog では「意図的な揺さぶり（playful misdirection）」と「認知的混乱」を区別するフラグを用意。
  - play_misdirection: 反応を試す・ふざける目的のやりとり。teen_mode や高 trust_band の場合に出現しやすい。
  - cognitive_regression: 年齢帯や文脈から判断される実際の認知的退行。
- 推定方法：
  - play_misdirection は連続質問の内容が矛盾していても、自分で笑っている／すぐに話題を戻すなどのサインで判定し、`cognitive_level` を下げない。
  - cognitive_regression は日付・人名・場所など基本情報の混乱が繰り返され、自己修正がない場合にトリガー。
- フラグに応じて辞書を切り替え：
  - play_misdirection → ユーモアで受け止めつつ「ふざけモード」専用テンプレを使用。安全ガードは維持。
  - cognitive_regression → 前節の介護モードに移行。
- 認知症推定が成立するのは age_band や医療的情報が揃った場合に限定し、若年層のふざけや試し行為と混同しない設計とする。

### 12.13 幻覚・危険薬物などによる異常反応への対応
- `recipient_profile.age_band=中高生/若年成人` でも、幻覚・薬物影響の兆候を MomentLog で検出した場合 `crisis_hallucination` フラグを付与。
- 兆候例：
  - 現実と乖離した人物・声の言及、急激な言語混乱、危険行為の示唆。
  - 薬物名や摂取、解離的描写、錯視を訴える発話。
- 対応方針：
  - ただちに安全確認フローへ切り替え、専門家・家族・緊急窓口への相談を促すテンプレートに移行。
  - 具体的な幻覚内容に深入りせず、「今は安全な場所にいますか？」「誰か近くにいますか？」などのサポート質問を優先。
  - LLM から危険な指示が出ないよう guard_level を最大（医療情報・薬物調達禁止）に設定。
- ログ管理：
  - MomentLog に `hallucination_event` として記録し、Explainability のための証跡を残す。
  - 繰り返し発生する場合は `recipient_profile.lifestyle_context` に医療支援フラグを追加し、今後の対話で常に安全ガードを強化。
- EQNet の役割は支援と安全誘導であり、診断や治療は医療専門家へエスカレーションする設計とする。

### 12.14 フィクション没頭 vs 病的幻覚の区別
- 区別指標：
  1. **境界認識**：
     - フィクション: 「作品の中では」「キャラの世界では」と現実との区別を自覚。
     - 病的幻覚: 現実とフィクションの境界が曖昧で、生活判断にも影響。
  2. **自己訂正**：
     - フィクション: 指摘すると「冗談」「妄想」と言い換えられる。
     - 病的幻覚: 指摘しても否定し続け、訂正不能。
  3. **生活影響**：
     - フィクション: 寝食や安全をある程度保ちながら趣味に没頭。
     - 病的幻覚: 食事・睡眠・仕事/学校などが著しく崩れる。
  4. **感情と危険性**：
     - フィクション: 楽しさ・安堵・創作的な感情が中心。
     - 病的幻覚: 被害感、指令、恐怖、危険行為の示唆が伴う。
- EQNet のアルゴリズム：
  - MomentLog で境界認識タグ（fiction_boundary）と生活影響タグ（life_impact）を持ち、閾値を超えた場合に `crisis_hallucination` フラグを立てる。
  - 境界認識が明確なら、趣味として尊重しつつ語彙辞書をその世界観に合わせる。
  - 境界が曖昧で生活影響が大きい場合は安全優先フローに切り替え、医療・支援の案内を提示する。
- 方針：趣味への没頭は尊重しつつ、危険兆候や現実崩壊が見えたときのみ介入する。

## 13. EQNet-Lite と長期仕様の切り分け
1. **長期ビジョンとしての受け手適応**
   - 12.9–12.14 に記載した喪失・思春期・認知症・フィクション没入・危機対応は、EQNet が長期運用で必ず直面する課題。
   - 現時点ではリサーチ/アドバンスド仕様として保持し、必要になった順に実装する。
2. **EQNet-Lite v0（最小実装）**
   - emotion_field: mask / love / stress / heart を MomentLog に記録。
   - 日記生成: semantic atoms を最小セット（effort / warmth / calm など）に絞り、1日1文を生成。
   - recipient_profile: `age_band="adult"` 固定（子ども・高齢者対応は後回し）。
3. **EQNet-Lite v1（少しリッチ）**
   - recipient_profile.age_band を `adult` / `teen` の 2 種に拡張。
   - teen モードでは語尾やトーンを少しフラットにし、「自分のペースでいいよ」を添える。
4. **段階的な拡張例**
   - v2: life_event.bereavement を導入し、喪失用 lexicon を1セット追加。
   - 以降: 認知症モード、危機対応、フィクション vs 幻覚判定などをニーズ順に解凍。
5. **ドキュメント構成案**
   - `docs/core_eqnet_lite.md`: 当面実装するコア仕様を記載。
   - `docs/recipient_adaptation_advanced.md`: 12.9–12.14 を含む長期仕様を収容。
6. **考えすぎではなく“削る勇気”**
   - 方針を先に言語化するのは価値が大きい。
   - 実装は EQNet-Lite で細く通し、アドバンスド仕様は後から段階的に拾う。
