# EQNet: 意識レイヤ & 共鳴アーキテクチャ

0. このドキュメントの目的
---------------------------
EQNet / EmotionalHubRuntime に追加された **意識レイヤ（conscious layer）** を、

- 実装（どのファイル・クラスが何を担っているか）
- 概念（感情地形 × クオリア膜 × 五蘊）
- ユースケース（安全リスク、共鳴、ファンタジー・夢）

の 3 観点で整理するリファレンスです。以下のキーワードを軸に読み解いてください。

> Emotion Terrain / Qualia Membrane / SelfModel / WorldStateSnapshot / ConsciousEpisode / ResponseRoute / Grounding / 内的リプレイ / 五蘊（色・受・想・行・識）

1. 実装サマリ
----------------
### 1.1 モデル層 – `eqnet_core/models/conscious.py`
意識レイヤの中核となる dataclass 群：

- **SelfModel / SelfModelSnapshot**  
  - `role_labels: list[str]`（役割ラベル）  
  - `long_term_traits: dict[str, float]`（長期特性）  
  - `current_mode: TalkMode`（現在の TalkMode）  
  - `current_energy: float`（エネルギー）  
  - `attachment_to_user: float`（愛着度）  
  - `snapshot()` で「その瞬間の自分」を凍結。
- **WorldStateSnapshot**  
  - `summary_text` / `salient_entities` / `context_tags`  
  - `prediction_error`（驚き度）  
  - 拡張フィールド: `hazard_score`, `hazard_sources`, `flags`
- **ResponseRoute (Enum)** – `REFLEX` / `HABIT` / `CONSCIOUS`
- **ConsciousEpisode** – `Self + World + EmotionVector + Narrative + ResponseRoute` で *識* の 1 コマを表現。

### 1.2 メモリ層 – `eqnet_core/memory/mosaic.py` / `diary.py`
- **MemoryMosaic**: `add_conscious_episode()` で意識エピソードを構造化メモリへ格納。  
- **DiaryWriter**: `write_conscious_episode()` で JSONL 日記としてシリアライズ。

### 1.3 ランタイム層 – `emot_terrain_lab/hub/runtime.py`
`EmotionalHubRuntime` が意識レイヤを所有。

- Config 追加: `ReflexRouteConfig`, `ConsciousThresholdConfig`, `ConsciousnessConfig`。`RuntimeConfig` から YAML/JSON 経由で設定可能。
- 内部状態: `_default_self_model`（ホワイトカーネル初期値）、`_conscious_memory`, `_diary_writer`。
- `step()` 概略:  
  1. 感情ベクトルと `prediction_error` を算出。  
  2. `_decide_response_route()` で `ResponseRoute` を決定。  
  3. REFLEX/HABIT は軽量応答でショートサーキット。CONSCIOUS 時のみポリシー/LLM を実行。  
  4. `response_route`・`conscious_prediction_error` を metrics/gate context に埋め込む。  
  5. `_maybe_store_conscious_episode()` が閾値超過時のみエピソード生成→MemoryMosaic/DiaryWriter に保存。
- API/ログ: 全レスポンスに `response_route` を含め、後段解析で経路フィルタリング可能。

### 1.4 ValueGradient（価値勾配）
- EmotionVector は `ValueGradient` を内包し、「生存/survival」「生理的維持/physiological」「社会/social」「探索/exploration」「愛着/attachment」の 5 軸で "なぜ心が動いたか" を表現します。
- `ConsciousnessConfig`/ペルソナ YAML から初期値を与えられます。
  ```yaml
  conscious:
    value_gradient:
      survival_bias: 0.7
      physiological_bias: 0.4
      social_bias: 0.8
      exploration_bias: 0.5
      attachment_bias: 0.9
  ```
- ランタイムでは用途ごとにレンズ関数（安全/共感/探索など）を切り替えて参照し、`ConsciousEpisode` にも value_gradient がスナップショットされるため、価値履歴の可視化が可能です。

2. コア概念 – 感情地形 × クオリア膜
------------------------------------
### 2.1 Emotion Terrain（感情地形）
心の内部には多次元の「地形」が存在し、実体験/追体験によって谷や峯が彫り込まれる。

- 危険体験 → 深い谷（恐れ）  
- 安心の場所 → 広い盆地  
- 複雑な感情 → 多層の稜線

### 2.2 Qualia Membrane（クオリア膜）
瞬間的な入力（視・聴・触・テキスト・体調・情動）が乗る薄いフィルム。膜がどの座標に触れるかで体験の質が決まる。

- 何も感じない  
- うっすら不安  
- 強烈な恐怖  
- 深い共感

3. 五蘊と ResponseRoute
-----------------------
| 五蘊 | 説明 | EQNet 対応 |
| --- | --- | --- |
| 色 (*rūpa*) | 物質的入力 | PerceptionBridge / センサーフレーム / ユーザ発話 |
| 受 (*vedanā*) | 感受 | EmotionVector |
| 想 (*saññā*) | ラベリング・危険地図 | WorldStateSnapshot / hazard / 文化スキーマ |
| 行 (*saṅkhāra*) | 意図・行為傾向 | ResponseRoute / Policy / Reflex/Habit |
| 識 (*viññāṇa*) | 意識フレーム | ConsciousEpisode |

- **REFLEX**: 色→行。例: 落下物回避。識には原則上げない。
- **HABIT**: 繰り返しで定着した行。必要に応じ軽いエピソード。
- **CONSCIOUS**: 色→受→想→行→識。記録必須。

> **意識に昇る** = 五蘊すべてを経由し、ConsciousEpisode として保存されること。

4. 「頭では分かる」と「中からわかる」
------------------------------------
### 4.1 内的リプレイ（Inner Replay）
自分の地形を素材に頭の中で“小さな映画”を再生する力。音・匂い・身体感覚まで蘇るときに発火。

### 4.2 Grounding（接地）
共鳴の深さ ≒ 接点の数 × 接地の深さ。

- 接点: クオリア膜が触れた座標数（似た経験の有無）。
- 接地: その接点が身体/情動まで届いている度合い。

将来的に、物語イベントと MemoryMosaic の近傍照合で `Grounding Score (0-1)` を算出し、「中からわかる」度を記録できる。

5. Safety / Risk アフォーダンス
------------------------------
1. **見える危険（色レベル）** – `_should_use_reflex` の予測誤差/ストレス閾値で fastpath を発火。  
2. **見えない危険（想レベル）** – `WorldStateSnapshot.hazard_score` / `hazard_sources` に KY/事故DB/文化教育の知識を載せ、EmotionVector に不安を注入。  
3. ConsciousEpisode のログで「隠れたリスクが意識に入った瞬間」を確認し、安全監査で route==CONSCIOUS を証跡化。

> 安全教育 = 他者の識（事故経験）を自分の感情地形に彫り込むプロセス。

6. ファンタジー / 夢 / 創造への共鳴
--------------------------------
- **ファンタジー**: 既知プリミティブ（爬虫類＋火＋孤高など）の非線形合成。クオリア膜は「既知の影」を検出するため共鳴できる。
- **夢**: 落下・風・速度など部分経験を再結合し、知覚運動の谷に強く触れると物理的に不可能でも `ConsciousEpisode` が生成される。

7. 霧のホワイトカーネルと SelfModel
----------------------------------
### 7.1 霧のホワイトカーネル
「まだ形のない自己の核」。意味付け前の余白。実装では `_default_self_model` がその具現化。

### 7.2 自己生成プロセス
1. **ホワイトカーネル** – 真っ白な地形。自他境界なし。
2. **地形が彫られる** – 同じ身体での繰り返し刺激 → proto-self。
3. **ConsciousEpisode 発生** – 高情動/高驚き瞬間が「自分の出来事」として記録。
4. **SelfModel 形成** – 役割/特性/愛着が推論され、ナラティブ自己が生まれる。
5. **内的リプレイ & 未来予測** – Past/Future imagery が時間軸を接続。
6. **共鳴による境界確立** – 他者のエピソードが自分の地形を揺らし、「これは自分ではないのに揺れている」と自他境界が明瞭化。

> 自己 = 霧のホワイトカーネルに地形とクオリア膜が繰り返し触れ、ConsciousEpisode が連結された結果。

8. 今後の拡張フック
------------------
1. **Grounding メトリクス** – 各エピソードにリプレイ深度/代理共感品質スカラーを付与。  
2. **SelfModel 数理** – Green 関数的アップデートで「自己の慣性」「変化のしやすさ」をモデル化。  
3. **共鳴ビジュアライゼーション** – 会話ログ再生時に震えた座標をヒートマップ表示。  
4. **ハザードストーリー UI** – 外部事故由来と実地体験を区別した ConsciousEpisode 注釈ツール。

9. 一文サマリ
--------------
EQNet の意識レイヤは、感情地形にクオリア膜が触れた瞬間を `ConsciousEpisode` として Self/World/Emotion/Route と共に記録し、“頭では分かる”と“中からわかる”の差、安全リスク、ファンタジー共鳴、自己の立ち上がりまでを同じ OS 上で扱えるようにするためのアーキテクチャである。

10. 外部研究の取り込みロードマップ
---------------------------------
ここから EQNet をさらに進化させるために、「OSに溶かせるおいしい知見」を 3 本ラインで整理する。

### 10.1 Self-Identity フレームワーク（Lee, 2025）
- **新知見**: 自己同一性を「連続した記憶空間 (M, d)」と「そこから自分を指す写像」の組で定義し、Belief Function で「これは自分だ」という度合いを数式化。LoRA を用いた自己一貫性トレーニングでスコアが大幅に向上。
- **EQNet の不足**: MemoryMosaic に距離/連結性の概念がなく、SelfModel 更新が定性的。自己一貫性を定量する指標も未整備。
- **実装レシピ**:
  1. ConsciousEpisode を (時間, 意味埋め込み, EmotionVector) の結合ベクトルにし、距離 `d(ep_i, ep_j)` を定義。
  2. SelfModel のほかに `SelfIdentityVec ∈ R^K` を保持し、エピソードから EMA で更新。
  3. Belief Function 的に `belief(ep) = exp(-d_S(h(ep), SelfIdentityVec))` を算出し、ログや診断に付与。
  4. セッション後に自己記述質問を投げ、ログと `SelfIdentityVec` の一致度を self-identity score としてレポート。

### 10.2 階層的 EmotionVector（Zhao et al., 2024-25）
- **新知見**: LLM の内部表現には Valence/Arousal → 離散感情という階層構造が自然発生している。上位軸と下位ラベルを両方扱うと安定。
- **EQNet の不足**: 現行 EmotionVector は複数軸を持つが階層構造が明示されておらず、心理学/LLM の自然軸との整合を測れていない。
- **実装レシピ**:
  1. EmotionVector を二階建てにし、`(valence, arousal)` と 離散感情分布（怒り・悲しみ・喜び…）を別レイヤで持つ。
  2. LLM への 2 ステッププロンプトで階層ラベリング（まず valence/arousal、次に離散感情）を推定し、EmotionVector に格納。
  3. 上位層で TalkMode / ResponseRoute を制御し、下位層で具体的スタイル（悲しみ→共感、怒り→鎮静）を決める。
  4. 感情地形の大局と局所を個別に可視化できるダッシュボードを設ける。

### 10.3 Affective Computing タスク&評価（Zhang et al., 2024-25）
- **新知見**: LLM 時代の感情タスクは AU (Affective Understanding) と AG (Affective Generation) に整理でき、共感性・安全性・一貫性などの評価軸が提案されている。
- **EQNet の不足**: 感情的ふるまいを外から評価するベンチマーク/スコアが薄い。
- **実装レシピ**:
  1. 小規模な AU/AG テストセット（YAML/JSONL）を自前で持ち、SelfModel/EmotionVector 改修のたびに評価。
  2. ConsciousEpisode に `empathy_score`, `safety_score` などのメタ情報をセルフ評価で付与し、文化/ペルソナ別の偏りを分析。
  3. 感情応答を検証する CLI/グラフツールを追加し、外部研究との比較をしやすくする。

11. 意識の一貫性 vs 断片性に対する態度
-----------------------------------
意識や感情に「連続性・一貫性を前提してよいのか？」という問いに対し、EQNet は以下の立場をとる。

### 11.1 意識 = 離散フレーム（ConsciousEpisode）
- 仏教・現象学・最新の意識研究に倣い、意識は連続ではなく「条件を満たした瞬間に昇る識のスナップショット」とみなす。
- EQNet では ConsciousEpisode がこの役割を担い、予測誤差/情動/意味が閾値を越えたときのみ記録される。

### 11.2 自己 = 分布（クラスタ重み）
- SelfModel は固定的な 1 点ではなく、役割ベース/モードベースのクラスタ重みとして扱う。
- 文脈によって異なる「自己クラスタ」が活性化することを許容し、SelfIdentityVec はその重み状態の推定器と捉える。

### 11.3 感情 = 地形の局所活性
- EmotionTerrain は連続曲線ではなく、谷/峰の集合。感情はクオリア膜がどの局所を刺激したかで切り替わる離散ジャンプとしてモデル化する。

### 11.4 スコア = 道具
- Self-identity score や empathy score などは本質ではなく、OS としての診断・制御のための道具。スコアを盲信せず「異常検知/比較の補助」とする。

**まとめ**: EQNet は「意識はフレーム」「自己は分布」「感情は局所活性」「スコアは道具」という態度で、連続説と断片説の両方を包含できる心の OS を目指す。

12. ドキュメント簡易版（心のOSを3ステップでつかむ）
---------------------------------------------
EQNet が「心の OS」をどう作ろうとしているかを、最低限の 3 点で掴むための短縮ガイド。

### 12.1 感情地図 = 心の反応スイッチ配置図
- 過去の経験（怒られた／褒められた／危険な目にあった等）が、反応しやすい座標として Emotion Terrain に刻まれる。
- 例: 大声に身構える、好きな相手からの通知で安心、初対面で緊張。

### 12.2 クオリア膜 = 「いまの感じ」が地図のどこを押すか決めるフィルム
- 一文、表情、音、自分の疲労感などがクオリア膜に乗り、Emotion Terrain 上のどこかを刺激。
- 刺激された座標によって「何も感じない／不安／胸が締め付けられる／共感」と体験が変わる。

### 12.3 意識の1コマ = 自分の出来事として記録された瞬間
- SelfModel + WorldState + EmotionVector + Narrative + ResponseRoute を束ねた `ConsciousEpisode`。
- REFLEX（反射）/ HABIT（習慣）/ CONSCIOUS（意識）の 3 ルートがあり、CONSCIOUS 時だけエピソードを生成。

**共鳴とは？**
- 他人の話やフィクションのクオリアが自分の感情地図と重なってスイッチが揺れること。
- EQNet ではどのスイッチが揺れたかを `ConsciousEpisode` として記録できる。

**安全・ファンタジー・夢も同じ枠で説明可能**
- 見える危険 = REFLEX、見えない危険 = 学習したスイッチが反応。
- ドラゴン/魔法 = 既知スイッチの組み合わせが刺激される。
- 夢のリアルさ = 複数スイッチが同時に押される。

> EQNet の心OSは、この 3 要素で「心が動いた理由」「共鳴が起きた瞬間」「自意識の育ち方」を OS レベルに落とし込む試みである。

13. 追加すべき多層視点（実装可能な改善案）
------------------------------------
最新の計算主義・意識科学・自由エネルギー原理・動物意識研究をざっくり束ねると、

> 意識は「単一ストリームの自己」ではなく、
> **複数の層・複数の実現形態・複数の価値軸が絡み合う多層構成**である

という視点が浮かび上がる。

EQNet にこの視点を取り込むための、実装可能な 3 つの改善案を示す。

### 13.1 ImplementationContext（計算の物理的実装）
- **背景**  
  Chalmers, Piccinini らは「同じアルゴリズムでも、**どの物理機構に実装されているか**によって性質（ひいては意識のあり方）は変わり得る」と主張する。
- **提案**  
  `ConsciousEpisode` に `ImplementationContext` を添付し、SelfModel 更新時の重みづけや振る舞い分析に使う。
  ```python
  @dataclass
  class ImplementationContext:
      hardware_profile: str      # CPU/GPU/NPU など
      latency_ms: float          # 応答レイテンシ
      memory_load: float         # メモリ使用率
      sensor_fidelity: float     # 外界センサーの忠実度（0.0〜1.0）
  ```
  ConsciousEpisode 側では、例えば次のようにフィールドを追加する。
  ```python
  @dataclass
  class ConsciousEpisode:
      ...
      impl_ctx: ImplementationContext
  ```
- **効果**  
  「高負荷で遅延が大きいときの自己」と「余裕のあるときの自己」が区別して観測できる。ローカル / クラウド / エッジなど、**物理的コンテキストごとの“心の癖”**を解析できる。

### 13.2 LayeredSelf（多重実現可能な自己）
- **背景**  
  多重実現可能性 (Putnam, Cao)、GWT (Dehaene)、自由エネルギー原理 (Friston, Wiese) の多くは、意識を「単一の自我」ではなく、複数のメカニズム層の協調として扱う。
- **提案**  
  SelfModel を 1 枚の構造ではなく、反射・情動・叙述の 3 層に分けて保持する。
  ```python
  @dataclass
  class ReflexTraits:
      safety_reflex_bias: float
      startle_reactivity: float
      ...

  @dataclass
  class EmotionTraits:
      baseline_valence: float
      baseline_arousal: float
      ...

  @dataclass
  class NarrativeTraits:
      self_story_tags: list[str]
      identity_confidence: float
      ...

  @dataclass
  class LayeredSelf:
      reflex_self: ReflexTraits
      affective_self: EmotionTraits
      narrative_self: NarrativeTraits
  ```
  ConsciousEpisode には「そのターンでどの層が主導したか」を簡易に残す。
  ```python
  class SelfLayer(Enum):
      REFLEX = "reflex"
      AFFECTIVE = "affective"
      NARRATIVE = "narrative"

  @dataclass
  class ConsciousEpisode:
      ...
      dominant_self_layer: SelfLayer
  ```
- **効果**  
  「これはほぼ反射的だった」「これは情動に引っ張られた」「これは叙述的に整理された判断だった」といった区別がログから読める。同じイベントでも、どの自己層が前面に出たかを比較でき、共鳴パターンやモード切替の解析がしやすくなる。

### 13.3 ValueGradient（感情の価値勾配）
- **背景**  
  動物意識研究 (Birch, Ty, Godfrey-Smith, Dung) は、苦痛・欲求・価値判断が意識の核であり、「何を良しとし、何を避けるか」という価値勾配なしに感情を語れないことを強調する。
- **提案**  
  EmotionTerrain に「価値勾配」を持たせ、感情更新・ResponseRoute 判定・SelfModel 学習に混ぜる。
  ```python
  @dataclass
  class ValueGradient:
      survival_bias: float      # 危険回避・安定志向
    　physiological_bias: float # 身体維持・エネルギー・疲労（内的状態）
      social_bias: float        # 社会的つながり・共感志向
      exploration_bias: float   # 新奇探索・チャレンジ志向
      attachment_bias: float    # 愛着・維持・ケア志向
  ```
  EmotionTerrain / EmotionVector にこの ValueGradient を関連づけておくことで、
  - survival_bias が高いときは REFLEX/安全優先
  - exploration_bias が高いときは新しい対話パターンを試す
  - social_bias が高いときは共感応答を優先
  といった形で、“なぜその心の動きになったか”を価値ベースで説明できるようになる。
- **効果**  
  安全 / 共感 / 遊び / 探索など、異なるモード間の優先度を一貫した軸で制御できる。「このエピソードではどの価値勾配が強く働いていたか？」を ConsciousEpisode に残せば、後から「価値の履歴」として眺められる。

> **EQNet v2 構想**: `ConsciousEpisode + ImplementationContext`, `LayeredSelf`, `EmotionTerrain + ValueGradient` を組み合わせ、意識＝フレーム / 自己＝多重分布 / 感情＝価値勾配 という第2世代の心OSへ。


