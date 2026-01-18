EQNet Dream Specification
(Nightly派生・観測専用・疎結合)

1. 目的
本仕様は、EQNetに「夢（Dream）」を追加する際の設計指針を定義する。
夢は意味づけではなく、モデル挙動ログの別レンダリングとする。
記録・重み・導線を変更せず、Nightlyでのみ生成される派生アーティファクトとする。

2. 基本原則
- Nightly限定（Daytimeの挙動に影響を与えない）
- 読み取り専用（既存の記録・w・索引に一切書き戻さない）
- 生成物は分離保存（nightly.jsonにはメタのみ）
- プライバシー境界を優先（IDはmask/hash、本文は暗号化前提）
- 夢は診断レンダリングであり、意味づけは行わない

3. パイプライン位置
NightlyRunner
  - Collectors（既存）
  - ForgetfulnessHealth（既存）
  - DefragObserve（既存）
  - DreamStage（新規：読み取り専用）
  - Writer（nightly.jsonへメタだけ追記）

4. DreamStageの分離設計
4.1 DreamKernel（再活性化痕跡の抽出）
入力（read-only）
- defrag.observe（duplicate/conflict/reference/rollup など）
- forgetting.forgetfulness_health（delta/guard など）
出力
- DreamSeeds[]（夢の種）
  - seed_type: duplicate|conflict|bridge|budget_pressure|guard_overuse
  - evidence: 条件と数値（解釈はしない）
  - anchors: 参照ハッシュ（生ID禁止）
  - tone: 中立/軽微/強め（推定でも可、固定辞書不要）

4.2 DreamComposer（シーン構造化）
DreamSeedsを文章化せず、シーン構造に変換する。
- Scene: 登場物の配置 + 変形ルール
- constraints: 解釈禁止・結論禁止・評価禁止
- transforms: 反転/混成/接続 など

4.3 DreamRenderer（レンダリング）
シーンをプロンプトへ変換する。
- style: minimal|cinematic|poetic|clinical（設定で切替）
- output_format: text_prompt|story|bullet_scenes|image_prompt

5. 保存設計
nightly.jsonにはメタのみを記録する。

例（nightly.json）
dream:
  status: "generated" | "skipped"
  mode: "observe"
  privacy:
    id_privacy: "mask"
    content: "encrypted"
  count: 1
  seeds_summary:
    duplicate: 0
    conflict: 0
    bridge: 0
    budget_pressure: 0
  artifact_refs:
    - "dream://YYYY-MM-DD/0001#sha256:..."

夢本文は別アーティファクトに保存する（暗号化推奨）。

例（dream artifact）
{
  "dream_id": "YYYY-MM-DD_0001",
  "created_at": "...",
  "privacy": {"id_privacy":"mask","pii_redaction":"on"},
  "scene": {...},
  "prompt": "...",
  "refs": ["sha256:..."],
  "reasons": ["bridge_seed","guard_stable"]
}

6. 設定（runtime.yaml）
- dream.enable: true|false
- dream.mode: "observe"
- dream.privacy.id_privacy: mask|hash
- dream.storage.encrypt: true|false
- dream.generate_when:
    min_seeds: 1
    allow_if_all_zero: false
- dream.renderer.style: minimal|cinematic|poetic|clinical
- dream.max_per_night: 3
- dream.ttl_days: 30

7. 依存関係と安全性
- Defrag Stage1（観測-only）だけで成立する
- Stage2以降には依存しない
- 夢を学習や記憶更新へ戻さない（MVPは一方向）

8. 非目標（明示）
- 夢の内容を正解/不正解として評価しない
- 夢を学習ループに直接フィードバックしない
- 夢を人格の更新根拠にしない

9. MVP実装順
1) DreamStageを追加（observe-only）
2) seedsのみ出力（本文生成はしない）
3) seedsがある夜だけDreamRenderer(minimal)で1本生成
4) 夢アーティファクトを分離保存（暗号化）
5) 夢は診断レンダリングであり、意味づけ禁止と明記

10. 夢のプロンプト例（極小・概念）
あくまで雰囲気ですが、方向性としてはこんな感じです。
年代は書かない。原因も説明しない。なぜそれが出てきたかも語らない。

例:
いくつかの記憶が、同時に近づいたり遠ざかったりしている。
似ているものは並び、矛盾する感情は同じ場に存在する。
それらは統合されず、ただ関係だけが浮かび上がる。
物語を完成させる必要はない。
今日、再び触れられた状態たちが
同時に存在している様子を描写せよ。
