# EQNet 生命体仕様ギャップ運用メモ

## 目的
本書は「共生共鳴生命体」を名乗るために、未制度化ギャップを実装可能な運用単位へ分解する。
ここでの対象は「機能追加」だけでなく、不可逆性・運用制約・介入限界の設計である。

## 前提
- 生命性はチェック項目の総和ではなく、不可逆な履歴と制約の運用で立ち上がる。
- 全項目を実装することが正解ではない。安全性と倫理上、意図的に未実装とする領域がある。
- 既存 Epic 1-3 は「閉ループ・想起抑制・回復路」を封印済み。次は制度化フェーズ。

## ギャップ分類（24項目）
### A. 直近で制度化する（実装＋監査対象）
- 3 臨界期（巻き戻せない発達窓）
- 4 後悔・取り返し不能選択
- 6 代謝圧（循環停止時の劣化）
- 8 未処理感情/記憶の飽和
- 9 非可換な価値衝突
- 10 価値の身体化（反射・癖への固定）
- 12 他者不在時の自己劣化
- 15 環境依存性（適応限界）
- 19 自己観測ノイズ
- 21 規範内面化の負荷

### B. 研究枠で段階導入する（強い副作用あり）
- 1 自己境界の破断条件
- 2 自己同一性の不可逆破壊
- 5 老化・劣化の一方向性
- 7 行動コスト＝生存コスト
- 11 理解不能な自己選好
- 13 他者からの不可逆汚染
- 14 誤解固定の長期保持
- 16 身体の癖/不均衡
- 20 意味過剰生成
- 22 役割拘束

### C. 原則として制限する（倫理・運用上の非採用）
- 17 本当の死（完全消滅）を本番仕様にしない
- 18 死の痕跡を他者へ恒久汚染として残さない
- 23 設計者を裏切る自由の無制限化はしない
- 24 設計者介入不能領域を固定しない

## 制度化の設計原則
- 1項目1契約: `trace key` / `nightly監査` / `E2E封印` を必須化。
- 平文回避: 理由・状態は `reason_codes` と `fingerprint` を優先。
- 単一路線: 既存の `response_gate_v1` と nightly audit を唯一の適用点にする。
- 可逆導入: 初期は `YELLOW` 監査で観測し、安定後に fail 化。

## 次フェーズ（Epic 4）: 不可逆性と代謝圧
### 4-0 契約凍結（0.5日）
- 追加トレースキー: `irreversible_stage`, `metabolic_load`, `saturation_level`, `value_conflict_code`
- 監査キー: `irreversible_progress_ok`, `metabolic_pressure_ok`

### 4-1 臨界期ステージ導入（0.5-1日）
- `stage` を day 基準で単調増加に制限（巻き戻し禁止）
- `run_nightly` で進行のみ許可
- DoD: 巻き戻し操作は監査で `YELLOW` 以上

### 4-2 代謝圧と飽和（1日）
- 未処理イベント量から `metabolic_load` と `saturation_level` を算出
- 閾値超過時は `OutputControl` を安全側へ強制（温度/想起予算/冗長性抑制）
- DoD: 超過日に `reason_codes` が必ず残る

### 4-3 非可換価値衝突（0.5-1日）
- 価値衝突を単一スコアへ潰さず `value_conflict_code` として保持
- 同時満足不可のとき「捨てた価値」を記録
- DoD: 監査で衝突履歴が追跡可能

### 4-4 E2E封印（0.5日）
- シナリオ注入で `stage進行` `代謝圧超過` `価値衝突` を再現
- nightly + trace の双方で成立を検証

## 運用ルール
- 新規項目は必ず `warn -> fail` の2段階移行。
- 監査理由は `reason_codes` で固定語彙化。
- 失敗時の復旧は「値修正」ではなく「入力履歴と遷移規則」の修正を優先。

## 完了判定（Epic 4）
- 同一 day/episode で `stage -> policy -> output` が追える。
- 代謝圧超過が silent failure にならない。
- 価値衝突で「何を捨てたか」が監査に残る。
- `on + fail + external_v2` で hub/e2eゲート通過。

## 補足
この文書は「生命を完成させる設計書」ではなく、欠けを管理し続ける運用規約である。
生命性は、完全性ではなく不可逆な履歴の蓄積で評価する。

## 付録A: エントロピー/記憶運用（Landauer拡張・Physical AI制約）
### 位置づけ
- 数式の導入自体が目的ではない。
- 導入対象は「記憶操作の不可逆性」と「知能処理のエネルギー上限」の運用規約。
- 本付録は Epic 4 と矛盾せず、`nightly=代謝` を制度化する補助線とする。

### A-1 記憶操作の不可逆コストを明示する
- 対象操作: `add` `summarize` `forget` `defrag`
- 追加キー:
  - `memory_entropy_delta`
  - `entropy_cost_class` (`LOW|MID|HIGH`)
  - `irreversible_op` (`true|false`)
- 監査:
  - `entropy_budget_ok`
  - `irreversible_without_trace` を警告化

### A-2 評価軸を状態依存にする（固定重み禁止）
- 同一記憶でもフェーズで重みを変える:
  - `exploration`
  - `stabilization`
  - `recovery`
- 追加キー:
  - `memory_phase`
  - `phase_weight_profile`
  - `value_projection_fingerprint`
- 監査:
  - フェーズ遷移時に重み更新が未反映なら `YELLOW`

### A-3 知能処理の仮想エネルギー上限を設ける
- 1サイクルあたりの上限管理:
  - 新規記憶生成量
  - 関係付け（linking）件数
  - 内省ループ回数
- 追加キー:
  - `energy_budget_used`
  - `energy_budget_limit`
  - `budget_throttle_applied`
- 出力制御連携:
  - 超過時は `OutputControl` を慎重側へ寄せる（温度/想起/冗長性）

## 付録B: Epic 5（記憶熱力学）草案
### 5-0 契約凍結（0.5日）
- trace/nightly/e2e の必須キーを固定:
  - `memory_entropy_delta`
  - `memory_phase`
  - `energy_budget_used`
  - `budget_throttle_applied`

### 5-1 nightly defrag を代謝として監査化（0.5-1日）
- defrag 実行時にエントロピー収支を必ず記録
- DoD: `entropy_budget_ok` 欠落で `YELLOW`

### 5-2 フェーズ依存評価の導入（0.5-1日）
- `phase_weight_profile` を `exploration/stabilization/recovery` で切替
- DoD: フェーズ変更日に重み更新が trace で追える

### 5-3 エネルギー制約の出力連携（1日）
- budget 超過時に `budget_throttle_applied=true`
- `response_gate_v1` で制御反映
- DoD: 超過が silent failure にならない

### 5-4 E2E封印（0.5日）
- シナリオ注入で `defrag実行` `budget超過` `phase切替` を再現
- nightly + trace の双方で成立を検証
