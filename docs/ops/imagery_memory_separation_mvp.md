# imagery_memory_separation_mvp.md（Codex投入用 1枚メモ）

## 目的
- 空想/検討（thought/imagery）と経験（experience）を分離し、誤帰属（cryptomnesia 等）を設計で抑止しつつ監査可能にする。
- 既存の IMAGERY 実装（`ReplayMode.IMAGERY` / `future_imagery` log / TalkMode 上げ）を壊さず、運用を硬くする。

## 1) memory_kind 強制 enum 化
- 対象: `devlife/mind/replay_memory.py`, `emot_terrain_lab/hub/hub.py`（必要なら `schema/*`）

変更:
- `memory_kind` を Enum/Literal 化:
`experience | imagery | hypothesis | borrowed_idea | discussion | unknown`
- 書き込み時に enum バリデーション
- 不正値は `unknown` に正規化
- その際、監査イベントを `think_log` に記録（例: `MEMORY_KIND_NORMALIZED`）

受け入れ条件:
- enum 外値が永続保存されないテスト
- 既存ログ読み込み互換（欠損/未知は `unknown`）

## 2) 保存レイヤ分離（experience と thought系）
- 対象: `devlife/mind/replay_memory.py`, 呼び出し元 `emot_terrain_lab/hub/hub.py`

変更:
- `experience_log` と `think_log` を分離（ファイル分離 or 物理ストア分離）
- `imagery | hypothesis | borrowed_idea | discussion` は think 側へ
- `experience` は act 側（`experience_log`）へ

受け入れ条件:
- 同一 `turn_id` で think/act を結合できる
- Recall は両方検索可能だが既定提示は `experience` 優先
- 片側欠損（think のみ/act のみ）でも監査可能（壊れない）

## 3) 昇格ガード（thought -> experience）
- 対象: `emot_terrain_lab/hub/hub.py`（昇格・重み更新・保存経路）

変更:
- thought 系は直接 experience の重み更新/昇格を禁止
- `observed/executed` 等の検証イベントでのみ experience へ昇格許可
- 例: `promote_memory(thought_id, evidence_event_id)` のように証拠リンク必須化

受け入れ条件:
- thought 単独では experience 昇格しないテスト
- 検証イベントありの場合のみ昇格するテスト

## 4) 監査ログ必須分離（think_log.jsonl / act_log.jsonl）
- 対象: `emot_terrain_lab/hub/runtime.py`, `emot_terrain_lab/hub/hub.py`

変更:
- `think_log.jsonl`（検討ログ）
- 候補 / 却下理由 / `memory_kind` / `replay_source` / 監査イベント
- `future_imagery` は think 側に寄せる
- `act_log.jsonl`（実行ログ）
- 実際の発話・行動・決定（experience）
- 全ターンに `turn_id` を必須付与

受け入れ条件:
- すべてのターンに `turn_id` が付く
- think/act 片側欠損時も「何が起きたか」追跡可能（落ちない・破綻しない）

## 5) 出所再確認フック（Recall時）
- 対象: `emot_terrain_lab/memory/reference_helper.py` 付近（Recall/引用/参照の導線）

変更:
- 想起時に source 再判定:
`self | other | imagery | uncertain`
- 外部出力はイベント型（スコア非表示）:
`SOURCE_FUZZY`, `DOUBLE_TAKE`（など）
- 数値確率は内部保持のみ（UI/ログ表面はイベント中心）

受け入れ条件:
- 誤帰属疑い時にイベントが出るテスト
- UI/外部ログに確率値が露出しないテスト（必要なら）

## 6) imagery パラメータ外部化（ハードコード除去）
- 対象: `emot_terrain_lab/hub/runtime.py`（intention 生成箇所）, `config/runtime.yaml`（or config）

変更:
- `target_valence=0.6`, `target_love=0.7` を設定へ移動
- 設定優先順位を統一: `env > yaml > default`
- コードから固定値を除去（完全排除）

受け入れ条件:
- 固定値がコード上に存在しない
- 設定変更で挙動が変わるテスト

## 7) apply_imagery_update TODO 解消（v1）
- 対象: `eqnet/runtime/policy.py`

変更:
- `imagined_traj` を実利用（v1 は最小で可）
- 例: `delta_valence`, `delta_love`, `delta_fog` を抽出
- 方向特徴に応じて update を切替
- `delta_fog` が改善方向なら fog 優先補正
- `delta_valence/love` が改善方向なら emotion/talk mode 側へ反映

受け入れ条件:
- `imagined_traj` の方向に応じて更新が変わるテスト
- 既存テスト互換（回帰なし）

## 8) テスト追加（最小セット）
- 対象: `tests/`

追加:
- `memory_kind` enum 制約
- thought/imagery 非昇格保証
- `borrowed_idea -> self` 誤帰属疑い時の監査イベント記録
- 設定化した imagery ターゲット反映

受け入れ条件:
- 既存回帰なし
- 新規テストが CI で安定通過

## 実装順（推奨）
- `(6) -> (2) -> (1) -> (3) -> (4) -> (7) -> (5) -> (8)`
- 先にハードコード排除 `(6)` で設計の入口を整える
- 次に保存分離 `(2)` で事故を止める
- その上で enum 拘束 `(1)` と昇格ガード `(3)` で運用を硬くする
- 監査ログ `(4)` は分離決定後に入れる
- `imagined_traj` 活用 `(7)` は安全に機能追加
- 出所再確認 `(5)` は体験仕上げ（fastpath 的違和感）
- 最後にテスト `(8)` を固める

## docs 配置案
- 新規: `docs/ops/imagery_memory_separation_mvp.md`
- 追記リンク候補:
- `docs/ops/README.md` に「Memory Source Separation (MVP)」リンクを追加
- `docs/memory/README.md` に関連仕様としてリンクを追加

## JSONL 最小スキーマ例（貼るだけ）

`think_log.jsonl`（1行例）:
```json
{"ts":"2026-02-14T10:12:33Z","turn_id":"t-000123","session_id":"s-01","memory_kind":"imagery","replay_source":"future_imagery","candidate_id":"cand-7","decision":"rejected","reason":"low_evidence","audit_event":"SOURCE_FUZZY","meta":{"norm":"MEMORY_KIND_NORMALIZED"}}
```

`act_log.jsonl`（1行例）:
```json
{"ts":"2026-02-14T10:12:34Z","turn_id":"t-000123","session_id":"s-01","memory_kind":"experience","action_id":"act-991","utterance":"了解、ここからは実測ベースで進める。","executed":true,"evidence_event_id":"obs-55","source_confirmed":"self"}
```

