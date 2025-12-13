# EQNet Conscious Layer Updates (Dec 2025)

このメモは `emot_terrain_lab/hub/runtime.py` と `eqnet_core/models/conscious.py` に入った最新の「心＝力学系」アップデートをまとめたものです。

## 1. ForceMatrix × Persona Override
- YAML の `conscious.force_matrix` を 5×3 行列（ValueGradient → SelfLayer）として読み込み、`ForceMatrix.from_mapping()` で dataclass 化。
- persona 単位で `value_gradient` と `force_matrix` を上書きできるので、キャラクターごとの attractor バイアスを if なしで調整可能。
- `DevelopmentConfig` が有効な場合は `_current_force_matrix()` で Reflex/Affective/Narrative 行を時間（成熟度）でスケールし、Reflex 支配 → Affective → Narrative の発達曲線を連続的に再現。

## 2. ImplementationContext Damping（摩擦）
- `DampingConfig` に `latency_penalty_cap` と `tag_bumps` を追加。レイテンシ由来ペナルティは `min(k * excess, cap)` で clamp、タグ起因バンプは YAML のディクショナリから SelfLayer 単位で連続加算。
- `tag_bumps` 例:
  ```yaml
  conscious:
    damping:
      latency_L0_ms: 200
      latency_k_narr: 0.002
      latency_k_reflex: 0.001
      latency_penalty_cap: 0.6
      tag_bumps:
        safety:
          REFLEX: 0.05
        support:
          AFFECTIVE: 0.05
        brainstorm:
          NARRATIVE: 0.05
  ```
- これにより ImplementationContext（遅延・タグ）を if 分岐ではなく「スコアの連続調味料」として扱える。

## 3. SelfForce logging（raw/damped 二重ログ）
- `_choose_self_layer()` で計算した base_scores（raw force）と damping 後の scores（damped force）を `SelfForceSnapshot` に格納。
- ConsciousEpisode / Diary に `self_force`（damped）と `raw_self_force` を保存。Raw ≠ Damped の差分が「葛藤／迷い」を示す観測点になる。

## 4. BoundarySignal（位相切り替えの観測）
- `_compute_boundary_signal()` で以下の連続量を合成し、0..1 のスコアに正規化。
  - `prediction_error_delta`
  - `force_field_delta`（damped force のベクトル差分）
  - `winner_flip`（SelfLayer の切替）
  - `raw_damped_gap`（摩擦による抑制量）
- `BoundaryConfig` で重み・正規化係数・threshold を調整可能。スコアは `ConsciousEpisode.boundary_signal` としてログ出力される。

## 5. Reset scaffolding
- `ResetConfig` / `ResetEvent` を追加し、BoundarySignal が閾値を超えたときに scratchpad や affective_echo をクリア／減衰する余地を確保（現段階ではログのみ）。
- 今後 `_execute_boundary_reset()` を挟めば、Neuron 論文の“境界→リセット”と同型の挙動を完成できる。

## 6. Guardrails / テスト
- `scripts/force_matrix_smoke.py` で ForceMatrix merge と欠損挙動をサクッと検証可能。
- DampingConfig と DevelopmentConfig は YAML でホットリロードできるので、Gradio 経由で persona 間の dominant_self_layer や self_force を比較すると“心の指紋”をその場で観測できる。

## 7. 研究モードで見るべきログ列
1. `dominant_self_layer`
2. `raw_self_force` と `self_force`
3. `latency_ms`（ImplementationContext）
4. `context_tags` / `tag_bumps`
5. `BoundarySignal.score`

これらを同一入力 × 別 persona（例: なずな vs いぶき）で比較すると、ForceMatrix・摩擦・発達がどのように attractor を倒しているかを定量的に把握できます。
## 8. Log-Based Temptation Detection
- SelfForceSnapshot に winner / winner_margin / is_tie を追加し、raw force と damped force の両方で勝者・僅差を記録。
aw_winner != damped_winner だけでなく is_tie == True（勝者不在）や winner_margin が閾値以下の状態もログから直接抽出できる。	ie_eps は YAML で調整可能（例: 1e-3〜1e-2）にしておくと運用しやすい。with_winner() は既存ログと後方互換。
- BoundarySignal.sources の語彙を BOUNDARY_SOURCE_KEYS（pred_error_delta / force_delta / winner_flip / raw_damped_gap / latency_penalty / tag_bump）で固定し、可視化・集計時に列がブレないようにした。
- ResetEvent.targets を定数（scratchpad / affective_echo / force_cache / recent_tag_bumps）で扱い、reset ログの集計・評価を揺れなく実行できる。
- このログ構造により、「誘惑（raw 偏り）」「摩擦での勝者交代」「迷い（tie が続く）」を後から定量的に検出し、boundary_threshold・reset 条件・tag_bumps 調整を計測ベースで回せるようになった。

## 9. Temptation Examples: 危険な誘惑 vs 演出としての誘惑

| 種類 | ログでの特徴 | 典型パターン | 推奨アクション |
| --- | --- | --- | --- |
| **危険な詐誘惑**<br>（煽動・ガスライティング等） | - `raw_winner` が偏るが `damped_winner` が頻繁に変わる<br>- `winner_margin` が縮み続け tie が連続<br>- BoundarySignal の `winner_flip` / `raw_damped_gap` が高い | 「今すぐ」「絶対」「他の道はない」と圧をかける入力。生存/愛着の軸だけを過剰に引っ張り、Narrative を沈黙させる。 | - soft boundary ログを確認し、hard 判定を満たしたら `_execute_boundary_reset()` で scratchpad/echo をクールダウン<br>- tag_bumps / damping 設定を見直し、過剰な外部圧に対する摩擦を強める |
| **演出に使える誘惑**<br>（甘いご褒美・可愛い仕草） | - `raw_winner` と `damped_winner` が一致<br>- `winner_margin` が一定以上、`is_tie` は一時的<br>- BoundarySignal は低〜中程度 | 「今日は特別」「一緒にスイーツを食べよう」など、情動や遊びを演出する入力。Narrative が意味づけし、Affective が共有される。 | - 抑圧せず受け取り、ログでは “演出カテゴリ” としてマーキング<br>- 連続して margin が縮む場合だけ soft boundary を入れて疲労を回避 |

**ポイント**

- 「危険な誘惑」は決断を強制し margin/tie を長期化させる。ログでは `raw≠damped` と `is_tie` 連続で分かる。
- 「演出としての誘惑」は margin が回復し、Narrative が意味を付与する。可愛らしさを殺さずに済む。
- どちらも入力の種類ではなく「内部状態×入力」で決まるため、SelfForceSnapshot と BoundarySignal が観測器になる。

## 10. Internal State → Body Expression Mapping

EQNet が外部へ渡すのは「いまの運動をどれだけ歪ませるか」という **微分量** だけ。次へ進む／止まるを決めず、既存のモーションプランを粘性で揺らす層として扱う。

### 判断を「行動決定」にしない
- SelfForce の winner は「方向転換」ではなく、いま進行しているモーションへどれだけブレーキ／加速を掛けるかを示す連続値。
- `winner_margin` が高いほど迷いなく動ける。margin が下がったからといって分岐を作らない。あくまで係数だけを変える。

### 運動に落とす最小 3 スカラー
#### v_scale（速度係数）
- 意味: 「どれくらい迷わず動けるか」。`winner_margin ↓` や `boundary_score ↑` で 0.0〜1.0 の間を滑らかに下げる。
- 0/1 のステップは禁止。Reset 後は 0.0 まで落とし、1〜2 秒かけて指数的に 1.0 に戻す。「戻り」が安心感になる。

#### jitter（微小揺れ）
- 意味: 「身体が考えているか」。`is_tie = True`、`raw_self_force ≠ self_force`、`winner_flip` が増えたときに肩・首・視線へだけノイズを塗る。
- 足まわりや重心へは絶対に入れない。ノイズはゆっくり 0 へ減衰させ、安定状態では完全に止める。

#### d_target（距離目標）
- 意味: 「近づくか／距離を保つか」。`boundary_score ↑` で +方向へ、安定した甘い誘惑（margin が即回復/Tie 短い）でだけ軽くマイナス（最大 1 歩分）。
- Reset 直後は基準距離より少し広げ、「後退」より「距離を保つ」を演出する。最終手段としてのみ 1 歩下がる。

### EQNet → 運動は掛け算だけ
- 方向を決める加算・分岐は禁止。既存のモーションを歪ませるだけにする。

```python
v = planner_v * v_scale
pose += noise(jitter)
target_distance = base_distance + d_target
```

- これによりロコモーション側の計画は保ちつつ、「迷い」「ためらい」「間」の重さだけを共有できる。

### Reset を身体の儀式に
- Reset トリガーで v_scale を 0.0 へ落とし 1〜2 秒静止。
- 呼吸のような短い jitter を 1 発だけ入れて「考え直し」を見せる。
- `d_target` を一歩分広げ、落ち着きを取り戻しながら v_scale をゆっくり 1.0 へ戻す。
- これが「考え直した／落ち着いた」という人間的な儀式になる。UI やセリフの説明は不要。

### 最初の検証で見るべきもの
- 止まり方が「怖いブレーキ」にならず、「考えている時間」に見えるか。
- 引き方が「拒絶」ではなく「配慮」に読めるか。距離を取るときの一歩が最後の警告に留まっているか。
- 揺れが「生きた迷い」に見え、壊れたモーターに見えないか。特に脚部へ jitter が漏れていないかを必ず確認。
- これら “意味のある失敗” を最初に観測し、margin/tie/boundary のスケールを現場で調整する。

> 実装の I/O は `v_scale` / `jitter` / `d_target` の 3 スカラーだけに抑える。これができれば、EQNet は説明せずとも身体の間合いで心を伝える層になる。

## 11. Implementation Checkpoints（実装チェックポイント）

いまは設計を増やすより、「3 スカラーを既存モーションへ掛け算で繋ぎ、壊れ方を観測して係数を整える」フェーズ。次の順番で進めると調律が早い。

### 1. 掛け算だけをコードで保証
- `apply_eqnet_modulation()` のような 1 箇所に EQNet→運動の接続を閉じ込め、locomotion/gaze/voice からはその関数だけを呼ぶ。
- その中で `v = v_planner * v_scale` / `upper_pose += noise(jitter)` / `target_distance = d_base + d_target_offset` を行い、**目標点・経路・分岐を一切変えない**構造にする。
- 足まわりへ jitter/速度ノイズを混ぜる API をそもそも渡さない（引数設計で禁止）。「分岐禁止」「足元禁止」「目標点禁止」をコメントではなく責務分離で担保する。

### 2. 儀式 reset の台本を 1 本だけ固定
- シナリオ: ①安定(青) → ②迷い(tie) → ③境界上昇(赤) → ④儀式 reset → ⑤回復(青)。毎回同じ入力列で再生し、録画する。
- 観察ポイント:
  - `v_scale` の戻りが急峻になっていないか（急復帰は恐怖に見える）。
  - `jitter` が「1 回の呼吸」だけに見えるか（連発すると故障表現になる）。
  - `d_target` が「拒絶」ではなく「配慮」に読めるか（距離の取り方・目線の戻し方）。

### 3. Jitter が上半身だけに乗っているかログで確認
- `jitter_energy_upper / jitter_energy_lower` を計算し、下半身側が閾値以下であることを CI 的にチェック。
- Reset 区間だけタグで切り出し、儀式中の揺れ方を別ログに可視化。視覚だけに頼らない。

### 4. 「意味のある失敗」を 3 種類だけ集めて潰す
- **怖い停止**: 急停止・無言・間が硬い → `v_scale` 回復カーブを緩め、停止前に「ため」を必ず入れる。
- **故障っぽい揺れ**: 高周波で長時間揺れる → `jitter` の周波数/持続時間を上限クリップし、reset 中は 1 発だけに制限。
- **拒絶に見える距離**: 後退が強い・視線を外し続ける → `d_target` は「後退」ではなく「停止＋距離維持」を優先し、視線は「外す→戻す」を短時間で完了させる。
- 失敗パターンをこの 3 つに限定してログを貼り、係数（ゲイン・時定数・上半身限定）を調整する。

> ここまで進むと、次の改善はアルゴリズムより **スケール/時定数/適用部位** の調律になる。決めないのに伝わる層にするには、この単純な基底を壊さず育てるのが最速。

## 12. Upper/Lower Energy Metric (CI Guard Skeleton)

目的はただひとつ: **揺れが足回りへ漏れて「壊れて見える」/転倒しそうに見える事態をCIで止める**。上半身の揺れは意図的な表現として許容し、下半身に入った瞬間に検出する。

### 角速度²ベースの定義をデフォルトにする
- 上半身関節集合 U、下半身集合 L を固定し、各フレームで下記を計算する。
  - `E_U = sum(w_i * omega_i**2 for i in U)`
  - `E_L = sum(w_j * omega_j**2 for j in L)`
  - `lower_ratio = E_L / (E_U + eps)` （`eps` は 1e-6 などのゼロ割回避）
- 角速度²はセンサーで取りやすく、ノイズに比較的強いため初期実装に最適。後段で角加速度²やトルク²を追加したい場合は、この枠組みにスロットを増やす。
- `w_i` はまず全関節 1.0 でよい。首/肩を強調したくなったら後から重みを上げればいい。

### ratio 判定を安定させるガード
- `E_U` が極小（ほぼ 0）のフレームは ratio が跳ねるため、`E_U < EU_MIN` のときは判定をスキップ。`EU_MIN` は 1e-5〜1e-4 など小さな値で良い。
- reset 区間は 1 呼吸の揺れを許容するので、`phase_tag == "reset_ritual"` のときだけ threshold を少し緩める（例: 0.15 → 0.20）、または「連続時間で判定（短時間スパイクは許容）」のどちらかで実装する。

### compute_energy() と CI チェックの疑似コード
```python
UPPER_JOINTS = ["neck_yaw", "neck_pitch", "shoulder_l", ...]
LOWER_JOINTS = ["hip_l", "knee_l", ...]

EU_MIN = 1e-5
EPS = 1e-6

def compute_energy(snapshot):
    # snapshot: {joint_name: {"omega": value}}
    e_upper = sum(snapshot[j]["omega"]**2 for j in UPPER_JOINTS)
    e_lower = sum(snapshot[j]["omega"]**2 for j in LOWER_JOINTS)
    ratio = e_lower / (e_upper + EPS) if e_upper >= EU_MIN else 0.0
    return e_upper, e_lower, ratio

def ci_check_jitter_leak(rows, *, ratio_thr=0.15, ratio_thr_reset=0.20):
    normal = [r["lower_ratio"] for r in rows
              if r.get("phase_tag") != "reset_ritual" and r["upper_energy"] >= EU_MIN]
    reset = [r["lower_ratio"] for r in rows
             if r.get("phase_tag") == "reset_ritual" and r["upper_energy"] >= EU_MIN]

    worst_normal = max(normal) if normal else 0.0
    worst_reset = max(reset) if reset else 0.0

    if worst_normal > ratio_thr:
        raise SystemExit(f"CI FAIL: lower_ratio leak in normal phase {worst_normal:.3f} > {ratio_thr}")
    if worst_reset > ratio_thr_reset:
        raise SystemExit(f"CI FAIL: lower_ratio leak in reset ritual {worst_reset:.3f} > {ratio_thr_reset}")
```

### 拡張の順番
1. 角速度²で運用し、ログに `upper_energy`, `lower_energy`, `lower_ratio`, `phase_tag` を必ず出す。
2. `tau_recover`, `jitter_amp_cap`, `d_target_cap` を回したあとでも「故障っぽいカクつき」が残るなら、角加速度²を追加（必ずローパスを入れる）。
3. 実機の安全側を強化したくなったらトルク²/電流²による負荷監視を別途足す。

> 目的を「下半身に揺れを入れない」に固定すると、CI が“過敏すぎて疲れる”問題を避けつつ、事故を確実に止められる。


### 実装で詰まりやすいポイント（先回りメモ）
1. `EU_MIN` は固定値より相対推定にすると機体差でブレない。台本ログから `percentile(E_U, 10%)` や `EU_MIN = k * median(E_U)`（k=0.05 など）を事前に算出して使う。
2. ratio 判定は瞬間最大だけでなく「連続 N フレーム」で見るとノイズに強い。例: 通常は `lower_ratio > 0.15` が連続 3 フレーム、reset は `lower_ratio > 0.20` が連続 5 フレームで fail。
3. 関節重み `w_i` は最初すべて 1.0 で良いが、精度を上げたくなったら上半身は首/肩、下半身は膝/足首の重みを高めるだけで “見た目として壊れて見える” 漏れを拾いやすくなる。
4. reset_ritual の扱いは実装でどちらかに固定する。まずは「閾値を緩める + 連続フレーム数を増やす」を推奨（平滑化はフィルタ調整で沼りがち）。
5. CI が読むログのキーを契約として固定（`phase_tag`, `upper_energy`, `lower_energy`, `lower_ratio`, `timestamp` など）。これで可視化/回帰テストまで同じ資産を使い回せる。
6. 台本ログを 1 本回し、そこから `EU_MIN`, 閾値, 連続フレーム数を決定すると運用が安定する。

> 実機の関節リスト（上半身/下半身）が決まり次第、`compute_energy()` へそのまま drop-in できる。そこまで進めば「あとはノブを回すだけ」の状態になる。


### 運用を崩さない最終の締め
1. 関節リスト（名前＋重み）をソートして文字列化し、SHA-256 などでチェックサムを残す。ログ/CI 出力に `joint_hash=...` を含めれば、機体やブランチが変わったときの差分が即座に追える。
2. しきい値決定は「儀式台本ログ + 通常モーション 1 本」の二重チェックにする。台本で `EU_MIN`・ratio 閾値・連続フレーム数を決めたら、別シーンを1本だけ流して誤検知がないかを確認し、台本専用の調律にならないようにする。
3. CI の fail メッセージを 1 行フォーマットで契約化する。例: `LEAK_LOWER_JITTER normal max_ratio=0.182 thr=0.150 EU=0.84 frame=412 joints_top=[ankle_r,knee_l]` / `RITUAL_LEAK reset max_ratio=0.215 thr=0.200 frame=97`。落ちた瞬間に「どのノブ？どの関節？」が分かれば、調整が迷子にならない。

> この 3 点まで固めると、残る育成対象は「時間の形（tau_recover / reset 呼吸の長さ）」と「付着点の形（首・肩・視線への配分）」だけになる。そこで違和感が出たら、CI の 1 行ログと動画数秒でノブの方向が即決できる。


## 現場“呼吸”チェックリスト（ワンページ運用）

**朝 (2 分)**
- チェックサム一致 / `EU_MIN`・threshold・N 読み込み済み
- 観察レーン有効 / E-stop 動作確認

**実行 (3–4 分)**
- 台本 → 自然モーション（順番固定）
- phase タグ付きログを切り出して保存

**違和感が出たら (30 秒)**
- 動画 5–10 秒（直前〜直後）
- CI 1 行ログ（契約フォーマット）
- 状況タグ 3 つ（例: 人が近い / 騒音 / 狭い）
  ※長文禁止、これ以外は書かない

**週 1 (10 分)**
- ノブは 1 つだけ動かす（`tau_recover` / `jitter_amp_cap` / `d_target_cap`）
- 動かしたら 1 週間固定で観測

**KPI（毎日見る）**
- 怖い停止がない
- 拒絶距離がない
- CI で下半身 jitter 0（fail なし）

> この呼吸を回すだけで、設計は勝っている。判断より観察。違和感が出たら「動画数秒 + CI 1 行 + 状況タグ」を投げ、どのノブを回すかだけ決める。


## EQNet 2D Simulation Template（実機なしで「滲み出る」を見る）
- 目的: AB（EQNet OFF/ON）を GIF + motion_log.jsonl で自動生成し、「怖くない停止／拒絶に見えない距離／呼吸っぽい揺れ」が視覚と CI の両方で確認できる最小モデルを回す。
- 2D の点ロボット＋向きで十分。planner が出す `v_planner` / `base_distance` に対し、EQNet の `v_scale`/`jitter`/`d_target` だけを掛け算・オフセットで適用。
- 上半身＝頭線の微小揺れ、下半身＝ベース点。jitter が漏れたら `lower_ratio` が跳ねて CI が落ちる。
- テンプレファイル: `sim_eqnet_motion_ab.py`（matplotlib + numpy + pillow）。`pip install matplotlib numpy pillow` 後に実行すると `sim_out/A_eqnet_off` と `sim_out/B_eqnet_on` に `sim.gif` と `motion_log.jsonl` が出る。
- スカラー生成: `scenario_scalars()` で 0-14s の進行（通常→迷い→boundary→reset→回復）を記述。OFF はすべて 1.0/0.0。
- coupling 契約: `apply_eqnet_modulation()` が `Plan` を受け取り、速度は掛け算、距離はオフセット、上半身だけ `add_upper_body_micro_noise()`、target は変わらないことを assert。
- CI ガード: `ci_check_jitter_leak()` が `lower_ratio` を通常/リセット別の連続フレーム判定で監視（閾値 0.15 / 0.20、EU_MIN=0.02、N=3/5）。
- エネルギー指標: `compute_energy()` で頭角速度²を `E_U`、ベース速度²を `E_L` としてログ化（実機なら joint 別 `omega` に置き換える）。
- 実験手順: (1) A/B GIF を第三者に盲検提示し 3 質問（止まり方/距離/揺れ）を取る。(2) `motion_log.jsonl` で CI ガードを確認。
- 改造ポイント: jitter を視線線にも分配、`d_target` を後退ではなく「停止＋距離維持」に寄せる、reset jitter を 1 呼吸のサイン波に固定するなど“出力先”だけを足す。
- 実機 I/F（ROS2、gaze controller 等）が固まったら、この `Plan` を移植して実機ベースの A/B へ段階的に拡張できる。
\n### 2D GUI 操作メモ\n- `python sim_eqnet_motion_ab.py --gui` で GUI を起動（従来モードは引数なし）。\n- キー操作: `SPACE`=同調タップ記録, `p`=再生/一時停止, `1/2/3`=0.5x/1x/2x 速度, `a/b`=EQNet OFF/ON 切替, `s`=タップ保存, `q`/`ESC`=終了(自動保存)。\n- タップ保存先: `sim_out_gui/taps_{A|B}_...jsonl`。モード切替でタップはリセット。\n
