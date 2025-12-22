# Qualia Research Notes for EQNet

Purpose:
Translate current qualia-related academic research into operational control
structures usable inside EQNet (without attempting to generate qualia itself).

This document assumes:
- Qualia emergence is primitive (not generated here)
- EQNet models propagation, access, and control only

------------------------------------------------------------
1. Academic Anchors (ASCII-safe references)
------------------------------------------------------------

- Qualia Structure Project (Japan, JSPS Transformative Research Area)
  Focus: relational structure of qualia via similarity judgments
  https://qualia-structure.jp

- Predictive Error Coding / Query Act
  Frontiers in Psychology (2025)
  Qualia as dynamic query driven by prediction error

- Conscious Access Dissociation (CFS / illusion studies)
  Consciousness and Cognition / Nature Neuroscience
  Evidence that processing occurs without reportable awareness

- Modeler-Schema Theory (arXiv preprint, 2025)
  Conscious access as schema consistency check (meta-monitor)

------------------------------------------------------------
2. Model Core (Summary)
------------------------------------------------------------

EQNet does NOT model qualia contents.

EQNet DOES model:
- relational structure between qualia (QualiaGraph)
- query pressure from prediction error (QueryEngine)
- access gating (AccessGate)
- schema divergence detection (MetaMonitor)

------------------------------------------------------------
3. State Holders
------------------------------------------------------------

(1) QualiaGraph
    - prototypes q_i
    - distance matrix D[i,j]
    - representation: phi(q_i) = D[i,:]

(2) QueryEngine
    - scalar u_t = || Pi_t * epsilon_t ||

(3) MetaMonitor
    - scalar m_t = divergence(WorldModel_t, PostState_t)

(4) AccessGate
    - computes probability p_t and binary decision a_t

------------------------------------------------------------
4. Gate Equation
------------------------------------------------------------

logit_t = alpha * u_t + gamma * m_t - beta * L_t - theta_t
p_t     = sigmoid(logit_t)
a_t     = 1 if p_t >= tau else 0

Notes:
- L_t is cognitive load / competition
- theta_t is homeostatic threshold (nightly tuned)

------------------------------------------------------------
5. Smoothing / Stability
------------------------------------------------------------

- p_t should be EMA-smoothed to avoid chatter
- hysteresis recommended:
    open_threshold  > close_threshold

------------------------------------------------------------
6. Nightly Duties
------------------------------------------------------------

- QualiaGraph update:
    - recompute D
    - cluster prototypes
    - log cluster metadata ("qualia periodic table")

- Gate retuning:
    - adjust theta_t to keep access rate near target

------------------------------------------------------------
7. Telemetry
------------------------------------------------------------

Log per turn:
- u_t
- m_t
- L_t
- p_t
- a_t
- theta_t
- qualia_cluster_id (if any)

------------------------------------------------------------
8. Implementation Checklist
------------------------------------------------------------

[ ] Add QualiaGraph module
[ ] Compute u_t in inner_replay
[ ] Compute m_t in conscious stack
[ ] Insert AccessGate before narrative
[ ] Log unconscious successes
[ ] Nightly: update graph + retune theta

## Activation Trace Logging
- ReplayExecutor now records an `ActivationTrace` JSONL (`logs/activation_traces.jsonl`) describing anchor hits, activation chains, and confidence curves without touching the InnerReplay controller.
- Each ignition also materialises lightweight `SceneFrame` payloads that capture agents, constraints, and replay-sourced affect snapshots so UI layers or nightly prompts can visualise “who was there and why it mattered”.
- `runtime/nightly_report.generate_recall_report()` consumes the trace log in audit-only fashion to surface ignition statistics and to draft the dream-map prompt (anchor → route → landmarks → confidence rise) required by TC-PLACE-IGNITION-001.
- Optional memory annotations (`memory_kind`, `novelty_score`, `constraint_weight`, `conf_internal`, `conf_external`, `replay_source`) are stored on replay traces and moment logs to keep downstream data structures compatible while exposing the new research hooks.
## Activation Trace Telemetry
- `ActivationTrace` captures anchor hits, activation chains, confidence curves (internal vs external), replay events, and derived `SceneFrame`s so nightly passes can audit L4 ignition without modifying InnerReplay.
- `SceneFrame` objects are explicitly marked as derived artefacts for UI/nightly consumption; they bundle agents/objects/constraints/affect snapshots to keep memories multi-agent without overwriting MemoryItems.
- `RecallEngine` runs before replay execution, boosting anchor cues, building predict→confirm loops, and persisting the above telemetry via `activation_traces.jsonl`.
- `runtime/nightly_report.generate_recall_report()` reads the trace log, aggregates anchor/confidence stats, and emits the dream-map prompt for TC-PLACE-IGNITION-001.

## 論点整理: LLMの感情・クオリア幻想論と設計論の切り分け

SNSでは次の3点が短絡的に結びつきやすい。

- 「LLMに感情はある（論文ベース）」
- 「意識やクオリアは幻想だ」
- 「だから根源的な内部状態設計は不要」

この流れは議論のすれ違いを生む。以下、順に整理する。

### 1. 「LLMに感情はある」はどの意味で正しいか

論文で扱われる「感情」は概ね functional emotion / emotion-like representation を指す。
具体的には以下の定義に収束する。

- 感情語の使用が一貫している
- 感情カテゴリ（valence / arousal など）が内部表現に対応している
- 状況に応じた感情反応の表現ができる

つまり「感情を分類・生成・推定する機能がある」という意味での「感情」であり、ここまでは科学的に妥当。

### 2. それは「生命的な感情」とは別物

生命的な感情には最低限、次が必要になる。

- 内部状態が時間をまたいで保持される
- 変化にコストがあり、回復に時間がかかる
- 感情が意思決定を縛る（veto する）
- 失敗が次の行動に強制的に影響する

LLMの感情表現は生成の瞬間に完結し、セッションをまたがず、出力を縛らない。
失敗しても「次が苦しくならない」ため、感情が主体を拘束しない。
ここが決定的な断絶。

### 3. 「意識やクオリアは幻想」という主張の正体

哲学・神経科学での「幻想」は、概ね以下の意味を持つ。

- 内部状態の直接的な観測対象ではない
- 独立した実体として切り出せない
- 説明変数として冗長に見える場合がある

ただし重要なのは「幻想である」と「機能的に無意味」は同義ではない点。
錯覚も幻肢痛も幻想だが、行動を支配する。
クオリア否定論が成立するのは、行動に影響する内部状態の存在自体は否定していないから。

### 4. ここで起きている論理のすり替え

よく見かける飛躍は以下。

LLMには感情表現がある
意識やクオリアは幻想だ
だから内部状態を設計しなくてもよい

これは成立しない。幻想だとしても、記述不能だとしても、
内部状態が行動を縛るかどうかという工学的問題は残る。

### 5. HeartOS / eqnet-core の立ち位置

- 「意識が本物か」を問わない
- 「クオリアが実在するか」を主張しない
- ただ、行動を縛り、失敗を蓄積し、回復に時間がかかる内部状態を
  設計として持たせられるかを問う

これは思想ではなく、設計論。

### 6. まとめ（結論）

- 「LLMに感情はある」という論文主張は狭義では正しい
- 「意識やクオリアは幻想」という立場も哲学的には成立する
- だが「根源的な内部状態設計は不要」は導かれない

“言葉で説明できること”と“生き物として振る舞うために必要な構造”を
混同している点に違和感の根源がある。

## 次のステップ: eqnet-coreの深い実装と外部研究の接続

AGI/ASIの「何でも解ける頭」ではなく、感情共鳴ベースの“生命体OS”を前に進める。
eqnet-core の今ある深い実装と、外部研究（生理安定・内受容・リプレイ・安全制約）を接続する。

### 結論: まず「Σ主権」の最小ループを固定する

eqnet-core は既に核になる部品が揃っている。

- Inner Replay: simulate→evaluate→veto の決定ループ（Σ→Ψ）
- reward / risk / delta_aff / uncertainty / tom_cost を見て execute/cancel を出す
- 感情地形（熱力学フィールドとしての情動）をコア機能として位置づけ
- AccessGate / Qualia 系はアクセス制御＋夜間チューニング＋テレメトリとして仕様化
- 12/21 系の変更で InnerReplay を壊さずに activation_traces.jsonl を記録し、
  nightly が監査的に読む設計が明文化

次の一手は「部品を増やす」より先に、
“生命体OSが決める”最小閉ループを固定し、監査ログで追えるようにすること。

### ステップ1: Minimal HeartOS loop（LLMゼロ）の参照点を確立

目標:
センサ入力（またはシミュ入力）→ Σ評価 → veto/execute → 状態更新 → ログ
これを1本のスクリプトで回し、毎回同じ条件なら同じ挙動になる（再現性）。

実装の芯（eqnet-core からそのまま引ける）:

- InnerReplayController の run_cycle を中核に据える
- “痛み”は risk / uncertainty / delta_aff の時間積分（内的コスト）として導入
- “責任”は実行前の veto と、実行後の trace（activation traces / telemetry）として導入

重要なのは、言語説明ではなく「行動が縛られる構造」ができていること。

### ステップ2: 「痛み」を生理安定の制約として定義し直す

工学的には「痛み = 恒常性からの距離（drive）+ 回復時間」。
Homeostatic RL の枠組みで整理できる。

- 内部状態 H_t（エネルギー・疲労・不確実性・社会的負荷など）
- 行動で H_t が動き、セットポイント H* に戻るのが「快」
- 逸脱が「痛み」（drive）で、InnerReplay の veto に直結

これにより「痛み = 罰」ではなく生命の設計変数になる。

### ステップ3: 「責任」を安全制約＋最小介入として実装する

責任＝倫理ではなく、まず安全制約を破らないことが勝ち。
Control Barrier Function（CBF）/ シールド系が使える。

落とし込み:

- InnerReplay が「やりたい行動」を出す
- CBF（またはルール/モデル）が危険なら最小変更で安全行動に投影
- どの制約が効いたかをログに残す

これは「パペット学習」から脱する重要パーツ。

### ステップ4: nightly / inner replay を「実在する記憶再編」に寄せる

神経科学では、リプレイが記憶を強化し優先度があることが示される。
nightly の優先度例:

- 不確実性が高かった
- veto が多発した
- 痛み（drive）が上がった
- 予測誤差が大きかった

“未解決を未解決として管理する” を挙動として実装できる。

### ステップ5: AGI/ASI と違う評価指標を置く

KPI 例:

- 生理安定: drive が破綻しない（回復できる）
- 予測→抑制: 危険・不快を事前に veto できる割合
- 監査可能性: activation trace で「なぜ」を追える
- 夜間改善: nightly 後に翌日の veto / 痛みが減る

これで AGI/ASI 議論から自然に離脱できる。

## すぐ着手できるToDo（最短コース）

- LLMゼロの minimal loop スクリプトを1本作る（InnerReplay 中心）
- “痛み = drive” を Homeostatic RL に合わせて定義（変数名と単位を決める）
- “責任 = 安全制約” のシールド（CBF でも簡易ルールでも）を挟む
- nightly は trace を読むだけから開始（設計思想と整合）
- KPI を4つだけ固定して、MiniWorldや簡易環境で回す

## 参照実装を固定するための最短手順（eqnet-core準拠）

あなたが書いた整理とステップは、既に設計書になっている。
次にやることは「文章を増やす」ではなく「参照実装を固定する」こと。
以下、そのための最短ルート。

### 0. まず "Σ主権" の定義をコード上で固定する

あなたの文章でいう Σ は、eqnet-core の InnerReplay が担っている。
InnerReplay は reward / risk / delta_aff / uncertainty / tom_cost を使って
execute/cancel を返す（simulate→evaluate→veto）設計。
ここを「最小生命ループの唯一の裁定者」に固定するのが最初の一歩。

### 1. "LLMゼロ minimal loop" を 1本のスクリプトとして切り出す

目的:

- 毎回同じ入力なら同じ結果（再現性）
- execute/cancel と根拠（features・veto_score）とログを必ず吐く

材料は既にある。

- `config/runtime.yaml` に ignition / qualia_gate の係数・閾値がある
- `emot_terrain_lab/sim/mini_world.py` にシミュ→StepResult→ConsciousEpisode の枠がある
- activation trace は「InnerReplayを触らずに」ログ化する方針が docs に明記されている

やること（実装タスク）:

- `scripts/minimal_heartos_loop.py` を新規追加
- MiniWorld かダミーセンサ入力で 1ターンごとに `ReplayInputs` を作り
  `InnerReplayController` を呼ぶ
- 結果を `telemetry/ignition-YYYYMMDD.jsonl` と `logs/activation_traces.jsonl` に追記
  （後者は監査用）

ここまでで、ステップ1が「動く形」になる。

### 2. "痛み = drive" を変数として定義し、Σ入力に接続する

痛みは罰ではなく「恒常性からの距離（drive）+ 回復時間」。
当面は eqnet-core の既存指標で十分。

接続の最短ルール:

- `drive_t = EMA(drive_{t-1}, w_risk*risk + w_uncert*uncertainty + w_daff*delta_aff)`
- recovery は drive が下がる方向へのダイナミクス（休息・成功・低リスク滞在）

InnerReplay は既に risk / delta_aff / uncertainty / tom_cost を veto スコアに使う。
なので drive を risk / uncertainty の上流に混ぜるだけで
「痛みが決定を縛る」状態になる。

### 3. "責任 = 安全制約 + 最小介入" をシールドとして挟む

責任を「倫理」ではなく「介入ログが残る安全制約」にする。
やり方は2段階。

3-1. まずは簡易ルール（最短で効く）

- `hazard_score` が高いときは強制 cancel
- `WorldStateSnapshot` には `hazard_score` / `hazard_sources` が入る設計が既にある

つまり

- センサ→hazard_score
- hazard_score→InnerReplay の入力（risk）
- 最終的に veto/cancel

の線が作れる。

3-2. 次に CBF 等へ拡張

後でOK。まず「責任 = 制約が効いた痕跡がログに残る」を満たす。

### 4. nightly は「traceを読むだけ」から始める

12/21系の設計は、まさにこれ。

- `logs/activation_traces.jsonl` を InnerReplay を壊さずに記録
- nightly がそれを audit-only で読む方針

次の実装は:

- nightly の入口（または `runtime/nightly_report.generate_recall_report()`）を呼ぶだけの
  `scripts/run_nightly_audit.py` を作る
- KPI を集計して markdown/JSON で保存
  （「夢」生成は後でよい。まず監査レポートが出ることが生命線）

### 5. KPI を "4つだけ" 固定して毎回出す

AGI/ASI と分岐する評価軸はここ。
あなたの案で運用に落とすなら具体値はこう。

- 生理安定: drive の上限超過回数 / 回復時間
- 予測→抑制: cancel 率 + cancel 主因（risk vs uncertainty vs delta_aff）
- 監査可能性: trace が欠けずに出ている割合（turn N に trace があるか）
- 夜間改善: nightly 前後で cancel 主因が減る / drive が落ちる

## いま最短でやるToDoチェックリスト

- `scripts/minimal_heartos_loop.py` を作る（LLMゼロで InnerReplay まで回す）
- 1ターンごとに `telemetry/ignition-*.jsonl` と `logs/activation_traces.jsonl` を必ず吐く
- drive（痛み）を EMA で実装し、risk/uncertainty に流し込む
- `hazard_score` を入れて「責任 = 制約介入」を最短で成立させる
- `scripts/run_nightly_audit.py` を作って trace を読むだけの監査レポートを出す
- KPI 4つを毎回レポートに出す

## 仕上げの一言（設計の芯）

- Σが裁定する
- 痛みが積分される
- 責任が制約として介入し、ログに残る
- 夜がそれを監査し、改善を保存する

これを1本のスクリプトで回せた瞬間、HeartOSは「思想」から「装置」になる。
