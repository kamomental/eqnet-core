# Mock-Theta Completion Layer（要約）

## ひと言でいうと

EQNet が  
**「いま感じていること」** と **「これまでの文脈」** を毎ターン統合し、  
**「迷っているなら、迷っていると伝えられる」** ようにする  
**自己統合レイヤ**です。

---

## なぜ必要か

EQNet には、時間スケールの異なる情報が同時に存在します。

- **速いもの**
  - Affect（瞬間の感情）
  - センサー入力

- **遅いもの**
  - Memory Replay
  - Culture EMA
  - PDC（過去・関係性・未来の予測）

この両者が食い違うと、  
「どちらを信じるか」が暗黙になりがちです。

**Mock-Theta Completion Layer** は、そのズレを必ず顕在化させ、  
- 統合された現在の感情
- その状態への自信度（`mood_uncertainty`）

を明示的に計算し、TalkMode / Policy に渡します。

---

## Mock / Shadow / Completion

- **Mock（表の感情）**  
  今ターンで観測した affect。

- **Shadow（影の予測）**  
  記憶・文化・未来志向から得られる、遅い整合状態。

- **Completion（統合）**  
  Mock と Shadow を信頼度つきで混ぜた結果。

重要なのは、  
**「統合後の感情」** と同時に  
**「どれくらい自信があるか」** を必ず出力することです。

---

## 何をしている層か

> **「自分がどれくらい分かっているか」を自分で測り、  
> 振る舞いを変える層**

自信が低いときは、
- 確認質問を挟む
- 間を取る
- 言い切らない

といった判断を、  
**演出ではなく数値とログ**で行います。

---

## TalkMode / Policy / MaskLayer への影響

### TalkMode
- ASK / TALK の切り替えが **不確実性で決まる**
- 「なぜ質問が多かったか」に説明責任を持てる

### Policy
- 温度や pause を  
  **感情の強さではなく「不確かさ」**で調整
- 暴走や早口を構造的に抑制できる

### MaskLayer
- 「不確実なら、まずそれを認める」
  といった仕様を **inner spec に明示**
- 人格演出ではなく、**契約（仕様）として制御**

---

## 意識なのか？

これは **意識を生む装置ではありません**。  
自己状態を一貫して扱うための **制御層**です。

人間に例えるなら：

> **「分かっていないのに、分かったふりをしない能力」**

これを **ログ付きで回す**ための仕組みです。

---

## EQNet 全体での位置づけ

- 速い反応と遅い意味を、いきなり混ぜない  
- まず「迷い」として扱う  
- 迷いを数値化（`mood_uncertainty`）して  
  TalkMode / Policy / MaskLayer へ渡す  
- 結果を workspace / moment / telemetry に残し、  
  Observer / Nightly で検証可能にする  

**「考えているように見せる」のではなく、  
実際に *考えてから話す順序* を持つためのレイヤ**です。


# Mock-Theta Completion Layer

## Why “mock theta”?

- EQNet already separates **fast affect**（観測された即時感情）and **slow context**（memory replay、culture EMA、PDC future prototypes）。
- The “mock + shadow” idea captures the need to reconcile fast observations with slow predictions **each turn**:  
  - **mock** = visible affect  
  - **shadow** = invisible, slower dynamics  
  - **completion** = a consistent “now”
- The goal is **not** to prove consciousness.  
  It is to provide an explicit, measurable **self-integration layer** so TalkMode / Policy / MaskLayer react to uncertainty in a predictable, inspectable way.

---

## Shadow → Completion → Control

### 1. ShadowState

- Built right after PDC (prospective core) and before TalkMode / Policy inside  
  `EmotionalHubRuntime.step()`.
- **Inputs**
  - AffectSample (valence / arousal / confidence)
  - PDC outputs (future prototype weights, jerk, delta_m)
  - Memory replay stats (top scores, entropy, similarity)  
    → coming from InnerReplay / episode store
  - Culture EMA (72h climate + fast spikes)  
    → `culture_state` cached from the previous turn
  - Sensor confidence (percept-level)
- **Outputs (per turn)**
  - `pred_valence`, `pred_arousal`
  - Evidence:
    - `replay_entropy`
    - `pdc_alignment`
    - `culture_bias`
    - `jerk_norm`
  - Fit:
    - `residual = ||affect_now - pred||`
    - `alpha_used`
    - `mood_uncertainty`

---

### 2. Completion

- Use an adaptive blend:

alpha = sigma_obs^2 / (sigma_obs^2 + sigma_prior^2)


derived from confidence / residual / entropy.

- This avoids trusting the shadow when sensors are reliable, and vice versa.

- Final state:

completed_affect = (1 - alpha) * affect_now + alpha * shadow_pred


---

### 3. Control Targets

- **TalkMode**
- `mood_uncertainty` biases mode selection:
  - Insert **ASK when uncertain** before TALK
  - Or instruct MaskLayer to insert clarifying questions if `u > threshold`
  - WATCH-only fallback remains behind safety gates

- **PolicyHead**
- Keeps its monotonic mapping, then post-adjusts:
  ```
  temperature = lerp(temp, temp_safe, u)
  pause_ms    = lerp(pause_ms, pause_long, u)
  ```
- High jerk / high residual → slower, safer speech  
  （no hidden rules in a black box）

- **MaskLayer**
- Receives `completed_affect` and `mood_uncertainty` inside its immutable inner spec
- Prompt templates can literally say:  
  “If `mood_uncertainty` is high, acknowledge uncertainty before giving advice.”

---

### 4. Telemetry & audits

- `workspace.snapshot_v1` extends field meta with:
- `completed_affect`
- `mood_uncertainty`
- `alpha_used`
- `shadow_evidence`
- MomentLog gets the same payload, so nightly reports can aggregate:
- mean / p95 uncertainty
- time to recover after perturbations
- correlation with ASK/TALK transitions
- Interventions (completion off, forced perturbations) are logged via  
`intervention.applied_v1`, so ablations have traceable effects.

---

## Is “mock theta” meaningful?

- **Yes, as a design lens**  
The mock / shadow / completion language captures  
“visible fast affect + invisible slow context → explicitly reconciled state”  
in a way that aligns with EQNet’s architecture (replay, culture EMA, PDC).  
It demands a concrete blend and residual logging **each turn**.

- **No, for consciousness claims**  
This layer is a self-integration / uncertainty-control mechanism.  
It does not prove or generate qualia.  
Its value is that every decision now carries:
1. a completed affect state  
2. an uncertainty scalar  
3. logged evidence for later audits  

These are exactly the knobs Observer / Nightly need when explaining  
“why did we hesitate?”

- **Future work depends on InnerReplay maturity**  
Today, `InnerReplay` is knob-based.  
This design fixes the interface (`get_replay_stats()` returning entropy, alignment, etc.)  
so we can swap in deeper episodic search later **without breaking completion**.

---

## Next steps

1. Implement `ShadowEstimator` (PDC field) and the adaptive α-blend in  
 `EmotionalHubRuntime.step()`.
2. Feed `mood_uncertainty` into TalkMode / Policy / MaskLayer and log the result in  
 workspace / moment snapshots.
3. Add nightly metrics:
 - uncertainty recovery time  
 - ASK/TALK distribution versus `u`  
 - intervention responses
4. Once InnerReplay exposes real distributions, swap out placeholder stats.  
 The contract already matches.

---
