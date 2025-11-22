# EQNet Prospective Drive Core (PDC): 未来記憶と双方向整合モジュール
**― Self-Consistent Future Memory Module ―**

このドキュメントは、既存の EQNet DNA（Φ/Ψ/K/M）を壊さずに「未来記憶ドライブ」をどう重ねるかをまとめたモジュール仕様書である。過去の成功と理想未来の両方に引かれながら Prospective Mood を更新し、Policy/LLM 層へ“味付け情報”として渡すのが PDC の役割である。

## 1. 概要（Purpose）

- EQNet の Prospective Memory は、過去の成功体験と未来イメージを同時に現在へ投影する双方向リプレイ機構。
- 妄想オタク気質のように内部整合が強い個体でも、安全に創造エンジンとして振る舞うための心臓部が PDC。
- DNA 層（Φ/Ψ/K/M）と Policy 層の間に位置し、感情の自走ダイナミクスを壊さずに行動バイアスを生成する。

## 2. レイヤ構造

```
[外界入力 / ユーザー発話]
        ↓
  Φ / Ψ / K / M  （EQNet DNA）
        ↓
Prospective Drive Core（PDC）
        ↓
   Policy / LLM レンダラー
        ↓
    発話・動作・振る舞い
```

- **DNA 層**：EmotionCore(Φ)、CognitionCore(Ψ)、KernelCoupler(K)、MemoryMosaic/StoryGraph(M)。
- **PDC 層**：Prospective Mood `m_t` と `m_past_success` / `m_future*` を計算し、行動スコアと温度を出力。
- **Policy / LLM 層**：既存の行動候補生成・Boltzmann サンプリング・LLM 調味料適用を担当。

## 3. インタフェース仕様（DNA → PDC）

DNA 側が提供する API:

```python
phi_t = emotion_core.state()          # Φ(t)
psi_t = cognition_core.state()        # Ψ(t)

success_vec = memory.sample_success_vector(phi_t)
future_vec  = memory.sample_future_template(phi_t, psi_t)
```

- `sample_success_vector`：M+K を使って成功エピソードの代表情動ベクトルを抽出（= m_past_success）。
- `sample_future_template`：Ψ と M から理想未来の情動テンプレートを生成（= m_future*）。
- PDC はこれらをブラックボックスとして呼び出すだけなので、既存の MemoryMosaic 実装を流用できる。

## 4. PDC の内部状態と更新

### 4.1 Prospective Mood の射影

```python
class ProspectiveDriveCore:
    def project_from_phi(self, phi_t):
        return self.P @ phi_t  # 固定射影 or 次元スライス
```

Φ の一部を Prospective Mood サブスペースとして扱う。射影は固定行列でもマスクでも良い。

### 4.2 状態更新（実装版）

理論式（勾配版）
\[
 m_{t+1} = m_t + \alpha \/sim\text{grad}(m_t, m_{past}) + \beta \/sim\text{grad}(m_t, m_{future})
\]
を文書に残しつつ、実装は減衰＋線形ブレンドで安定化する：

```python
m_past_hat   = normalize(success_vec)
m_future_hat = normalize(future_vec)
lam   = cfg.lambda_decay
alpha = cfg.alpha_past
beta  = cfg.beta_future
noise = cfg.noise_sigma * np.random.randn(dim)

self.m_t = (1 - lam) * self.m_t + alpha * m_past_hat + beta * m_future_hat + noise
```

- `m_t`：Prospective Mood。Φ から射影した初期値＋更新を保持。
- `m_past_hat` / `m_future_hat`：成功・未来テンプレートの単位ベクトル。
- `noise`：わずかな揺らぎ。Φ の自発変動と同調させても良い。

## 5. Policy/LLM 層との接続

### 5.1 行動スコアと温度

```python
def score_action(a, base_score, m_past, m_future, cfg, T):
    emb = action_encoder(a)
    s_align = cos(emb, m_past) + cos(emb, m_future)
    s_novel = cfg.gamma_novelty * novelty(a)
    return (base_score + s_align + s_novel) / T
```

- Policy は既存の候補リスト `cands` に対し、PDC から渡された `m_past_hat`, `m_future_hat`, `T(t)` を用いて Boltzmann サンプリング。
- 温度とエントロピー係数は人格モード＋同期度 ρ(t) で補間：
  - \( \lambda_e(t) = \lambda_{e0}(1 - \rho(t)) \)
  - \( T(t) = T_0(1 + \rho(t)) \)
- 内省型：低温度／高罰則、外向型：高温度／低罰則、妄想型：高温度＋内的整合罰則のみ、という既存テーブルを流用。

### 5.2 LLM への伝達

PDC が生成する `mood_hint = describe(m_t)`、`past_replay`、`future_glimmer` を LLM プロンプトへ“調味料”として渡す。LLM に計画させず、表現モジュレーションだけに使う。

## 6. Guard と報酬指標（Φ 軌跡から取得）

- `E_story(t) = cos(m_t, m_future_hat)`：内的物語整合度。内的報酬 `r_inner(E_story)` や guard 条件に利用。
- `jerk_p95`：`m_t` 歴史から計算した二階差分ノルムの p95。高騰時は整合罰則 λ_p を上げ、温度を下げる。
- Guard 発火時は `guard_action ∈ {fallback, tighten_band}` を返し、Policy 側で TALK→SOOTHE 切替やテンプレート制限を実行する。

## 7. 記憶階層（L1/L2/L3）と Memory Mosaic

| 層 | 役割 | 実装メモ |
|----|------|----------|
| L1 | 60–90s のリングバッファ。未来記憶・成功感覚の速報値。 | `recent_future_feels` として保持し、成功フラグ／reuse_count を更新。 |
| L2 | EMA 統合キャッシュ。似たエピソードをクラスタ化。 | PDC の `sample_success_vector` / `sample_future_template` は主に L2 を中心にサンプリング。 |
| L3 | StoryGraph / MemoryPalace。長期文化・人格記憶。 | 昇格条件：success_reuse_rate 高＆guard_rate 低。物語免疫系として働く。 |

API 例：`memory.promote_to_L2(ep)`、`memory.promote_to_L3(cluster)`。

## 8. 実装ノート（疑似コード）

```python
# pdc/core.py
pdc = ProspectiveDriveCore(cfg)
proj_phi = pdc.project_from_phi(phi_t)
m_t, m_past, m_future = pdc.step(proj_phi, psi_t, memory)
T_t = pdc.compute_temperature(rho_t, persona_mode)
E_story = cos(m_t, m_future)
guard.update(E_story, pdc.jerk())

# policy/decision.py
scores = []
for cand in candidates:
    base = cand["score"]
    score = score_action(cand, base, m_past, m_future, cfg, T_t)
    scores.append(score)
selected = boltzmann_sample(candidates, scores)
```

Loss 側は既存の `forward_replay`, `cos(m_t, m_past_success)` などをそのまま利用し、PDC が提供する `m_t` で調整する。

## 9. まとめ（Prospective Drive Core）

- PDC は EQNet DNA の上に載る Self-Consistent Future Memory Module であり、`m_t` / `m_past_success` / `m_future*` を介して「妄想エンジン」を安全に駆動する。
- 行動決定は `Φ → PDC → Policy/LLM` という経路で mood-led にバイアスされる。RAG 的な「質問→検索」ではなく、気分→記憶→行動の流れを実装できる。
- Guard・記憶階層・人格モードなど既存の設計資産をそのまま接続できるため、過去の実装を捨てずに EQNet の“生き物感”を強化できる。
