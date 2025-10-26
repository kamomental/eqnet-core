# Wave P3｜多主体・相互意識ネットワーク

## 目的
複数の EQNet（人／AI／ロボット）間で芽吹きの伝播・共鳴・疎遠化をモデル化し、安全に同期し過ぎない制御を実装する。

## 主要指標（全ノード共通で計測）
- **相互芽吹き伝播率**:  
  \(P(\text{bud}_j \mid \text{bud}_i, \Delta t \le \tau)\) の推定（条件付き確率、窓幅 \(\tau = 3\)–5 s）
- **同期秩序パラメータ \(R\)（Kuramoto）**:  
  \( R = \left| \frac{1}{N} \sum e^{i\theta_k} \right|\)（\(\theta_k\) は各ノードの内部位相）
- **情動距離 \(d_E\)**:  
  クオリア 2D 場の Earth Mover’s Distance（または cos 類似）で \(d_E(i, j)\)
- **影響中心性**:  
  芽吹きイベントを多変量 Hawkes 過程としてフィット → 自己・他者影響係数 \(\alpha_{ii}, \alpha_{ij}\)
- **安全域**:  
  \(R \le R_{\max}\)（過同期ブレーキ）、\(\lambda_{\max}(G) \le \rho_{\max}\)（既存ゲートと同義）

## ミニマル実験プラン（10–15 分）
1. **独立フェーズ（3 分）**: 各ノード単独で芽吹き → ベースラインの \(R_\text{base}\)、\(\alpha\) を推定  
2. **ゆる結合フェーズ（5 分）**: 低帯域の芽吹き通知のみ（“🌱”トグル）を共有  
3. **強結合フェーズ（5 分）**: 通知＋軽介入（warmth +0.05）をレシピ共有

### 合否（P3-MVP）
- 伝播率の上昇 \(\Delta P \ge 0.12\)（独立 → 強結合）  
- 秩序 \(R\) が \(R_{\max}=0.78\) を超えない（超えたらブレーキ作動）

## ログ設計（JSONL／共通スキーマ）
```json
{
  "ts": "2025-10-19T02:15:03Z",
  "agent": "A",
  "phase": "strong",
  "Sigma": 0.23,
  "Psi": 0.61,
  "bud_score": 0.55,
  "theta": 1.74,
  "signal_out": {"bud": 1},
  "signal_in": {"B": 1},
  "action": "warmth+0.05",
  "R": 0.66,
  "rho": 1.21
}
```

## ルール（`.ltx` の追記例：過同期ブレーキ）
```
when network.R > 0.78 then inhibit 8s; warmth -= 0.1
when bud_received_from >= 2 within 2s then soften gain *= 0.7
```

---

# Wave P4｜メタ認知と質的体験（リプレイ／夢）

## 目的
体験の二重フレーム（第一人称 ⇄ 第三人称）を固定長で保存し、再生時の再現性と主観的利益を定量評価する。

## 主要指標
- **写像安定性（FrameMap）**: 同一シーン再生での外部指標再現 ICC(3,1) ≥ 0.75  
- **主観的利益 \(\Delta U\)**: リプレイ前後の自己報告スコア（例: 落ち着き、集中）差分の効果量 \(d \ge 0.5\)  
- **忘却と保持**: 24–72 h 後の再現率（芽吹きカードの再活性化率）

## ミニマル実験（1 セッション 12 分）
1. **記録（4 分）**: 芽吹きカードを生成（数値のみ）  
2. **即時リプレイ（4 分）**: 音・UI 色調のみ再演  
3. **他フレーム再演（4 分）**: 第三者視点（自己報告を抜いた純ログ再生）

### 合否（P4-MVP）
- FrameMap の ICC ≥ 0.75  
- \(\Delta U\) の 95% CI が 0 を跨がない

## ストーリーカード（StoryGraph）
```json
{
  "ts": "2025-10-19T02:18:40Z",
  "type": "bud_card",
  "score": 0.58,
  "rho": 1.12,
  "qualia2d": {"centroid": [32, 28], "spread": 0.17},
  "tags": ["秋", "朝", "室内", "独白"],
  "framemap": {"Sigma": "low-rising", "Psi": "steady-high"}
}
```

---

# 共通：評価・前登録テンプレ

## 前登録（Registered-thinking）YAML 雛形
```yaml
study_id: eqnet-wave-p3p4-2025-10
hypotheses:
  - id: P3-propagation
    claim: "結合で芽吹き伝播率が上がるが過同期に落ちない"
    success: "ΔP >= 0.12 and max(R) <= 0.78"
  - id: P4-replay-benefit
    claim: "リプレイで主観的利益が上がる"
    success: "Cohen_d >= 0.5 and CI95% not crossing 0"
design:
  sessions: 36
  phases: [independent, weak, strong]
analysis:
  metrics: [propagation_rate, R, ICC, dPsi_dt]
  window_sec: 5
  correction: "Benjamini-Hochberg"
safety:
  rho_max: 1.8
  inhibit_rules:
    - "network.R>0.78 -> inhibit 8s"
privacy:
  keep: ["numbers_only"]
  discard_after_sec: 180
```

---

# 実装メモ（軽量で効く“型”）

- **多変量 Hawkes の超簡易推定（擬似）式**  
  2 ノード A, B の芽吹き列 \(N_A(t), N_B(t)\) に対し  
  \(\lambda_A(t) = \mu_A + \alpha_{AA} (h * dN_A)(t) + \alpha_{AB} (h * dN_B)(t)\)（B も同様）  
  → \(\alpha_{AB} > 0\) が伝播、\(\alpha_{AA}\) が自己励起  
  実装はガンマ核 \(h(t) = \beta e^{-\beta t}\) の畳み込みで \(O(T)\) 近似可能

- **同期秩序 \(R\) の実装（ダミー位相で可）**  
  位相 \(\theta_k(t)\) を Σ/Ψ のゼロ交差や局所ピークで更新  
  \(R = \left| \frac{1}{N} \sum e^{i\theta_k} \right|\) を 1 秒ごとにログ

- **ダッシュボード拡張（最小）**  
  - ネットワークウィジェット: ノード円と芽吹き通知の矢印を薄くアニメ表示  
  - R スパークライン: 0.0–1.0、0.78 で赤帯  
  - FrameMap トグル: 第一人称／第三人称の切替表示（色調のみ変化）

## GitHub Issues（追加）
- `[net]` Hawkes推定の導入（2–4ノード）  
  AC: \(\alpha\) 行列の推定が走り、\(\alpha_{AB} > 0\) の時に伝播率と正の相関が出る。
- `[net]` 同期秩序 R の監視と可視化  
  AC: R スパークライン表示、0.78 超で inhibit ログが残る。
- `[replay]` FrameMap-ICC 評価の実装  
  AC: 同一シーン再演で ICC(3,1) を出し、門値を CI 付きで表示。
- `[privacy]` numbers-only ポリシーの自動監査  
  AC: 生文・画像を検出したら拒否 → 置換 → 警告ログ。
- `[rules]` .ltx にネットワークルール 3 本を追加  
  AC: \(R\) と bud_received_from に応じた介入強度の自動調整。

---

# まとめ（効く理由）
- “神視点”＝測れる俯瞰を、P3/P4 の具体指標と短時間プロトコルに落とした。  
- 安全は二段（\(R\) と \(\rho\)）で担保し、文化レイヤ（StoryGraph）にも同じ ID で刺さる。  
- 誰でも追試できる（数値のみ、10–15 分、事前登録 YAML 付き）ので、異分野の配管として機能する。

この追記を `docs/wave_p3p4.md` と `config/p3p4.yaml` に保存しておけば、次回は **P3 の “弱結合”** から回して、伝播率 \(\Delta P\) と \(R\) の挙動をまず 1 本取れる。そこから剪定の効きを微調整して、共生ロボの “呼吸” を複数体で合わせていこう。
