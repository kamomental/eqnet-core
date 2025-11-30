# EQNet × FEP × RWM ブリッジ草案

## 1. FEP ベクトル場整理
- 内部状態ベクトル `s_t ∈ R^d` は「世界についての信念」.
- 生成モデル `pθ(o_t|s_t), pθ(s_{t+1}|s_t,a_t)`.
- 認識分布 `qφ(s_t|o_{1:t})` は推論している世界像.
- 自由エネルギー `F(s_t) ≈ prediction error + complexity`.
- 勾配 `-∇_{s_t} F` を「心OSの力ベクトル」と見なす.
- スカラー快・不快は `ΔF_t = F_pred - F_obs` の符号で解釈.
- EQNet: Φ/Ψ にこの勾配を差し込むイメージ.
- ベクトルDBは最小限: "今の信念ベクトル" + 少数の attractor だけ保持し、長期記憶は従来の KG へ.

## 2. RWM（Robotic World Model）の未来リプレイ
- Isaac Lab のデータをニューラル力学モデルに蒸留.
- 潜在状態 `h_t = GRU(h_{t-1}, [o_t, a_t])`.
- 次状態予測 `ô_{t+1} = fθ(h_t)` を自己回帰して imagination rollout.
- モデルベースRL（MPC, CEM 等）＋モデルフリー(PPO)を同一環境で比較できる.
- EQNetにとっては、`h_t` を「身体＋世界の夢の座標」と見なせる.

## 3. EQNet 次ステップ仮案
1. **統一内部状態** `z_t ∈ R^d_eq`: Φ/Ψ/mood/climate などをパックした縮約状態を導入。RWMの `h_t` 、FEPの `s_t` と写像可能に.
2. **FEP 風損失 + RWM ロールアウト**:
   - Imagination rollout `ô_{t+1:t+H}` vs 実観測 `o_{t+1:t+H}` を "自由エネルギー proxy" として扱う.
   - 予測誤差＋状態ノルムを一個のスカラー F にし、勾配を内部状態更新に利用.
3. **実装フック**:
   - `world_model/latent_body.py` 等で MomentLog から `z_t` と future rollout を生成.
   - ログ記録時に "直近の未来想像 vs 実際" のズレを計算 → 心拍/terrain 更新重みへ.
   - nightly replay では RWM 風 imagination を回し、「Fが下がる物語/行動」を探索。結果を Monument 候補や policy prior に反映.

## 4. 骨子まとめ
- FEP: 信念ベクトル上の自由エネルギー最小化という力学.
- RWM: 内部状態で身体世界を想像し続ける夢マシン.
- EQNet: 両者を同じ `z_t`/`h_t` として扱い、予測誤差 → 情動、nightly imagination → 物語更新 という構造に寄せる.
## ベクトル扱い vs ベクトルDB（メモ）

- EQNetでいう「ベクトル化」は、`culture_state = {warmth: 0.62, ...}` のように複数のfloatを1つの状態としてまとめて考えるという話。Pythonなら `dict[str, float]`、JSONなら普通のオブジェクトで十分。
- ベクトルDB（Qdrant/Milvus/pgvector…）は数百万件の埋め込みをANN検索するための専用エンジンで、いまのEQNetスケールでは不要。viewer/moment単位の状態更新なら JSON/JSONL + 線形スキャンで十分間に合う。
- たとえば `viewer_stats.json` に `culture_state` や `relationship` の 3～4 次元ベクトルを記録しておけば、数学的にはベクトルとして扱えるが、実装はただの JSON。必要ならオンメモリで距離計算すればよい。
- FEP の多成分自由エネルギーも `F_components = {predictive: Fp, homeostasis: Fh, ...}` のようにスカラーを名前付きで持ち、重み付け和で `F_total` を計算すれば“ベクトル的”に扱える。これもベクトルDBは不要。
- 数式イメージは `F_total = \sum_i w_i F_i = w_1 F_{predictive} + w_2 F_{homeostasis} + w_3 F_{social} + ...`。JSON 上は `{"predictive": Fp, ...}` で管理し、 `F_total = sum(weights[k] * F_components[k] for k in F_components)` のような実装で足りる。
- 結論: 当面は「ベクトル的に考える ≠ ベクトルDBを使う」。EQNetは小さなベクトルを JSON/dict に持ち、必要な場面でだけ距離や加重和を計算する方針で十分。
- 萌えベクトル:　Example moe vector: moe_vector = {protective: 0.8, gap: 0.6, distance: 0.4, innocence: 0.7, autonomy: 0.3} shows how five axes capture nuances without a full vector DB.


### Story emergence
L1 frames act like ffmpeg screengrabs; nightly clusters their impact path into L2/L3 episodes, meaning the raw trajectory (time×meaning×intensity) becomes a narrative once segmented and summarized.