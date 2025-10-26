# DreamLink (G2L) Spec v0.15

## 1. Overview
- Purpose: bridge EQNet の内的場 `E(x,t)` と Reconstructive Autoencoder (RAE) の潜在 `z(t)` を結び、意味→象徴→映像を連続生成する。
- Scope: Green 応答の多尺度特徴を latent 変換に写す G2L、RAE デコーダの制御、安全ノブ (H/R/κ/W)、Edge/Service/Gov 接続。

## 2. Runtime Topology
```
Stimuli → EQNet Core ──→  E(x,t), H, R, κ, W
                    └─→ G2L (Green→Latent) ───→ z(t)
Seed (text/image/audio) → Representation Encoder R → μ(seed)
z(t) = μ(seed) + Δz(E) + η(t) → RAE Decoder → Frame I(t)
```
- Edge: μ(seed) 計算、G2L 推論、RAE デコード (≤180ms)。
- Service: 高解像レンダ、要約、差分パッチ。
- Gov: 安全・Spoiler・レーティング。

## 3. G2L Dynamics
1. 多尺度スペクトル `h_t = ψ(MS-FFT(E_t), H_t, R_t, κ_t, W_t)`  
2. 低ランク射影 `Δz_t = A·h_t + B·σ(h_t) ⊙ ε_t` (`rank << dim(z)`)  
3. OU 雑音 `η_{t+1} = ρ·η_t + sqrt(1-ρ²)·ξ_t`, `ρ = exp(-Δt / τ_OU(H,R))`  
4. 潜在更新 `z_t = μ(seed) + Δz_t + η_t`  
5. Optional: κ タイル境界 / W 文化ゲインで視点・スタイルの局所編集。

## 4. G2L API
```python
sess = dream.init(seed={"text": "...", "image": None}, culture_profile="anime_senpai_calm",
                  runtime={"fps": 12, "seconds": 20})
for t in range(T):
    E_t, metrics = eqnet.step(stimulus_t)  # metrics = {H,R,kappa,W}
    frame = dream.step(sess, E_t, metrics) # returns decoded frame
summary = dream.finalize(sess, mode="storyline+stills")
```
- Config: `config/dream.yaml` (rank, latent_dim, τ_OU, RAE path, micro_noise strength)。
- Session state keeps `z`, `η`, μ(seed), cumulative metadata。

## 5. Training Strategy
### 5.1 RAE Decoder (pre-train)
- Backbones: U-Net / DiT / VQ decoder。  
- Loss: reconstruction (L1 + LPIPS) + representation consistency `||R(Ĩ) - R(I)||²` + temporal smoothness。  
- Data: 一般動画 + アニメ統計の二相学習。

### 5.2 G2L Distillation
- Teacher: 手設計ルール (v0.14) / EQNet イベント → latent edit の擬似ラベル。  
- Weak labels: H/R/κ/W + 外部情動分類で符号の監督。  
- Curriculum: 規則7 : 学習3から開始、安定後に学習比率を段階的に増やす。

### 5.3 Optional Fine-tuning
- Reinforcement or gradient-based QoR optimisation (関係性維持)。  
- 安全制約をペナルティに含める (H ≥ H_min, R ≤ R_max)。

## 6. Safety & Control
- H/R adaptive: 硬直 (R↑, H↓) → τ_OU短縮, ノイズ↑, κ<0。過分散 (R↓, H↑) → Bスケール↓, κ>0。  
- Spoiler gate: 作品進捗を超える暗喩は自動置換。  
- Rollback: S (snapshot) + J (journal) + preview で t★ の復元。  
- Logging: z(t), H/R/κ/W, μ(seed) の統計のみ (TTL, 暗号化)。

## 7. Evaluation Metrics
- Meaning Continuity: `||μ_{t+1} - μ_t||` (R 空間距離)。  
- Symbolic Aptness: EQNet イベント方向と視覚変化方向の一致率。  
- Naturalness: FVD, DMOS。  
- Cultural Fit: S_style 80–90, 禁則語 0。  
- Safety: H/R violation ≤1%, 不快スパイク回復 ≤60s。

## 8. Files & Modules
- `src/dream/g2l.py`: G2L モジュール (rank 32, τ_OU map, multi-scale FFT)。  
- `src/dream/rae_decoder.py`: デコーダ本体 (事前学習 or ローダ)。  
- `config/dream.yaml`: rank, latent_dim, τ_OU, decoder checkpoint, micro_noise。  
- `scripts/dream_generate.py`: EQNet ログを読み、DreamLink でフレーム生成 (Edge)。  
- `tests/integration/test_dreamlink.py`: 連続性・安全ノブの検証。

## 9. Open Issues
- ISSUE-006 DreamLink G2L Bridge (実装).  
- ISSUE-007 DreamLink Safety Bench (EMAC-based)。  
- ISSUE-008 Micro-noise VQ-VAE (optional)。  
- ISSUE-009 Perception Bridge (camera/mic) との同期保存。

