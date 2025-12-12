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
