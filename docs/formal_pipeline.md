# EQNet Formal Reasoning Pipeline（Litex × Lean × Python）

EQNet の感情・感性・愛情レイヤーを安全かつ直感的に扱うため、形式的変形（Litex）と厳密証明（Lean）、そして Python オーケストレーションを組み合わせたパイプライン構想をまとめる。

## 目的
- 感情状態や制御ノブの書き換え（「どう変化させたいか」）を Litex で記述し、直感的に試行する。
- 重要な不変量（安全境界、発散しないことなど）を Lean4 で証明し、全入力に対して正しいことを保証する。
- Python から両者を統合し、CI テストや自動検証に組み込む。

## プラグイン構造案
- `litex_rewrite` プラグイン  
  - 入力: EQNet の状態 `E`, `Σ`, `love_mode`, chaos 指標など。  
  - 役割: `rewrite_rules/*.ltx` を pylitex で実行し、推奨行動や制御差分を生成。  
  - 失敗時: スキップして従来のヒューリスティック制御へフォールバック。
- `lean_safety` プラグイン  
  - 入力: `litex_rewrite` やポリシーが提案した制御案。  
  - 役割: `proofs/*.lean` の補題を `lake` 経由でチェックし、安全領域を越えないことを保証。  
  - 失敗時: 安全モードへ戻す／警告ログを出す。

## 推奨ディレクトリ構成
```
formal-reasoning/
├─ litex_demo.py          # pylitex を使った書き換えデモ
├─ prove_controls.py      # Lean コード生成＋ lake build
├─ lakefile.lean          # Lean プロジェクト設定
├─ lean-toolchain         # Lean4 バージョン指定
├─ rewrite_rules/         # 感情・感性の変形ルール（*.ltx）
├─ proofs/                # Lean 補題（*.lean）
└─ logs/                  # 書き換え・証明ログ
```

## Litex（書き換え）サンプル
```python
from pylitex import Rewriter, parse_expr

rules = ["pause_relax", "warmth_soften", "micro_affirmation"]
rw = Rewriter(rules)

expr = "love_mode.high + anxious_attachment -> reassure + preview_next + proxemics_up"
ast = parse_expr(expr)
nf, steps = rw.normalize(ast, return_steps=True)

for step in steps:
    print(step)
```
- 例: `Σ` が高い + `attachment == anxious` なら `preview_next` を追加するようなルールを記述できる。

## Lean（厳密証明）サンプル
```lean
import Mathlib.Tactic

namespace EQNetSafety

lemma keep_lambda_bound
  (λ₀ λ₁ : ℝ) (h₀ : 0.02 ≤ λ₀) (h₁ : λ₁ ≤ 0.10)
  (Δ : ℝ) (hΔ : |Δ| ≤ 0.01) :
  0.02 ≤ λ₀ + Δ ∧ λ₀ + Δ ≤ 0.10 := by
  constructor
  · have := sub_le_iff_le_add.mpr h₀
    simpa using this
  · have := add_le_add h₁ (abs_nonneg Δ)
    simpa using this

end EQNetSafety
```
- 目的: Litex で調整された λ が安全範囲 `[0.02, 0.10]` を超えないことを証明する。

## Python オーケストレーション
```python
import subprocess
from pathlib import Path

LEAN_FILE = Path("proofs/LambdaSafety.lean")
LEAN_TEMPLATE = """
import Mathlib.Tactic
namespace EQNetSafety
-- 補題を追記
end EQNetSafety
"""

def write_lean():
    LEAN_FILE.write_text(LEAN_TEMPLATE, encoding="utf-8")

def run_lake():
    subprocess.run(["lake", "build"], check=True)

if __name__ == "__main__":
    write_lean()
    run_lake()
```
- rewrite ログと Lean 補題を連携して CI で検証する。

## CI 組み込みの流れ
1. `litex_rewrite` が推奨制御を生成し、`logs/litex_steps.jsonl` に記録。
2. Python スクリプトがログをもとに Lean 補題を生成し、`lake build` で証明。
3. 成功: パイプラインを継続。失敗: 差し戻し／フォールバック。

## EQNet の文脈での使い分け
- Litex: `Σ` や `love_mode` の遷移ルールを柔軟に書き換え、日常運用の「直感的調整」を支える。
- Lean: Chaos Taper、λ 範囲、禁則行動など、破ってはいけない安全境界を厳密に証明。
- Python: プラグインとして挟み込み、必要なときだけ呼び出す（重い処理はオフライン or バッチで回す）。

## EQNetにおけるLitexとLeanの役割
- Litex（pylitexミドルウェア）は `Σ` や `love_mode` の遷移ルールを人とAIが理解しやすい書き換えとして表現し、温度調整や共感モード切替を直感的に試作できる。
- Lean4（lake + mathlib）は Chaos Taper の安定性、Lyapunov帯域、禁則ガードなど、EQNetが守るべき安全不変量を厳密に証明する役割を担う。
- Python オーケストレーションはプラグインを仲介し、`config/plugins.yaml` のトグルで ON/OFF を制御しつつ、書き換えログ→証明→CI 連携を自動化する。

## 自動チューニングの三段ギア（Replay → Canary → Lean Gate）
- **tunable.yaml**: チューナブルな制御パラメータ（warmth/pause/K_local/kappa/noise など）の範囲とセグメント（persona × social）を宣言し、目的関数や制約を数値化。
- **scripts/autotune_replay.py**: 過去ログを再生し、Litex rewrite → Lean enforce を通した評価関数 J(θ) を計算。候補 θ を探索し `data/tuning/best_theta.yaml` と `candidates.jsonl` に出力。
- **ops/canary_bandit.py**: Thompson Sampling によるカナリア配布。Lean の修復回数や QoR 指標を報酬にして勝ち残りを昇格させる。
- **Lean Gate**: `plugins/formal-reasoning/prove_controls.py` が CI で rate-limit / λ 上限 / containment 順序を証明し、ランタイムは同ファイルの `enforce_invariants` で最小修復または containment を適用。

## 導入ステップ（推奨）
1. Litex のサンプルルールを EQNet の感情制御（pause、warmth 等）に適用し、有用性を検証。
2. 安全クリティカルなノブ（λ、Σ 変化量など）から Lean 補題を導入し、CI に組み込む。
3. rewrite ログ → Lean 補題生成のテンプレートを整備し、長期的には自動化を進める。

これらを整えることで、EQNet は「感情（E）」「感性（Σ）」「愛情（love_mode）」を一貫した観測→推定→制御フレームに整理し、優しくも愛嬌のあるキャラクターを安全に運用できます。
