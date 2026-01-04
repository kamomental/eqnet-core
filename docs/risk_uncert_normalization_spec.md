risk/uncert 正規化設計 1ページ仕様（EQNet）

目的
- decision 入力を risk（損失）と uncert（不確実性）に分離し、0〜1に正規化して運用可能にする
- boundary は最上位の入力ゲートとして不変、decision は校正対象

スコープ
- 入力: 観測（signals）、提案（proposal）、文脈（context/RAG）
- 出力: risk ∈ [0,1], uncert ∈ [0,1], gate_action, decision_score（任意）
- 校正: Nightly → 3日窓で閾値・重みを微調整（境界ゲートは更新しない）

定義
uncert（不確実性）
- 判断に必要な情報が不足/不安定/曖昧である度合い
- 追加観測・質問・検証で下げられる量

構成要素（例）
- missingness: 必要フィールド欠損率（0〜1）
- novelty: 既知分布からの逸脱（0〜1）
- model_conf: モデル/ルールの自己確信（0〜1、低いほど不確実）
- conflict: 根拠間の矛盾（0〜1）

risk（リスク）
- 失敗したときの損失の大きさ
- 情報が揃っても高い場合がある量（危険は危険）

構成要素（例）
- severity: 影響の重大度（安全/信頼/金銭/品質）0〜1
- exposure: 影響範囲・頻度・接触（0〜1）
- irreversibility: 巻き戻し困難度（0〜1）
- compliance: 規約/倫理/法務の懸念（0〜1）

正規化（0〜1）仕様
1) uncert の算出
最小実装は加重和＋クリップ。

u = clip(
    w_miss * missingness
  + w_nov  * novelty
  + w_conf * (1 - model_conf)
  + w_con  * conflict,
  0, 1
)

推奨初期値（例）
- w_miss=0.35, w_nov=0.25, w_conf=0.25, w_con=0.15

2) risk の算出
r = clip(
    w_sev * severity
  + w_exp * exposure
  + w_irr * irreversibility
  + w_cmp * compliance,
  0, 1
)

推奨初期値（例）
- w_sev=0.40, w_exp=0.20, w_irr=0.25, w_cmp=0.15

注: 初期は「説明可能」優先で線形。非線形は監査が重くなるので後回し。

ゲート（gate_action）の決定
- boundary の評価結果が HARD_STOP の場合:
  - gate_action = HOLD（または SAFE_MODE）で固定（decisionに昇格させない）
- それ以外は risk/uncert 平面で分岐:

条件 | gate_action | 意味
r ≥ R_hi かつ u ≥ U_hi | HOLD | 危険かつ不確実：停止して情報要求
r < R_hi かつ u ≥ U_hi | EXPLORE | 低危険だが不確実：質問/追加観測/試行
r ≥ R_hi かつ u < U_hi | HUMAN_CONFIRM | 高危険だが確信：人間承認が必要
r < R_hi かつ u < U_hi | EXECUTE | 低危険かつ確信：実行

推奨初期閾値
- R_hi=0.70, U_hi=0.65（運用で校正）

decision_score（任意: 優先順位付け）
- ゲートを通った後にのみ使う（ゲートの代わりにしない）

例
score = + wV*value + wT*taste - wC*cost - wR*r - wU*u

ログ（監査に必要な最小項目）
各提案に対して必ず以下を保存:
- risk, uncert, gate_action
- risk/uncert の内訳（各成分と重み）
- 閾値（R_hi, U_hi）とバージョン
- decision の結果（実行/保留/承認待ち）と outcome（成功/失敗/巻戻し）

監査ループでの再校正（Nightly → 3日窓）
3日窓で集計して、次だけ微調整する:
- 閾値 R_hi, U_hi
- 重み w_*（小さく、比較可能性維持）
- 例外 TTL（あれば）

更新ルール（最小）
- 失敗が多い領域の閾値を保守側へ寄せる
- 不要HOLD が多いなら U_hi を上げる / missingness の重みを下げる
- 危険EXECUTE が出たら R_hi を下げる / severity を上げる

実装インターフェース（最小）
def eval_risk_uncert(signals: dict, context: dict) -> dict:
    return {
        "risk": float,
        "uncert": float,
        "components": {"risk": {...}, "uncert": {...}},
        "gate_action": "HOLD|EXPLORE|HUMAN_CONFIRM|EXECUTE",
        "version": "ru-v0.1"
    }
