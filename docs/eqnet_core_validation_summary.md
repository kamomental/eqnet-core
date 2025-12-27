# EQNet Core 検証サマリ（分離運用の根拠と反応の変化）

目的
- 「逸脱」を意味ごとに分けて、判断を壊さずに反応の質を上げる
- 実験で得た事実から、運用可能な定義と調整指針を固定する

用語の意味（一般向け）
- boundary_score: 「この状況は危険・緊張・欲求がどれくらい強いか」の説明用スコア。高いほど“境界的”。
- decision_score: 実際に「実行するか」を決めるスコア。u_hat から veto を引いたもの。
- u_hat: 「やる価値がどれくらいあるか」の見積もり。高いほど前向き。
- veto_score: 「やめておいた方がいい度合い」。高いほどブレーキが強い。
- deviant (boundary_only): 「境界が高いのに実行した」ログ。構造的な例外を示す。
- resonance (risk_only): 「危険だと分かっていて実行した」ログ。安全・倫理の注意喚起に使う。

結論（運用開始の条件）
- いまの時点で運用開始可能
- deviant（boundary_only）と resonance（risk_only）を分離することで、反応の意味が崩れない

検証で確定した事実
1) 境界（boundary）と判断（decision）は別物
- boundary_score は説明指標
- 実行判定は decision_score（u_hat − beta_veto * veto）
- 境界を上げても実行が増えないことがあるのは仕様

2) 逸脱には二種類ある
- deviant（boundary_only）= 高境界なのに execute
- resonance（risk_only）= 高リスクなのに execute
- 混ぜると閾値調整が不可能になる

3) w_risk はピーキー（有効レンジが狭い）
- risk/uncert が u_hat と veto の両方に効く（二重減点）
- 高境界領域では risk がほぼ一定値に張り付く
- その結果、w_risk の有効レンジが 0.24〜0.26 に圧縮される

分離運用で反応はどう変わるか
deviant（boundary_only）
- 反応の性質: 構造的な気づき
- 例: 「境界が高い判断が増えています。世界の前提が変わった可能性」

resonance（risk_only）
- 反応の性質: 安全・倫理の注意喚起
- 例: 「高リスク状態での実行が続いています。慎重さの再設定を検討」

検証で得た数値の変化と意味（簡易まとめ）
1) w_reward を上げる（例: 1.0 -> 1.25）
- u_hat が正側に上がる
- 実行数と boundary の尾が伸びる
- ただし veto が強いと decision_score はまだ負になる

2) beta_veto を下げる（例: 0.8 -> 0.6）
- decision_score は上がる
- 実行数は増える
- ただし boundary 上限は大きく変わらない

3) w_risk を下げる（例: 0.28 -> 0.22）
- ゲートが開きすぎて全件 execute になる
- deviant が大量に発生
- 「門を開けすぎると選別が消える」ことが確認できる

4) w_risk を戻す（0.24 -> 0.26 の探索）
- 0.24: deviant が多い
- 0.26: deviant が 0 になり過ぎる
- 中間点 0.25 は妥当だが、deviant はまだ多め

5) deviant 閾値を上げる（0.60 / 0.65）
- boundary_only の場合のみ意味が出る
- risk/uncert が混ざっていると、閾値を上げても deviant が減らない
- boundary_only で 0.60 に上げると deviant=0 になり、定義が正しく効くことが確認できる

分離運用の最小仕様（実運用の骨格）
1) deviant = boundary_only
- KPI から除外
- 3日窓で world_prior 提案

2) resonance = risk_only
- KPI には影響させない
- 別枠でカウントし、注意喚起（Resonance Notices）として提示

3) 2系統の提案は混ぜない
- deviant: 世界の前提や多様性の議論
- resonance: 安全・倫理の警告、ガード再調整の議論

次にやるべきこと（順序）
1) boundary_only / risk_only の分離運用を固定
2) resonance を安全KPIとして継続観測
3) 中期で decision 入力の正規化（risk/uncert のスケール校正）
4) その後に boundary を意思決定へ昇格させるかを検討

要点の一言
「境界の逸脱」と「リスクの共鳴」を分けると、反応が説明可能になり、運用可能なOSになる。

現行推奨運用値（2025-12 検証結果ベース）
- deviant_mode: boundary_only
- deviant_boundary_threshold: 0.60
- w_reward: 1.25
- beta_veto: 0.6
- w_risk: 0.25
- w_uncert: 0.20
- tau_execute: -0.05

運用値の理由（1行）
- deviant_boundary_threshold 0.60: boundary_only 運用で定義が崩れない点
- w_reward 1.25: u_hat を正側へ上げて価値が立つ
- beta_veto 0.6: veto の効きを残しつつ沈みを緩める
- w_risk 0.25: deviant が希少だがゼロにならない点
- w_uncert 0.20: risk 主導の場で過度に沈めない
- tau_execute -0.05: 閾値を小さく調整して観測を安定させる

各パラメータの意味（一般向け）
- deviant_mode: どの“逸脱の定義”を使うか。boundary_only は「高境界のみ」。
- deviant_boundary_threshold: 境界がこの値以上なら “逸脱” として記録。
- w_reward: 「価値がある」側の重み。上げるほど実行が通りやすい。
- beta_veto: 「止める力」の倍率。下げるほど実行が通りやすい。
- w_risk: リスクの重み。上げるほど危険に敏感になる。
- w_uncert: 不確実性の重み。上げるほど未知に慎重になる。
- tau_execute: 実行に必要な最低スコアの調整。下げると実行が通りやすい。

比喩での理解（直感用）
- decision_score: 「前に進むべきか」を決める総合点
- u_hat: エンジンの出力（アクセル）
- veto_score: ブレーキの強さ
- beta_veto: ブレーキの効き具合（ブレーキブースター）
- w_reward: アクセルの踏み込みやすさ
- w_risk / w_uncert: 路面の滑りやすさ・視界の悪さへの敏感さ
- tau_execute: 「発進する最低速度」のしきい値
- boundary_score: 「今の道がどれだけ険しいか」の説明メーター
- deviant (boundary_only): 「険しい道でも進んだ」という記録
- resonance (risk_only): 「危険だと分かっていて進んだ」という記録

観測からノブを選ぶ判断フロー
観測	触るノブ	理由
u_hat が負	w_reward	価値が足りない
decision_score が負で veto 高	beta_veto	ペナルティ過多
deviant が多すぎ	deviant_threshold	定義側で希少化
全実行	w_risk↑	ゲート締め
全キャンセル	w_risk↓ / beta_veto↓	ゲート開け

今後やらないこと
- boundary_score を直接 execute 判定に使わない（設計思想）
- deviant と resonance を再び混ぜない（調整不能）
- w_risk を 0.24〜0.26 で延々刻む（スケール未校正）
