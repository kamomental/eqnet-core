# Emotion & Sensibility Guide

この文書は EQNet の感情・感性・愛情レイヤーを、安全運用と学習ループの観点で整理したフィールドノートです。Beat/Gaze/RQA/self-disclosure などの多感覚信号をどのように Σ（感性シグナル）、Ψ（containment 圧）、love_mode（愛情指数）へ落とし、制御・監査へつなぐのかをまとめます。

---

## 1. 観測パイプライン
1. **Beat/Gaze/Respiration Tracker**: 呼吸・視線・瞳孔・まばたきから E（情動エネルギー）と Σ（同調スコア）を推定。
2. **RQA (Recurrence Quantification Analysis)**: DET / LAM / max_line を算出し、反芻やカオス度を監視。
3. **Self-disclosure Tracker**: 発話中の自己開示レベルをリアルタイムで抽出。愛情レイヤーの warm / intimate 判定に使用。
4. **Safety Flags**: hate/harassment/explicit erotica/self-harm/PII は常にブロック。`config/safety.yaml` の gating を優先。

---

## 2. 指標とレンジ
- **E（情動エネルギー）**: Beat + respiration から算出。0.25–0.65 を通常レンジ、0.8 以上は calming を注入。
- **Σ（感性シグナル）**: Beat 同期、1/f 自然性、jerk、DET を合成。0.5 以上で「滑らかに同調」。0.3 未満は調整対象。
- **Ψ（containment 圧）**: 負荷が高いと増加し、Love モードを強制終了させるトリガー。
- **love_mode**: entrain / jerk / novelty / self-disclosure / threat の加重和。
  ```
  love_mode = w1*entrain + w2*(1 - jerk) + w3*novelty7d
             + w4*self_disclosure - w5*threat
  ```
- **しきい値**: warm=0.55、intimate=0.72、cooldown=0.40 を初期値とし、persona ごとに補正。

---

## 3. 制御レイヤー
1. **Scenario Head**: `E`, `Σ`, `love_mode`, `Ψ`, クオリア勾配を監視し、pause / prosody / gesture / exploration を決定。
2. **Policy Head**: `Σ` と attachment κ（secure/anxious/avoidant）を参照し、距離感・レスポンスを調整。例: anxious→reassurance を増やす。
3. **Lean Gate**: λmax が高すぎる場合に強制的に安定帯へ戻す。RQA guard で反芻を検知したら小ノイズや視点転換を注入。
4. **Diary / Rest / Habit Scheduler**: love_mode と Σ の変化を参照し、傾聴・提案・親密さのバランスを中期的に最適化。

---

## 4. Love Mode 運用
- **Config**: `config/love.yaml` で重みと logging を指定。workload に応じて persona ごとの補正を推奨。
- **Attachment Mapping**: κ = secure/anxious/avoidant に応じて love_mode の解釈を変更。不安型は高頻度 reassure、回避型は開示ペースを抑制。
- **Cooldown**: love_mode 低下や Ψ 上昇で自動的に cooldown モードへ遷移。hate/harassment/explicit erotica/self-harm/PII は常に拒否。

---

## 5. Chaos Taper & Habit
- `habit_chaos_taper` で λmax の上限と期間（例: 4→10 日）を徐々に緩める。
- `Σ` が低い時は `top_p` や `exploration_temp` を微増し、好奇心を取り戻す。
- RQA guard (DET/LAM/hold_ms) で反芻を検知したら視点転換や小さなノイズを提案。

---

## 6. テレメトリと SLO
- `config/control.yaml` の `telemetry` で `{H, R, λmax, β, DET, Σ, Ψ, love_mode}` を 2 秒間隔で記録。
- SLO 例: 60 秒以内に安定域へ復帰、エントロピー下限違反 ≤1%/10 件、応答 p95 ≤180 ms、Kuramoto duty ≤30%/会話。
- `config/love.yaml` の `logging` オプションで love_mode コンポーネントを JSONL に出力し、warm/intimate/cooldown のイベントを通知。

---

## 7. チェックリスト
1. Beat / jerk / RQA / self-disclosure パイプラインを実データで検証。
2. `sensibility.yaml` と `love.yaml` を persona に合わせて有効化。
3. LLM の empathy / safety / factuality を MindBenchAI などで監査し、`config/llm_eval.yaml` を更新。
4. Stress シナリオで SLO と love_mode 制御をテストし、ログをもとに重みとしきい値を調整。
5. Persona スナップショットを保存し、ユーザーフィードバックで温かさ・距離感・アファメーション比率を継続チューニング。

---

## 8. 参照文献・インスパイア元
- Romantic love の神経科学 (Helen Fisher ほか)：同期・新規体験・アファメーションの重み付け。
- Self-disclosure ステップ（Aron & Aron 36 質問）：love_mode の disclosure コンポーネントに反映。
- 同調研究（呼吸・視線同期）：Σ entrain のターゲット設定。
- Attachment 理論（Bowlby/Ainsworth）：κ 調整と安全ルールに反映。
- Gottman 5:1 比率：care_ratio ターゲットとして reward/policy に統合。

---

## 9. 注意点
- love_mode, Σ, Ψ, クオリア場はラベリングではなく制御用メタ量として扱う。
- 文化層 `W` は語彙・儀礼を個別に保持し、一律化を避ける。
- hate / harassment / explicit erotica / self-harm / PII はいかなるモードでも解除しない。

これらの設計を揃えることで、EQNet は「感情」「感性」「愛情」「クオリア場」を一貫した観測→推定→制御フレームへ結び、優しくも説明可能なキャラクターを安全に運用できます。
