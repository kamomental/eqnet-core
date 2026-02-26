# ACE準拠: 重み更新なし運用（eqnet-core）

## 目的

この文書は、`eqnet-core` を **fine-tuningなし** で改善するための運用方針を定義します。  
ACE（Agentic Context Engineering）の要点を、既存の
`trace_v1 -> nightly audit -> runtime overlay` に接続します。

## 3段構成（できる3段）

1. **観測する**  
`trace_v1` に `tool_call|micro_outcome` を残し、`reason_codes`/`fingerprint`/`success`/`cost` を契約キーで記録する。

2. **反省して差分化する**  
nightly audit で成功/失敗を集計し、`rule_delta.v0`（`add|modify|disable`）へ append-only で反映する。

3. **実行時に必要分だけ注入する**  
`scenario/world_type/gate_action` 条件で一致した delta のみ overlay 適用し、結果を再び trace に戻す。

## ACEの核（eqnet-core向け）

- 改善対象は重みではなく運用コンテキスト
- 全文再生成ではなく局所delta更新
- 自由文ではなく契約キー中心
- 常時フル注入せず、条件一致のみ注入
- 劣化/矛盾は nightly で `YELLOW/RED` 判定

## セキュリティと運用規律

- 平文理由より `reason_codes` を優先
- 秘匿情報は trace/delta に保存しない
- 契約キー欠落は監査失敗扱い
- 直接反映より `shadow -> canary -> on` を優先

## 最小チェックリスト

- [ ] `trace_v1` が `tool_call|micro_outcome` を記録
- [ ] nightly が `micro_outcome_coverage/tool_failure_modes/delta_conflict_count` を出力
- [ ] `rule_delta.v0` が append-only で管理される
- [ ] runtime が条件ベースで delta を選択注入する
- [ ] fail-closed をテストで保証する

