# DerivedMetrics 生成器 Runbook v1.0.0（日本語）

## 何のための仕組みか（課題と目的）
課題:
- 指標の「正しさ」を人の設定に頼ると、運用中にゆがむ
- 変更の理由が後から追えず、監査が弱い

目的:
- 指標はログから派生させる
- 派生の規約は固定し、変更は系統分岐として残す
- 実行そのものも監査対象にする

## これで何ができるようになったか（本質）
- 値はすべてログから派生する
- 派生の規約は versioned で固定される
- 規約変更は系列分岐として記録される
- 実行そのものが監査される
- 再起動しても履歴は壊れない

つまりこれは「指標を計算するツール」ではなく、
「価値観の変化を後から検証できる計測装置」になる。

## 概要
- 平文ゼロ: 数値/ID/ハッシュのみ
- 追記のみ（append-only）
- 外部注入不可: 指標値や任意 window_ms の上書きは不可
- `calc_version` に `v1.0.0+cfg.<hash>` を刻む

## 実行
Batch（バックフィル）:
```powershell
python derived_metrics_generator\emit_derived_metrics.py `
  --in events.jsonl `
  --out events.derived.jsonl `
  --mode batch `
  --windows short,mid,long `
  --strict-integrity true
```

Follow（運用）:
```powershell
python derived_metrics_generator\emit_derived_metrics.py `
  --in events.jsonl `
  --out events.derived.jsonl `
  --mode follow `
  --windows short,mid,long `
  --strict-integrity true `
  --emit-control-audit true `
  --audit-out events.audit.jsonl
```

## 確認（失敗しやすい順）
1) `events.audit.jsonl` が作成され、追記されている
2) 先頭付近に `action=observe` がある
3) Ctrl+C 後、末尾付近に `action=stop` がある
4) 2行目以降の `trace.integrity.prev_hash` が空でない
5) derived 側の `calc_version` に `+cfg.<hash>` が付与されている

簡易チェック（Windows）:
```powershell
findstr /c:"\"action\":\"observe\"" events.audit.jsonl
findstr /c:"\"action\":\"stop\"" events.audit.jsonl
```

## reason_code_hash 一覧（v1.0.0）
- OBSERVE_START: 生成器が入力読み取りを開始
- STOP_NORMAL: 正常終了（batch 完了 / 通常停止）
- STOP_INTERRUPT: 割り込み停止（Ctrl+C）

## トラブルシュート
監査ログが出ない:
- `--emit-control-audit true` と `--audit-out <path>` を確認

prev_hash が空のまま:
- 監査ファイルが追記モードであること
- 最終行がJSONとして正しくパースできること

cfg hash が変わる:
- `config/derived_metrics.yaml` が変更されている

## 不変条件（守る）
- 任意 window_ms 指定は不可
- 指標値の override / inject は不可
- derived_metrics / control_audit は追記のみ
- cfg 変更は `calc_version` の系統分岐として可視化

## 最後に（設計者視点のひとこと）
この実装は最適化ではなく、逃げ道を一つずつ塞いだ構成。
「正しく動くか」ではなく「正しくしか動けないか」を目指している。
