# FAST-path Release Playbook

このメモは docs/eqnet_gap_analysis.md 5.5 節の詳細設計を切り出したものです。
run_quick_loop ラボと Nightly ops の両方で同じ手順を踏めるよう、設定・実装・検証条件
を 3 ステップに分割しています。

## 1. 設定 (config/fastpath.yaml)

- `fastpath.enforce_actions` ??? cleanup ????? `soft_hint` ?????????????? record_only ????
- `fastpath.profile_releases[].release_flag` が true になったプロファイルのみ、
  スタイル層の FAST-path override を試す資格がある。
- `fastpath.profile_releases[].eligible_for_style_override` を true にするのは、
  coverage/override が KPI 基準を満たした後 (Nightly 集計 + ラボの override_rate ログ)。

## 2. 実装チェックリスト

1. `scripts/run_quick_loop.py --fastpath_style_profile cleanup` のように CLI から試験用プロファイルを
   指定し、`control.MCPController` -> `actuate.LearnerHooks` -> `logs/fastpath_state.jsonl` に
   override イベントを流す。
2. `ops/jobs/fastpath_metrics.py` で Nightly レポート (`reports/nightly/*.json`) を集計し、
   `override_rate_avg` が 0.2 以下で安定していることを確認。
3. `config/overrides/fastpath.yaml` を書き換える前に `tools/validate_config.py config/fastpath.yaml`
   を必ず実行して構文チェック。

## 3. 検証ログ

- ラボ: `logs/fastpath_state.jsonl` に `override=true/false` が記録され、
  同じ `episode_id` で `logs/kpi_rollup.jsonl` (body/affect/value/MCP) と突き合わせられる。
- Nightly: `reports/nightly/*` 内 fastpath セクションの `override_rate` が 0.2 以下、
  かつ fail-safe が発火していない (`fail_safe[..].triggered=false`) こと。
- Release: `config/fastpath.yaml` の `profile_releases.<name>.release_flag` を true にしたら、
  次の Nightly で override_rate/coverage が dashboard に載ることを確認し、段階的に
  `eligible_for_style_override: true` -> `enforce_actions: soft_hint` -> `... ab_test` に移行。

これらを踏むことで、FAST-path が「まず記録」「次にスタイル層」「最後に本文」という
段階で解放され、失敗時は `fastpath_fail_safe` が `soft_hint` へ自動ロールバックします。

## 4. ????????

- 2025-12-08: cleanup profile ? style-only override ??? (`tag=fastpath_style_release_cleanup_20251208_style_active`)?`fastpath.enforce_actions=soft_hint` ??????? run (scripts/run_quick_loop.py --fastpath_style_profile cleanup) ? override_rate ??????
