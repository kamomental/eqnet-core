# Ops Docs

Runbooks, audits, and operational checklists.

- Forgetting runbook: `docs/forgetting_runbook.md`
- Risk/uncertainty checklist: `docs/risk_uncert_impl_checklist.md`
- Ops log template: `docs/ops_log_template.md`
- Fastpath release notes: `docs/fastpath_release.md`
- Frontier/phase checklists: `docs/frontier_phase1_checklist.md`
- NDCG eval: `docs/ndcg_eval.md`
- Weekly convergence completion: `docs/ops/weekly_convergence_completion.md`
- Love preset runbook: `docs/ops/love_preset_runbook.md`
- Green/Core整合方針（ADR相当）: `docs/ops/green_core_alignment.md`
- Weekly love preset checklist: `docs/ops/weekly_love_preset_checklist.md`
- ACE-aligned weightless operation: `docs/ops/ace_weightless_operation.md`
- ACE準拠 重み更新なし運用 (JA): `docs/ops/ace_weightless_operation_ja.md`

## Artifact Policy

- Treat nightly outputs as runtime artifacts, not source files.
- Do not commit `telemetry/audit/nightly_audit_*.json`, `trace_v1/`, or `rule_delta.v0.jsonl`.
- Validate structure and thresholds through tests, not by pinning environment-dependent JSON snapshots.
