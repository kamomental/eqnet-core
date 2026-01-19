# DerivedMetrics Generator Runbook v1.0.0

## Overview
- Plaintext-free logging: metrics, IDs, hashes only
- Append-only outputs
- No external injection: no CLI options for overriding metric values or window_ms
- cfg hash is embedded in `calc_version` as `v1.0.0+cfg.<hash>`

## Commands
Batch (backfill):
```powershell
python derived_metrics_generator\emit_derived_metrics.py `
  --in events.jsonl `
  --out events.derived.jsonl `
  --mode batch `
  --windows short,mid,long `
  --strict-integrity true
```

Follow (ops):
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

## Verification (most common failures first)
1) `events.audit.jsonl` exists and grows
2) First audit entry has `action=observe`
3) After Ctrl+C, last audit entry has `action=stop`
4) From the 2nd audit line onward, `trace.integrity.prev_hash` is non-empty
5) Derived events include `calc_version` with `+cfg.<hash>`

## reason_code_hash catalog (v1.0.0)
- OBSERVE_START: generator started reading input
- STOP_NORMAL: normal exit (batch complete or normal shutdown)
- STOP_INTERRUPT: KeyboardInterrupt (Ctrl+C)

## Troubleshooting
Audit file not created:
- Check `--emit-control-audit true` and `--audit-out <path>`

prev_hash stays empty:
- Ensure the audit file is append-only and non-empty
- Verify the last line is valid JSON

cfg hash changes unexpectedly:
- `config/derived_metrics.yaml` changed or was replaced

## Invariants (do not break)
- No arbitrary window_ms input
- No metric override/injection CLI options
- Append-only output for derived_metrics and control_audit
- cfg changes must surface as `calc_version` suffix
