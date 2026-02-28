# Immune Signature HMAC Runbook

## Scope
- This runbook defines runtime behavior for `immune_signature_v`.
- Goal: keep signatures verifiable while avoiding plaintext leakage.

## Signature Modes
- `immune_signature_v=2`: HMAC-SHA256 (preferred)
- `immune_signature_v=1`: legacy digest fallback

## Environment Variables
- `EQNET_IMMUNE_HMAC_KEY_B64`: base64-encoded HMAC key material
- `EQNET_IMMUNE_HMAC_KEY_ID`: key identifier (for trace/audit correlation)
- `EQNET_ENV`: runtime environment (`production` or `prod` enables strict mode)

## Runtime Policy
- Production (`EQNET_ENV=production|prod`):
  - Missing key: fail-fast at startup
  - Invalid key: fail-fast at startup
- Non-production:
  - Missing/invalid key: fallback to `immune_signature_v=1`
  - Emit one startup warning (no key material in logs)

## Trace Contract
- Trace may include:
  - `immune_signature`
  - `immune_signature_v`
  - `immune_key_id`
- Never output key material or decoded key bytes.

## Key Rotation
- Rotate by changing:
  - `EQNET_IMMUNE_HMAC_KEY_B64`
  - `EQNET_IMMUNE_HMAC_KEY_ID`
- Keep `immune_key_id` stable during a rollout window, then switch atomically.

## Rate Window Semantics
- Turn-level telemetry (`detox_rate`, `reject_rate`): computed over current in-memory guard window.
- Nightly telemetry (`immune_guard.repeat_hit_rate` etc.): rollup over nightly operation records.
