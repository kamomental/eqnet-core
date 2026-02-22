#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eqnet.telemetry.eval_report_writer import EvalReportWriter, EvalReportWriterConfig
from eqnet.telemetry.proposal_links_writer import ProposalLinkWriter, ProposalLinkWriterConfig
from eqnet.telemetry.change_decision_writer import (
    ChangeDecisionWriter,
    ChangeDecisionWriterConfig,
)


def _iso_week_string_from_timestamp_ms(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _parse_json_arg(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("json arg must decode to object")
    return data


def _build_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    keys = set(before.keys()) | set(after.keys())
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta[str(key)] = float(a) - float(b)
    return delta


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(arg_value: str | None) -> Path | None:
    if arg_value:
        return Path(arg_value)
    env_value = os.getenv("EQNET_GOLDEN_REPLAY_MANIFEST", "").strip()
    if env_value:
        return Path(env_value)
    return None


def _verify_manifest(telemetry_dir: Path, manifest_path: Path) -> tuple[bool, str, str, str]:
    if not manifest_path.exists():
        return False, "", "", "manifest_not_found"
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except Exception:
        return False, "", "", "manifest_parse_error"
    if not isinstance(manifest, dict):
        return False, "", "", "manifest_invalid_type"
    if str(manifest.get("schema_version") or "") != "golden_replay_manifest.v0":
        return False, "", "", "manifest_schema_invalid"
    manifest_id = str(manifest.get("manifest_id") or "")
    manifest_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        return False, manifest_id, manifest_sha, "manifest_inputs_invalid"
    policy = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
    allow_empty_inputs = bool(policy.get("allow_empty_inputs", False))
    if not inputs and not allow_empty_inputs:
        return False, manifest_id, manifest_sha, "manifest_inputs_empty"
    for group in inputs:
        if not isinstance(group, dict):
            return False, manifest_id, manifest_sha, "manifest_group_invalid"
        paths = group.get("paths")
        hashes = group.get("sha256")
        if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
            return False, manifest_id, manifest_sha, "manifest_group_shape_invalid"
        for rel, expected in zip(paths, hashes):
            rel_path = str(rel or "")
            expected_hash = str(expected or "").lower()
            target = telemetry_dir / rel_path
            if not target.exists():
                return False, manifest_id, manifest_sha, f"input_missing:{rel_path}"
            actual = _sha256_file(target).lower()
            if actual != expected_hash:
                return False, manifest_id, manifest_sha, f"input_sha_mismatch:{rel_path}"
    return True, manifest_id, manifest_sha, "ok"


def _latest_canary_gate_decision(telemetry_dir: Path, proposal_id: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_key: tuple[int, int] | None = None
    line_index = 0
    gate_decisions = {"ACCEPT_CANARY", "REJECT", "ACCEPT_ROLLOUT", "ROLLBACK"}
    for path in sorted(telemetry_dir.glob("change_decisions-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                line_index += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("proposal_id") or "") != proposal_id:
                    continue
                if str(row.get("decision") or "") not in gate_decisions:
                    continue
                ts = row.get("timestamp_ms")
                if not isinstance(ts, int):
                    try:
                        ts = int(ts or 0)
                    except (TypeError, ValueError):
                        ts = 0
                key = (int(ts), line_index)
                if latest_key is None or key >= latest_key:
                    latest = row
                    latest_key = key
    return latest


def _canary_eval_link_count(telemetry_dir: Path, proposal_id: str) -> int:
    count = 0
    for path in sorted(telemetry_dir.glob("proposal_links-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("schema_version") or "") != "proposal_link.v0":
                    continue
                if str(row.get("proposal_id") or "") != proposal_id:
                    continue
                if str(row.get("link_type") or "") != "canary_eval":
                    continue
                count += 1
    return count


def _latest_canary_eval_report_id(telemetry_dir: Path, proposal_id: str) -> str | None:
    latest_eval_report_id: str | None = None
    latest_key: tuple[int, int] | None = None
    line_index = 0
    for path in sorted(telemetry_dir.glob("proposal_links-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                line_index += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("schema_version") or "") != "proposal_link.v0":
                    continue
                if str(row.get("proposal_id") or "") != proposal_id:
                    continue
                if str(row.get("link_type") or "") != "canary_eval":
                    continue
                ts = row.get("timestamp_ms")
                if not isinstance(ts, int):
                    try:
                        ts = int(ts or 0)
                    except (TypeError, ValueError):
                        ts = 0
                key = (int(ts), line_index)
                if latest_key is None or key >= latest_key:
                    latest_key = key
                    latest_eval_report_id = str(row.get("eval_report_id") or "")
    return latest_eval_report_id or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a canary eval report and proposal link.")
    ap.add_argument("--telemetry_dir", default="telemetry")
    ap.add_argument("--proposal_id", required=True)
    ap.add_argument("--verdict", required=True, choices=["PASS", "FAIL", "INCONCLUSIVE"])
    ap.add_argument("--metrics_before", required=True, help='JSON object string, e.g. {"contract_errors_ratio":0.1}')
    ap.add_argument("--metrics_after", required=True, help='JSON object string, e.g. {"contract_errors_ratio":0.05}')
    ap.add_argument("--method", default="canary_eval")
    ap.add_argument("--timestamp_ms", type=int, default=0)
    ap.add_argument("--manifest_path", default="", help="Path to golden replay manifest v0 JSON.")
    ap.add_argument("--force", action="store_true", help="Allow duplicate canary eval for same proposal_id.")
    ap.add_argument("--rerun_reason", default="", help="Short reason when rerunning with --force.")
    args = ap.parse_args()

    timestamp_ms = int(args.timestamp_ms) if int(args.timestamp_ms) > 0 else int(datetime.now(timezone.utc).timestamp() * 1000)
    source_week = _iso_week_string_from_timestamp_ms(timestamp_ms)
    metrics_before = _parse_json_arg(args.metrics_before)
    metrics_after = _parse_json_arg(args.metrics_after)
    delta = _build_delta(metrics_before, metrics_after)

    telemetry_dir = Path(args.telemetry_dir)
    manifest_path = _resolve_manifest_path(args.manifest_path)
    if manifest_path is None:
        print("[error] manifest path is required (--manifest_path or EQNET_GOLDEN_REPLAY_MANIFEST)", file=sys.stderr)
        raise SystemExit(3)
    verified, manifest_id, manifest_sha, verify_reason = _verify_manifest(telemetry_dir, manifest_path)
    if not verified:
        print(f"[error] manifest verification failed: {verify_reason}", file=sys.stderr)
        raise SystemExit(3)

    latest = _latest_canary_gate_decision(telemetry_dir, str(args.proposal_id))
    decision = str((latest or {}).get("decision") or "")
    actor = str((latest or {}).get("actor") or "")
    if decision != "ACCEPT_CANARY" or actor not in {"human", "auto"}:
        print(
            f"[error] proposal_id={args.proposal_id} is not approved for canary eval (latest decision={decision or 'none'})",
            file=sys.stderr,
        )
        raise SystemExit(2)
    previous_eval_report_id = _latest_canary_eval_report_id(telemetry_dir, str(args.proposal_id))
    if not args.force:
        existing_links = _canary_eval_link_count(telemetry_dir, str(args.proposal_id))
        if existing_links >= 1:
            print(
                f"[error] canary eval already exists for proposal_id={args.proposal_id}; use --force to override",
                file=sys.stderr,
            )
            raise SystemExit(4)

    eval_writer = EvalReportWriter(EvalReportWriterConfig(telemetry_dir=telemetry_dir))
    eval_path = eval_writer.append(
        timestamp_ms=timestamp_ms,
        proposal_id=args.proposal_id,
        method=args.method,
        verdict=args.verdict,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        source_week=source_week,
        delta=delta,
        extra={
            "inputs_manifest_id": manifest_id,
            "inputs_manifest_sha256": manifest_sha,
            "rerun": bool(args.force),
            "rerun_reason": str(args.rerun_reason or "") if bool(args.force) else "",
            "previous_eval_report_id": previous_eval_report_id if bool(args.force) else None,
        },
    )
    eval_report_id = json.loads(eval_path.read_text(encoding="utf-8").strip().splitlines()[-1])["eval_report_id"]

    link_writer = ProposalLinkWriter(ProposalLinkWriterConfig(telemetry_dir=telemetry_dir))
    link_path = link_writer.append(
        timestamp_ms=timestamp_ms,
        proposal_id=args.proposal_id,
        eval_report_id=str(eval_report_id),
        link_type="canary_eval",
        source_week=source_week,
    )

    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=timestamp_ms,
        proposal_id=args.proposal_id,
        decision="LINK_EVAL_REPORT",
        actor="auto",
        reason="canary_eval_recorded",
        source_week=source_week,
        extra={
            "linked_eval_report_id": str(eval_report_id),
            "linked_eval_verdict": str(args.verdict),
            "linked_manifest_id": manifest_id,
        },
    )

    print(f"[info] eval report recorded: {eval_path}")
    print(f"[info] proposal link recorded: {link_path}")


if __name__ == "__main__":
    main()

