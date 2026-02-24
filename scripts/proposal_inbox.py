#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Proposal Inbox CLI for approval-gated nightly proposals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from eqnet.runtime.companion_policy import load_lifelong_companion_policy
from eqnet.runtime.future_contracts import evaluate_companion_constraints
from eqnet.telemetry.change_decision_writer import ChangeDecisionWriter, ChangeDecisionWriterConfig
from eqnet.telemetry.sync_cue_execution_writer import (
    SyncCueExecutionWriter,
    SyncCueExecutionWriterConfig,
)


@dataclass(frozen=True)
class InboxItem:
    proposal_id: str
    kind: str
    origin_channel: str
    ttl_sec: int
    expires_at_ts_ms: int
    priority_score: float
    priority_reason_codes: List[str]
    companion_reason_codes: List[str]
    reason_codes: List[str]
    policy_meta: Dict[str, Any]
    payload: Dict[str, Any]
    status: str
    suppression_reason_codes: List[str]


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_nightly_path(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.glob("nightly_audit_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no nightly report found under {reports_dir}")
    return candidates[-1]


def _priority_of_proposal(proposal: Mapping[str, Any]) -> float:
    value = proposal.get("priority_score")
    if isinstance(value, (int, float)):
        return float(value)
    baseline = proposal.get("baseline_snapshot")
    if isinstance(baseline, Mapping):
        score = baseline.get("forecast_lite_score")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


def _ordered_reason_codes(codes: Iterable[str]) -> List[str]:
    unique = sorted(set(str(code) for code in codes if str(code).strip()))
    blocked = [code for code in unique if code.startswith("BLOCKED_") or code.startswith("COMPANION_POLICY_")]
    others = [code for code in unique if code not in blocked]
    return blocked + others


def _companion_reason_codes(codes: Iterable[str]) -> List[str]:
    return [code for code in _ordered_reason_codes(codes) if code.startswith("BLOCKED_") or code.startswith("COMPANION_POLICY_")]


def collect_inbox_items(payload: Mapping[str, Any], *, companion_policy: Mapping[str, Any] | None = None) -> List[InboxItem]:
    items: List[InboxItem] = []

    def _append_from(records: Any, *, fallback_kind: str) -> None:
        if not isinstance(records, list):
            return
        for idx, proposal in enumerate(records):
            if not isinstance(proposal, dict):
                continue
            if proposal.get("requires_approval") is not True:
                continue
            kind = str(proposal.get("kind") or fallback_kind)
            proposal_id = str(proposal.get("proposal_id") or f"{fallback_kind.lower()}-{idx}")
            reasons = list(proposal.get("reason_codes") or [])
            priority_reasons = list(proposal.get("priority_reason_codes") or [])
            reasons.extend(priority_reasons)
            if companion_policy:
                reasons.extend(evaluate_companion_constraints(proposal, companion_policy=companion_policy))
            ordered = _ordered_reason_codes(reasons)
            companion_reasons = _companion_reason_codes(ordered)
            policy_meta = proposal.get("policy_meta") if isinstance(proposal.get("policy_meta"), dict) else {}
            origin = str(proposal.get("origin_channel") or "unknown")
            ttl_sec = int(proposal.get("ttl_sec") or 0)
            ts_raw = proposal.get("ts_utc")
            ts_ms = 0
            if isinstance(ts_raw, str) and ts_raw:
                txt = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
                try:
                    ts_ms = int(datetime.fromisoformat(txt).timestamp() * 1000)
                except Exception:
                    ts_ms = 0
            expires = int(ts_ms + (ttl_sec * 1000)) if ts_ms > 0 and ttl_sec > 0 else 0
            items.append(
                InboxItem(
                    proposal_id=proposal_id,
                    kind=kind,
                    origin_channel=origin,
                    ttl_sec=ttl_sec,
                    expires_at_ts_ms=expires,
                    priority_score=_priority_of_proposal(proposal),
                    priority_reason_codes=_ordered_reason_codes(priority_reasons),
                    companion_reason_codes=companion_reasons,
                    reason_codes=ordered,
                    policy_meta=dict(policy_meta),
                    payload=dict(proposal),
                    status="PENDING",
                    suppression_reason_codes=[],
                )
            )

    _append_from(payload.get("realtime_forecast_proposals"), fallback_kind="REALTIME_FORECAST_PROPOSAL")
    _append_from(payload.get("preventive_proposals"), fallback_kind="PREVENTIVE_PROPOSAL")
    _append_from(payload.get("sync_cue_proposals"), fallback_kind="SYNC_CUE_PROPOSAL")
    items.sort(key=lambda item: (-float(item.priority_score), item.kind, item.proposal_id))
    return items


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _active_sync_suppression(telemetry_dir: Path, *, now_ts_ms: int) -> Dict[str, Any]:
    latest: Dict[str, Any] = {}
    for path in sorted(telemetry_dir.glob("sync_downshifts-*.jsonl")):
        for row in _read_jsonl(path):
            ts = row.get("timestamp_ms")
            until = row.get("cooldown_until_ts_ms")
            if not isinstance(ts, (int, float)) or not isinstance(until, (int, float)):
                continue
            if int(ts) > int(now_ts_ms):
                continue
            if not latest or int(ts) > int(latest.get("timestamp_ms") or 0):
                latest = dict(row)
    if not latest:
        return {"active": False, "reason_codes": []}
    until_val = int(latest.get("cooldown_until_ts_ms") or 0)
    active = int(now_ts_ms) < until_val
    reasons = [str(x) for x in (latest.get("reason_codes") or [])] if isinstance(latest.get("reason_codes"), list) else []
    return {"active": active, "reason_codes": _ordered_reason_codes(reasons)}


def apply_sync_suppression(items: List[InboxItem], suppression: Mapping[str, Any]) -> List[InboxItem]:
    if not bool(suppression.get("active")):
        return items
    reasons = [str(x) for x in (suppression.get("reason_codes") or [])] if isinstance(suppression.get("reason_codes"), list) else []
    patched: List[InboxItem] = []
    for item in items:
        if item.kind == "SYNC_CUE_PROPOSAL":
            patched.append(
                InboxItem(
                    proposal_id=item.proposal_id,
                    kind=item.kind,
                    origin_channel=item.origin_channel,
                    ttl_sec=item.ttl_sec,
                    expires_at_ts_ms=item.expires_at_ts_ms,
                    priority_score=item.priority_score,
                    priority_reason_codes=item.priority_reason_codes,
                    companion_reason_codes=item.companion_reason_codes,
                    reason_codes=item.reason_codes,
                    policy_meta=item.policy_meta,
                    payload=item.payload,
                    status="SUPPRESSED",
                    suppression_reason_codes=reasons,
                )
            )
        else:
            patched.append(item)
    return patched


def _iso_week(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _short_fp(meta: Mapping[str, Any]) -> str:
    fp = str(meta.get("policy_fingerprint") or "")
    if not fp:
        return ""
    return fp[:8]


def render_inbox_markdown(items: List[InboxItem]) -> str:
    lines = ["# Proposal Inbox", ""]
    if not items:
        lines.append("- pending: 0")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"- pending: {len(items)}")
    lines.append("")
    for item in items:
        fp = _short_fp(item.policy_meta)
        lines.append(f"- {item.proposal_id} [{item.kind}] priority={item.priority_score:.3f} fp={fp} status={item.status}")
        if item.reason_codes:
            lines.append(f"  reason_codes: {','.join(item.reason_codes)}")
        if item.suppression_reason_codes:
            lines.append(f"  suppression_reason_codes: {','.join(item.suppression_reason_codes)}")
    lines.append("")
    return "\n".join(lines)


def append_decision(
    *,
    telemetry_dir: Path,
    proposal_id: str,
    action: str,
    reason_codes: List[str],
    actor: str = "human",
    timestamp_ms: int | None = None,
) -> Path:
    ts_ms = int(timestamp_ms if timestamp_ms is not None else datetime.now(timezone.utc).timestamp() * 1000)
    norm_action = str(action).strip().lower()
    if norm_action == "approve":
        decision = "ACCEPT_SHADOW"
    elif norm_action == "decline":
        decision = "REJECT"
    else:
        raise ValueError("action must be approve or decline")
    normalized_codes = _ordered_reason_codes(reason_codes)
    reason = normalized_codes[0] if normalized_codes else ("APPROVED" if decision != "REJECT" else "DECLINED")
    writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    return writer.append(
        timestamp_ms=ts_ms,
        proposal_id=proposal_id,
        decision=decision,
        actor=actor,
        reason=reason,
        source_week=_iso_week(ts_ms),
        extra={
            "reason_codes": normalized_codes,
            "inbox_action": norm_action,
            "schema_hint": "proposal_inbox_v0",
        },
    )


def append_sync_execution(
    *,
    telemetry_dir: Path,
    proposal: Mapping[str, Any],
    timestamp_ms: int | None = None,
) -> Path:
    ts_ms = int(timestamp_ms if timestamp_ms is not None else datetime.now(timezone.utc).timestamp() * 1000)
    proposal_id = str(proposal.get("proposal_id") or "").strip()
    cue_type = str(
        proposal.get("sync_cue")
        or proposal.get("cue_type")
        or "SYNC_CUE_UNSPECIFIED"
    ).strip()
    ttl_sec = int(proposal.get("ttl_sec") or 0)
    writer = SyncCueExecutionWriter(SyncCueExecutionWriterConfig(telemetry_dir=telemetry_dir))
    return writer.append(
        timestamp_ms=ts_ms,
        proposal_id=proposal_id,
        cue_type=cue_type,
        ttl_sec=ttl_sec,
        source_week=_iso_week(ts_ms),
        extra={
            "origin_channel": str(proposal.get("origin_channel") or ""),
            "sync_order_parameter_r": proposal.get("sync_order_parameter_r"),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Approval inbox for nightly proposals.")
    ap.add_argument("--nightly-report", default=None, type=str, help="Path to nightly audit JSON.")
    ap.add_argument("--reports-dir", default="reports", type=str, help="Directory to search latest nightly report.")
    ap.add_argument("--telemetry-dir", default="telemetry", type=str, help="Telemetry directory for decisions.")
    ap.add_argument("--companion-policy", default="configs/lifelong_companion_policy_v0.yaml", type=str, help="Companion policy path.")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--out", default=None, type=str, help="Optional output file for list command.")
    ap.add_argument("--action", choices=["list", "approve", "decline"], default="list")
    ap.add_argument("--proposal-id", default=None, type=str, help="Target proposal ID for approve/decline.")
    ap.add_argument("--reason-code", action="append", default=[], help="Reason code (repeatable).")
    ap.add_argument("--disable-sync-execution", action="store_true", help="Do not emit sync execution ticket on approve.")
    args = ap.parse_args()

    report_path = Path(args.nightly_report) if args.nightly_report else _latest_nightly_path(Path(args.reports_dir))
    payload = _read_json(report_path)
    companion_policy = load_lifelong_companion_policy(Path(args.companion_policy))
    items = collect_inbox_items(payload, companion_policy=companion_policy)
    now_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    suppression = _active_sync_suppression(Path(args.telemetry_dir), now_ts_ms=now_ts_ms)
    items = apply_sync_suppression(items, suppression)

    if args.action == "list":
        if args.format == "json":
            out_payload = {
                "schema_version": "proposal_inbox_result_v0",
                "nightly_report": str(report_path.as_posix()),
                "pending_count": len(items),
                "generated_at_eval_ts_ms": now_ts_ms,
                "day_key_range": {"from": str(payload.get("day") or ""), "to": str(payload.get("day") or "")},
                "items": [
                    {
                        "proposal_id": item.proposal_id,
                        "kind": item.kind,
                        "origin_channel": item.origin_channel,
                        "ttl_sec": item.ttl_sec,
                        "expires_at_ts_ms": item.expires_at_ts_ms,
                        "priority_score": item.priority_score,
                        "priority_reason_codes": item.priority_reason_codes,
                        "companion_reason_codes": item.companion_reason_codes,
                        "suppression_reason_codes": item.suppression_reason_codes,
                        "reason_codes": item.reason_codes,
                        "requires_approval": True,
                        "status": item.status,
                        "decision_reason_code": "",
                        "policy_meta": item.policy_meta,
                    }
                    for item in items
                ],
            }
            text = json.dumps(out_payload, ensure_ascii=False, indent=2)
        else:
            text = render_inbox_markdown(items)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(text)
        raise SystemExit(0)

    if not args.proposal_id:
        raise SystemExit("--proposal-id is required for approve/decline")
    matched = [item for item in items if item.proposal_id == args.proposal_id]
    if not matched:
        raise SystemExit(f"proposal not found in inbox: {args.proposal_id}")
    decision_path = append_decision(
        telemetry_dir=Path(args.telemetry_dir),
        proposal_id=args.proposal_id,
        action=args.action,
        reason_codes=list(args.reason_code or []),
    )
    execution_path = None
    if (
        args.action == "approve"
        and not bool(args.disable_sync_execution)
        and matched
        and str(matched[0].kind) == "SYNC_CUE_PROPOSAL"
    ):
        execution_path = append_sync_execution(
            telemetry_dir=Path(args.telemetry_dir),
            proposal=matched[0].payload,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "decision_path": str(decision_path.as_posix()),
                "execution_path": str(execution_path.as_posix()) if isinstance(execution_path, Path) else "",
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
