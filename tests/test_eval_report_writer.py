from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from eqnet.telemetry.change_decision_writer import ChangeDecisionWriter, ChangeDecisionWriterConfig
from eqnet.telemetry.eval_report_writer import EvalReportWriter, EvalReportWriterConfig
from eqnet.telemetry.proposal_links_writer import ProposalLinkWriter, ProposalLinkWriterConfig


def test_eval_report_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = EvalReportWriter(EvalReportWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-1",
        method="replay_eval",
        verdict="PASS",
        metrics_before={"contract_errors_ratio": 0.10},
        metrics_after={"contract_errors_ratio": 0.05},
        source_week="2025-W01",
        eval_report_id="00000000-0000-0000-0000-000000000011",
        delta={"contract_errors_ratio": -0.05},
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "schema_version",
        "eval_report_id",
        "timestamp_ms",
        "proposal_id",
        "method",
        "verdict",
        "metrics_before",
        "metrics_after",
        "source_week",
    ]:
        assert key in payload
    assert payload["schema_version"] == "eval_report.v0"
    assert payload["verdict"] == "PASS"


def test_proposal_link_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = ProposalLinkWriter(ProposalLinkWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-1",
        eval_report_id="e-1",
        link_type="shadow_eval",
        source_week="2025-W01",
        link_id="00000000-0000-0000-0000-000000000012",
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["schema_version"] == "proposal_link.v0"
    assert payload["proposal_id"] == "p-1"
    assert payload["eval_report_id"] == "e-1"
    assert payload["link_type"] == "shadow_eval"


def test_eval_report_writer_rejects_invalid_verdict(tmp_path: Path) -> None:
    writer = EvalReportWriter(EvalReportWriterConfig(telemetry_dir=tmp_path))
    with pytest.raises(ValueError):
        writer.append(
            timestamp_ms=1735689600000,
            proposal_id="p-1",
            method="replay_eval",
            verdict="MAYBE",
            metrics_before={},
            metrics_after={},
            source_week="2025-W01",
        )


def test_record_shadow_eval_cli_writes_report_and_link(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260215.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-shadow-test",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260215.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-100",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
        decision_id="00000000-0000-0000-0000-000000000099",
    )
    cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-100",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    eval_files = sorted(telemetry_dir.glob("eval_reports-*.jsonl"))
    link_files = sorted(telemetry_dir.glob("proposal_links-*.jsonl"))
    decision_files = sorted(telemetry_dir.glob("change_decisions-*.jsonl"))
    assert len(eval_files) == 1
    assert len(link_files) == 1
    assert len(decision_files) == 1
    eval_payload = json.loads(eval_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    link_payload = json.loads(link_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert eval_payload["proposal_id"] == "p-100"
    assert eval_payload["inputs_manifest_id"] == "golden-mecpe-shadow-test"
    assert eval_payload["rerun"] is False
    assert eval_payload.get("previous_eval_report_id") is None
    assert link_payload["proposal_id"] == "p-100"
    assert link_payload["eval_report_id"] == eval_payload["eval_report_id"]
    decision_rows = [
        json.loads(line)
        for line in decision_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    linked = [row for row in decision_rows if row.get("decision") == "LINK_EVAL_REPORT"]
    assert linked
    assert linked[-1]["proposal_id"] == "p-100"
    assert linked[-1]["actor"] == "auto"
    assert linked[-1]["linked_eval_report_id"] == eval_payload["eval_report_id"]


def test_record_shadow_eval_cli_fails_without_accept_shadow_decision(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260216.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-shadow-test2",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260216.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-200",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0
    assert not list(telemetry_dir.glob("eval_reports-*.jsonl"))


def test_record_shadow_eval_cli_fails_on_manifest_sha_mismatch(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260217.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-shadow-test3",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260217.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # tamper after manifest creation
    replay_file.write_text('{"x":2}\n', encoding="utf-8")
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-300",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-300",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0
    assert not list(telemetry_dir.glob("eval_reports-*.jsonl"))
    assert not list(telemetry_dir.glob("proposal_links-*.jsonl"))


def test_record_shadow_eval_cli_fails_on_empty_manifest_inputs(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest_empty.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-empty",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-400",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-400",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0
    assert not list(telemetry_dir.glob("eval_reports-*.jsonl"))


def test_record_shadow_eval_cli_rejects_duplicate_without_force(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260218.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-shadow-test4",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260218.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-500",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-500",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    second = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert second.returncode == 4
    eval_files = sorted(telemetry_dir.glob("eval_reports-*.jsonl"))
    assert len(eval_files) == 1
    assert len([l for l in eval_files[0].read_text(encoding="utf-8").splitlines() if l.strip()]) == 1


def test_record_shadow_eval_cli_allows_duplicate_with_force(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260219.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-shadow-test5",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260219.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-600",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    base_cmd = [
        sys.executable,
        "scripts/record_shadow_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-600",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    subprocess.run(base_cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    subprocess.run(
        base_cmd + ["--force", "--rerun_reason", "regression_check"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    eval_files = sorted(telemetry_dir.glob("eval_reports-*.jsonl"))
    assert len(eval_files) == 1
    rows = [
        json.loads(line)
        for line in eval_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[-1].get("rerun") is True
    assert rows[-1].get("rerun_reason") == "regression_check"
    assert rows[-1].get("previous_eval_report_id") == rows[0].get("eval_report_id")


def test_record_canary_eval_cli_writes_report_and_link(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260220.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-canary-test",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260220.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-700",
        decision="ACCEPT_CANARY",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    cmd = [
        sys.executable,
        "scripts/record_canary_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-700",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    eval_files = sorted(telemetry_dir.glob("eval_reports-*.jsonl"))
    link_files = sorted(telemetry_dir.glob("proposal_links-*.jsonl"))
    assert len(eval_files) == 1
    assert len(link_files) == 1
    eval_payload = json.loads(eval_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    link_payload = json.loads(link_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert eval_payload["method"] == "canary_eval"
    assert link_payload["link_type"] == "canary_eval"


def test_record_canary_eval_cli_rejects_without_accept_canary(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    replay_file = telemetry_dir / "mecpe-20260221.jsonl"
    replay_file.write_text('{"x":1}\n', encoding="utf-8")
    replay_sha = hashlib.sha256(replay_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "golden_replay_manifest.v0",
                "manifest_id": "golden-mecpe-canary-test2",
                "created_at_utc": "2026-02-22T00:00:00Z",
                "policy": {"pii_allowed": False, "source": "telemetry"},
                "inputs": [
                    {
                        "kind": "telemetry_jsonl",
                        "name": "mecpe_records",
                        "paths": ["mecpe-20260221.jsonl"],
                        "sha256": [replay_sha],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # only ACCEPT_SHADOW exists
    decision_writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=telemetry_dir))
    decision_writer.append(
        timestamp_ms=1735689600000,
        proposal_id="p-800",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="approved",
        source_week="2025-W01",
    )
    cmd = [
        sys.executable,
        "scripts/record_canary_eval.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "p-800",
        "--verdict",
        "PASS",
        "--metrics_before",
        "{\"contract_errors_ratio\":0.1}",
        "--metrics_after",
        "{\"contract_errors_ratio\":0.05}",
        "--timestamp_ms",
        "1735689600000",
        "--manifest_path",
        str(manifest_path),
    ]
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0
