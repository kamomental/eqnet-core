from __future__ import annotations

import json
import logging
import os
import hashlib
import time
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from eqnet.hub.runtime_contract import HubRuntime
from eqnet.qualia_model import update_qualia_state
from eqnet.runtime.life_indicator import LifeIndicator
from eqnet.runtime.nightly.metabolism_tool import (
    DEFAULT_METABOLISM_POLICY,
    load_metabolism_policy,
    run_metabolism_cycle,
)
from eqnet.runtime.nightly.repair_tool import run_repair_cycle
from eqnet.runtime.policy import PolicyPrior
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit
from emot_terrain_lab.hub.qualia_logging import append_qualia_telemetry
from emot_terrain_lab.ops.nightly_life_indicator import (
    compute_life_indicator_for_day,
    load_qualia_log,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MEMORY_THERMO_POLICY: Dict[str, Any] = dict(DEFAULT_METABOLISM_POLICY)
TX_PHASE_INIT = "INIT"
TX_PHASE_STAGED = "STAGED"
TX_PHASE_COMMITTING = "COMMITTING"
TX_PHASE_COMMITTED = "COMMITTED"
TX_PHASE_ROLLING_BACK = "ROLLING_BACK"
TX_PHASE_ROLLED_BACK = "ROLLED_BACK"
TX_PHASE_FAILED = "FAILED"
ALLOWED_TX_TRANSITIONS = {
    TX_PHASE_INIT: {TX_PHASE_STAGED, TX_PHASE_FAILED},
    TX_PHASE_STAGED: {TX_PHASE_COMMITTING, TX_PHASE_FAILED},
    TX_PHASE_COMMITTING: {TX_PHASE_COMMITTED, TX_PHASE_ROLLING_BACK, TX_PHASE_FAILED},
    TX_PHASE_ROLLING_BACK: {TX_PHASE_ROLLED_BACK, TX_PHASE_FAILED},
    TX_PHASE_ROLLED_BACK: {TX_PHASE_FAILED},
    TX_PHASE_FAILED: set(),
    TX_PHASE_COMMITTED: set(),
}


class InternalRuntimeSurface(Protocol):
    """Minimal legacy surface for wiring ExternalRuntimeDelegate."""

    config: Any
    embed_text_fn: Callable[[str], Any]
    _trace_safety: Any

    def _set_latest_state(self, kind: str, value: Any) -> None:
        ...

    def _get_latest_state(self) -> Dict[str, Any]:
        ...


class ExternalRuntimeDelegate(HubRuntime):
    """Backward-compatible external runtime bridge.

    The implementation delegates to ExternalRuntimeDelegateV2 so it does not
    depend on hub local execution methods.
    """

    def __init__(self, surface: InternalRuntimeSurface, *, runtime_version: str) -> None:
        self.runtime_version = runtime_version
        self._delegate_v2 = ExternalRuntimeDelegateV2(
            config=surface.config,
            embed_text_fn=surface.embed_text_fn,
            latest_state_writer=surface._set_latest_state,
            latest_state_reader=surface._get_latest_state,
            runtime_version=runtime_version,
            trace_dir_env="EQNET_TRACE_V1_DIR",
            boundary_threshold=getattr(surface._trace_safety, "boundary_threshold", 0.5),
        )

    def log_moment(
        self,
        raw_event: Any,
        raw_text: str,
        *,
        idempotency_key: Optional[str] = None,  # noqa: ARG002
    ) -> None:
        self._delegate_v2.log_moment(
            raw_event,
            raw_text,
            idempotency_key=idempotency_key,
        )

    def run_nightly(
        self,
        date_obj: Optional[date] = None,
        *,
        idempotency_key: Optional[str] = None,  # noqa: ARG002
    ) -> None:
        self._delegate_v2.run_nightly(
            date_obj,
            idempotency_key=idempotency_key,
        )

    def query_state(
        self,
        *,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._delegate_v2.query_state(as_of=as_of)


class ExternalRuntimeDelegateV2(HubRuntime):
    """Surface-independent external runtime implementation."""

    def __init__(
        self,
        *,
        config: Any,
        embed_text_fn: Callable[[str], Any],
        latest_state_writer: Callable[[str, Any], None],
        latest_state_reader: Callable[[], Dict[str, Any]],
        runtime_version: str,
        trace_dir_env: str,
        boundary_threshold: float,
    ) -> None:
        self._config = config
        self._embed_text_fn = embed_text_fn
        self._write_latest = latest_state_writer
        self._read_latest = latest_state_reader
        self.runtime_version = runtime_version
        self._trace_dir_env = trace_dir_env
        self._boundary_threshold = float(boundary_threshold)

    def log_moment(
        self,
        raw_event: Any,
        raw_text: str,
        *,
        idempotency_key: Optional[str] = None,  # noqa: ARG002
    ) -> None:
        moment_entry = raw_event
        text_emb = self._embed_text_fn(raw_text)
        qstate = update_qualia_state(
            prev_state=None,
            moment_entry=moment_entry,
            text_embedding=text_emb,
        )
        self._write_latest("qualia", qstate)
        append_qualia_telemetry(self._config.telemetry_dir, qstate)

    def run_nightly(
        self,
        date_obj: Optional[date] = None,
        *,
        idempotency_key: Optional[str] = None,  # noqa: ARG002
    ) -> None:
        day = date_obj or date.today()
        day_key_compact = day.strftime("%Y%m%d")
        tx_id = self._nightly_transaction_id(day=day, idempotency_key=idempotency_key)
        if _env_truthy("EQNET_NIGHTLY_TX_IDEMPOTENT") and idempotency_key and self._is_tx_committed(tx_id):
            return
        self._advance_tx_phase(tx_id, TX_PHASE_INIT)
        qualia_path = self._config.telemetry_dir / f"qualia-{day_key_compact}.jsonl"
        qualia_records = load_qualia_log(qualia_path)
        thermo_policy = _memory_thermo_policy(self._config)
        previous_thermo = self._read_latest().get("memory_thermo")
        defrag_result = run_metabolism_cycle(
            qualia_records=qualia_records,
            policy=thermo_policy,
            previous_state=previous_thermo if isinstance(previous_thermo, Mapping) else None,
        )
        repair_report = _run_forgetting_reweight(self._config)
        if repair_report is not None:
            defrag_result["repair"] = repair_report
            forgetting = repair_report.get("forgetting")
            if isinstance(forgetting, Mapping):
                defrag_result["forgetting"] = dict(forgetting)
            defrag_result["repair_status"] = str(repair_report.get("repair_status") or "")
            defrag_result["repair_tool_version"] = str(repair_report.get("repair_tool_version") or "")
            defrag_result["repaired_events_count"] = int(repair_report.get("repaired_events_count") or 0)
            defrag_result["repair_plan_id"] = str(repair_report.get("repair_plan_id") or "")
            defrag_result["repair_replay_token"] = str(repair_report.get("replay_token") or "")
            defrag_result["repair_ops"] = list(repair_report.get("repair_ops") or [])
            defrag_result["repair_ops_digest"] = str(repair_report.get("repair_ops_digest") or "")
        defrag_result["nightly_transaction_id"] = tx_id
        defrag_result["nightly_transaction_phase"] = TX_PHASE_INIT
        defrag_result["nightly_transaction_atomic"] = False
        life_indicator = compute_life_indicator_for_day(
            qualia_records,
            num_diary_entries=0,
            num_self_reflection_entries=0,
        )
        policy_prior = PolicyPrior()
        try:
            staged = self._stage_nightly_artifacts(
                day=day,
                life_indicator=life_indicator,
                policy_prior=policy_prior,
                memory_thermo=defrag_result,
            )
            self._advance_tx_phase(tx_id, TX_PHASE_STAGED)
            self._advance_tx_phase(tx_id, TX_PHASE_COMMITTING)
            self._commit_nightly_artifacts(staged, tx_id=tx_id)
            self._advance_tx_phase(tx_id, TX_PHASE_COMMITTED)
            defrag_result["nightly_transaction_phase"] = TX_PHASE_COMMITTED
            defrag_result["nightly_transaction_atomic"] = True
            self._write_latest("memory_thermo", defrag_result)
            self._write_latest("life_indicator", life_indicator)
            self._write_latest("policy_prior", policy_prior)
        except Exception:
            self._advance_tx_phase(tx_id, TX_PHASE_FAILED)
            raise
        self._run_nightly_audit(day)

    def query_state(
        self,
        *,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        latest = self._read_latest()
        latest_q = latest.get("qualia")
        latest_li = latest.get("life_indicator")
        latest_pp = latest.get("policy_prior")
        state: Dict[str, Any] = {
            "latest_qualia": None,
            "life_indicator": None,
            "policy_prior": None,
            "danger": {},
            "healing": {},
        }
        if latest_q is not None:
            vec = latest_q.qualia_vec
            dim = int(vec.shape[0]) if hasattr(vec, "shape") else len(vec)
            state["latest_qualia"] = {
                "timestamp": latest_q.timestamp.isoformat(),
                "dimension": dim,
                "qualia_vec": vec.tolist(),
            }
        if latest_li is not None:
            state["life_indicator"] = {
                "identity": latest_li.identity_score,
                "qualia": latest_li.qualia_score,
                "meta_awareness": latest_li.meta_awareness_score,
            }
        if latest_pp is not None:
            state["policy_prior"] = latest_pp.__dict__
        resolved_day_key = _normalize_day_key(as_of)
        if resolved_day_key is None:
            if latest_q is not None and getattr(latest_q, "timestamp", None) is not None:
                resolved_day_key = latest_q.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")
            else:
                resolved_day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state["resolved_day_key"] = resolved_day_key
        return state

    def _save_life_indicator(self, day: date, li: LifeIndicator) -> None:
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._config.state_dir / f"life-indicator-{day.strftime('%Y%m%d')}.json"
        payload = {
            "identity": li.identity_score,
            "qualia": li.qualia_score,
            "meta_awareness": li.meta_awareness_score,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_policy_prior(self, pp: PolicyPrior) -> None:
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._config.state_dir / "policy-prior-latest.json"
        path.write_text(json.dumps(pp.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_nightly_report(self, day: date, li: LifeIndicator, pp: PolicyPrior) -> None:
        self._config.reports_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "life_indicator": {
                "identity": li.identity_score,
                "qualia": li.qualia_score,
                "meta_awareness": li.meta_awareness_score,
            },
            "policy_prior": pp.__dict__,
        }
        path = self._config.reports_dir / f"nightly-{day.strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_nightly_audit(self, day: date) -> None:
        day_key = day.strftime("%Y-%m-%d")
        trace_root = Path(
            os.getenv(self._trace_dir_env)
            or getattr(self._config, "trace_dir", self._config.telemetry_dir / "trace_v1")
        )
        out_dir = Path(getattr(self._config, "audit_dir", self._config.telemetry_dir / "audit"))
        day_dir = trace_root / day_key
        if not day_dir.exists():
            return
        if not any(day_dir.glob("*.jsonl")):
            return
        cfg = NightlyAuditConfig(
            trace_root=trace_root,
            out_root=out_dir,
            date_yyyy_mm_dd=day_key,
            boundary_threshold=self._boundary_threshold,
            health_thresholds=getattr(self._config, "audit_thresholds", None),
        )
        try:
            generate_audit(cfg)
        except Exception:
            LOGGER.warning("nightly audit failed for %s", day_key, exc_info=True)

    def _nightly_transaction_id(self, *, day: date, idempotency_key: Optional[str]) -> str:
        payload = {
            "day": day.isoformat(),
            "idempotency_key": str(idempotency_key or ""),
            "runtime_version": self.runtime_version,
        }
        if not idempotency_key:
            payload["now_ns"] = time.time_ns()
            payload["pid"] = os.getpid()
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _stage_nightly_artifacts(
        self,
        *,
        day: date,
        life_indicator: LifeIndicator,
        policy_prior: PolicyPrior,
        memory_thermo: Mapping[str, Any],
    ) -> dict[str, Any]:
        tx_id = str(memory_thermo.get("nightly_transaction_id") or self._nightly_transaction_id(day=day, idempotency_key=None))
        day_compact = day.strftime("%Y%m%d")
        staging_root = (
            Path(self._config.reports_dir)
            / ".nightly_staging"
            / f"{day_compact}-{tx_id}"
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        target_to_staged: dict[Path, Path] = {}

        life_payload = {
            "identity": life_indicator.identity_score,
            "qualia": life_indicator.qualia_score,
            "meta_awareness": life_indicator.meta_awareness_score,
        }
        policy_payload = dict(policy_prior.__dict__)
        memory_payload = dict(memory_thermo)
        report_payload = {
            "life_indicator": dict(life_payload),
            "policy_prior": dict(policy_payload),
            "nightly_transaction": {
                "id": tx_id,
                "phase": str(memory_thermo.get("nightly_transaction_phase") or TX_PHASE_INIT),
                "atomic": bool(memory_thermo.get("nightly_transaction_atomic", False)),
            },
        }

        outputs: list[tuple[Path, dict[str, Any], str]] = [
            (
                Path(self._config.state_dir) / f"life-indicator-{day_compact}.json",
                life_payload,
                "life_indicator",
            ),
            (
                Path(self._config.state_dir) / "policy-prior-latest.json",
                policy_payload,
                "policy_prior",
            ),
            (
                Path(self._config.state_dir) / "memory-thermo-latest.json",
                memory_payload,
                "memory_thermo",
            ),
            (
                Path(self._config.reports_dir) / f"nightly-{day_compact}.json",
                report_payload,
                "nightly_report",
            ),
        ]
        for target, payload, label in outputs:
            staged_file = staging_root / f"{label}.json"
            staged_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            target_to_staged[target] = staged_file
        return {
            "tx_id": tx_id,
            "staging_root": staging_root,
            "target_to_staged": target_to_staged,
        }

    def _commit_nightly_artifacts(self, staged: Mapping[str, Any], *, tx_id: str | None = None) -> None:
        staging_root = Path(staged.get("staging_root"))
        target_to_staged = staged.get("target_to_staged")
        if not isinstance(target_to_staged, Mapping):
            raise ValueError("target_to_staged is required")
        backup_root = staging_root / "backup"
        backup_root.mkdir(parents=True, exist_ok=True)
        backups: list[tuple[Path, Path]] = []
        committed_targets: list[Path] = []
        try:
            for target_raw, staged_raw in target_to_staged.items():
                target = Path(target_raw)
                staged_file = Path(staged_raw)
                if not staged_file.exists():
                    raise FileNotFoundError(f"staged file missing: {staged_file}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    backup = backup_root / f"{len(backups):03d}-{target.name}.bak"
                    os.replace(target, backup)
                    backups.append((target, backup))
            for target_raw, staged_raw in target_to_staged.items():
                target = Path(target_raw)
                staged_file = Path(staged_raw)
                os.replace(staged_file, target)
                committed_targets.append(target)
        except Exception:
            if tx_id:
                self._advance_tx_phase(tx_id, TX_PHASE_ROLLING_BACK)
            for target in committed_targets:
                try:
                    if target.exists():
                        target.unlink()
                except Exception:
                    pass
            for target, backup in backups:
                try:
                    if backup.exists():
                        os.replace(backup, target)
                except Exception:
                    pass
            if tx_id:
                self._advance_tx_phase(tx_id, TX_PHASE_ROLLED_BACK)
            raise
        finally:
            try:
                shutil.rmtree(staging_root, ignore_errors=True)
            except Exception:
                pass

    def _tx_journal_path(self, tx_id: str) -> Path:
        root = Path(self._config.state_dir) / "nightly_tx"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{tx_id}.json"

    def _tx_current_phase(self, tx_id: str) -> str | None:
        path = self._tx_journal_path(tx_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        phase = payload.get("phase")
        if isinstance(phase, str) and phase:
            return phase
        return None

    def _is_tx_committed(self, tx_id: str) -> bool:
        return self._tx_current_phase(tx_id) == TX_PHASE_COMMITTED

    def _advance_tx_phase(self, tx_id: str, next_phase: str) -> None:
        current = self._tx_current_phase(tx_id)
        if current is None:
            if next_phase != TX_PHASE_INIT:
                # late recovery path: allow writing the observed phase
                pass
        elif current == next_phase:
            return
        else:
            allowed = ALLOWED_TX_TRANSITIONS.get(current, set())
            if next_phase not in allowed:
                return
        payload = {
            "tx_id": tx_id,
            "phase": next_phase,
            "updated_at_ms": int(time.time() * 1000),
        }
        self._tx_journal_path(tx_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _normalize_day_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _memory_thermo_policy(config: Any) -> Dict[str, Any]:
    return load_metabolism_policy(config)


def _run_defrag_observe(
    qualia_records: list[dict[str, Any]],
    policy: Dict[str, Any],
    phase_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    before = _measure_defrag_metrics(qualia_records, policy)
    # Stage-1 observe: no structural mutation, only metering.
    after = dict(before)
    delta = _metrics_delta(before, after)
    memory_entropy_delta = _entropy_delta_from_metrics(
        delta,
        phase_ctx.get("delta_weights") or {},
    )
    entropy_cost_class = _entropy_cost_class(
        memory_entropy_delta,
        phase_ctx.get("entropy_cost_class_thresholds") or {},
    )
    entropy_budget_ok = True
    energy_budget_used = float(memory_entropy_delta)
    energy_budget_limit = float(policy.get("energy_budget_limit", 0.10))
    budget_throttle_applied = energy_budget_used >= energy_budget_limit
    profile_cfg = policy.get("output_control_profiles") if isinstance(policy.get("output_control_profiles"), dict) else {}
    reason_cfg = policy.get("throttle_reason_codes") if isinstance(policy.get("throttle_reason_codes"), dict) else {}
    output_control_profile = str(
        profile_cfg.get("throttled" if budget_throttle_applied else "normal")
        or ("cautious_budget_v1" if budget_throttle_applied else "normal_v1")
    )
    throttle_reason_code = str(reason_cfg.get("budget_exceeded") or "BUDGET_EXCEEDED") if budget_throttle_applied else ""
    projection_fp = _value_projection_fingerprint(
        policy_version=str(policy.get("policy_version") or "memory-ops-v1"),
        entropy_model_id=str(policy.get("entropy_model_id") or "defrag-observe-v1"),
        memory_phase=str(phase_ctx.get("memory_phase") or "stabilization"),
        phase_weight_profile=str(phase_ctx.get("phase_weight_profile") or "default"),
        delta_weights=phase_ctx.get("delta_weights") or {},
        thresholds=phase_ctx.get("entropy_cost_class_thresholds") or {},
    )
    return {
        "defrag_status": "observed",
        "defrag_mode": "observe",
        "defrag_metrics_before": before,
        "defrag_metrics_after": after,
        "defrag_metrics_delta": delta,
        "memory_entropy_delta": memory_entropy_delta,
        "entropy_cost_class": entropy_cost_class,
        "irreversible_op": False,
        "entropy_budget_ok": entropy_budget_ok,
        "memory_phase": str(phase_ctx.get("memory_phase") or "stabilization"),
        "phase_weight_profile": str(phase_ctx.get("phase_weight_profile") or "default"),
        "value_projection_fingerprint": projection_fp,
        "phase_override_applied": bool(phase_ctx.get("phase_override_applied", False)),
        "energy_budget_used": energy_budget_used,
        "energy_budget_limit": energy_budget_limit,
        "budget_throttle_applied": budget_throttle_applied,
        "output_control_profile": output_control_profile,
        "throttle_reason_code": throttle_reason_code,
        "policy_version": str(policy.get("policy_version") or "memory-ops-v1"),
        "entropy_model_id": str(policy.get("entropy_model_id") or "defrag-observe-v1"),
    }


def _measure_defrag_metrics(
    qualia_records: list[dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, float]:
    enabled = policy.get("enabled_metrics") or []
    metrics: Dict[str, float] = {}
    for metric_name in enabled:
        key = str(metric_name)
        if key == "memory_item_count":
            metrics[key] = float(len(qualia_records))
        elif key == "link_count":
            metrics[key] = float(
                sum(
                    1
                    for row in qualia_records
                    if isinstance(row, dict)
                    and any(
                        candidate in row
                        for candidate in ("parent_id", "link_id", "edge_id", "ref_id")
                    )
                )
            )
        elif key == "bytes_estimate":
            metrics[key] = float(
                sum(len(json.dumps(row, ensure_ascii=False, default=str)) for row in qualia_records)
            )
        elif key == "summary_count":
            metrics[key] = float(
                sum(
                    1
                    for row in qualia_records
                    if isinstance(row, dict)
                    and str(row.get("type") or row.get("kind") or "").lower() == "summary"
                )
            )
    return metrics


def _metrics_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    keys = set(before.keys()) | set(after.keys())
    return {k: float(after.get(k, 0.0) - before.get(k, 0.0)) for k in sorted(keys)}


def _entropy_delta_from_metrics(delta: Dict[str, float], weights: Dict[str, Any]) -> float:
    total = 0.0
    scale = 0.0
    for key, value in delta.items():
        weight = float(weights.get(key, 1.0))
        total += abs(float(value)) * weight
        scale += abs(weight)
    if scale <= 0.0:
        return 0.0
    return float(total / scale)


def _entropy_cost_class(memory_entropy_delta: float, thresholds: Dict[str, Any]) -> str:
    high = float(thresholds.get("high", 0.20))
    mid = float(thresholds.get("mid", 0.05))
    if memory_entropy_delta >= high:
        return "HIGH"
    if memory_entropy_delta >= mid:
        return "MID"
    return "LOW"


def _resolve_phase_context(policy: Dict[str, Any], previous_thermo: Any) -> Dict[str, Any]:
    profiles = policy.get("phase_profiles") if isinstance(policy.get("phase_profiles"), dict) else {}
    default_phase = str(policy.get("default_phase") or "stabilization")
    selected_phase = default_phase
    override_phase = policy.get("memory_phase_override")
    override_applied = False
    if isinstance(override_phase, str) and override_phase in profiles:
        selected_phase = override_phase
        override_applied = True
    elif isinstance(override_phase, str) and override_phase:
        selected_phase = default_phase
    if isinstance(previous_thermo, dict):
        prev_phase = previous_thermo.get("memory_phase")
        if (
            not (isinstance(override_phase, str) and override_phase in profiles)
            and isinstance(prev_phase, str)
            and prev_phase in profiles
        ):
            selected_phase = prev_phase
    profile = profiles.get(selected_phase) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        profile = {}
    weights = profile.get("delta_weights") if isinstance(profile.get("delta_weights"), dict) else policy.get("delta_weights") or {}
    thresholds = (
        profile.get("entropy_cost_class_thresholds")
        if isinstance(profile.get("entropy_cost_class_thresholds"), dict)
        else policy.get("entropy_cost_class_thresholds") or {}
    )
    return {
        "memory_phase": selected_phase,
        "phase_weight_profile": str(profile.get("phase_weight_profile") or f"phase.{selected_phase}.default"),
        "delta_weights": dict(weights),
        "entropy_cost_class_thresholds": dict(thresholds),
        "phase_override_applied": override_applied,
    }


def _value_projection_fingerprint(
    *,
    policy_version: str,
    entropy_model_id: str,
    memory_phase: str,
    phase_weight_profile: str,
    delta_weights: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> str:
    payload = {
        "policy_version": policy_version,
        "entropy_model_id": entropy_model_id,
        "memory_phase": memory_phase,
        "phase_weight_profile": phase_weight_profile,
        "delta_weights": delta_weights,
        "entropy_cost_class_thresholds": thresholds,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _forgetting_policy(config: Any) -> Dict[str, Any]:
    runtime_policy = getattr(config, "runtime_policy", None)
    if not isinstance(runtime_policy, dict):
        return {}
    forgetting = runtime_policy.get("forgetting")
    if not isinstance(forgetting, dict):
        return {}
    return dict(forgetting)


def _resolve_replay_memory_path(config: Any, forgetting_cfg: Mapping[str, Any]) -> Path:
    explicit = forgetting_cfg.get("replay_memory_path")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit)
    state_dir = Path(getattr(config, "state_dir", Path("state")))
    return state_dir / "replay_memory.jsonl"


def _load_replay_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _rewrite_replay_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _extract_seed_ids(event: Mapping[str, Any]) -> list[str]:
    seeds: list[str] = []
    meta = event.get("meta")
    if isinstance(meta, Mapping):
        replay = meta.get("replay")
        if isinstance(replay, Mapping):
            seeds_payload = replay.get("seeds")
            if isinstance(seeds_payload, list):
                for seed in seeds_payload:
                    if not isinstance(seed, Mapping):
                        continue
                    trace_id = seed.get("trace_id")
                    if trace_id:
                        seeds.append(str(trace_id))
    return seeds


def _collect_recall_counts(events: list[dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        for trace_id in _extract_seed_ids(event):
            counts[trace_id] = counts.get(trace_id, 0) + 1
    return counts


def _load_monument_episode_ids(memory_dir: Optional[str]) -> Tuple[set[str], Optional[str]]:
    if not memory_dir:
        return set(), "memory_dir_missing"
    try:
        from eqnet.memory.store import MemoryStore

        store = MemoryStore(Path(memory_dir))
        _, monuments = store.load_all()
        episode_ids = {str(ep_id) for mon in monuments for ep_id in (mon.episodes or []) if ep_id}
        return episode_ids, None
    except Exception as exc:  # pragma: no cover - defensive
        return set(), str(exc)


def _apply_forgetting_reweight(
    events: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    now_ts: float,
) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
    recall_k = _safe_float(cfg.get("recall_k"), 0.0)
    recall_weight = _safe_float(cfg.get("recall_weight"), 0.0)
    affect_weight = _safe_float(cfg.get("affect_weight"), 0.0)
    interference_weight = _safe_float(cfg.get("interference_weight"), 0.0)
    interference_k = _safe_float(cfg.get("interference_k"), 0.0)
    reconsolidation_rate = _safe_float(cfg.get("reconsolidation_rate"), 0.0)
    base_delta = _safe_float(cfg.get("base_delta"), 0.0)
    max_delta_w = _safe_float(cfg.get("max_delta_w"), 0.0)
    min_w = _safe_float(cfg.get("min_w"), 0.0)
    max_w = _safe_float(cfg.get("max_w"), 1.0)
    monument_w_lock = bool(cfg.get("monument_w_lock", True))
    monument_floor = cfg.get("monument_connection_floor")
    monument_floor = _safe_float(monument_floor, -1.0) if monument_floor is not None else None
    consent_floor = _safe_float(cfg.get("consent_floor"), min_w)
    consent_tags = {str(tag) for tag in (cfg.get("consent_override_tags") or [])}
    memory_dir_raw = cfg.get("memory_dir")
    memory_dir = str(memory_dir_raw) if isinstance(memory_dir_raw, str) and memory_dir_raw.strip() else None

    recall_counts = _collect_recall_counts(events)
    monument_episode_ids, monument_err = _load_monument_episode_ids(memory_dir)

    locked_applied = 0
    floors_applied = 0
    consent_applied = 0
    changed = 0

    for event in events:
        trace_id = str(event.get("trace_id", ""))
        episode_id = str(event.get("episode_id", ""))
        memory_kind = str(event.get("memory_kind", ""))
        w_before = _safe_float(event.get("weight"), 1.0)

        tags_raw = event.get("tags")
        tags: list[str]
        if isinstance(tags_raw, list):
            tags = [str(tag) for tag in tags_raw]
        elif tags_raw is None:
            tags = []
        else:
            tags = [str(tags_raw)]

        consent_override = any(tag in consent_tags for tag in tags)
        is_monument = memory_kind.lower() == "monument"
        lock_weight = monument_w_lock and is_monument

        if consent_override:
            w_after = min(w_before, consent_floor)
            consent_applied += 1
        elif lock_weight:
            w_after = w_before
            locked_applied += 1
        else:
            recall_count = recall_counts.get(trace_id, 0)
            recall_score = min(1.0, recall_k * float(recall_count)) if recall_k > 0 else 0.0

            emotion_mod = abs(_safe_float(event.get("emotion_modulation"), 0.0))
            affect_score = min(1.0, emotion_mod)

            interference_score = 0.0
            meta = event.get("meta")
            if isinstance(meta, Mapping):
                interference = meta.get("interference")
                if isinstance(interference, Mapping):
                    interference_score = _safe_float(interference.get("similarity"), 0.0)
            interference_score = min(1.0, interference_k * interference_score)

            delta = base_delta
            delta += recall_weight * recall_score
            delta -= affect_weight * affect_score * reconsolidation_rate
            delta -= interference_weight * interference_score
            if max_delta_w > 0:
                delta = _clamp(delta, -max_delta_w, max_delta_w)
            w_after = _clamp(w_before + delta, min_w, max_w)

        if monument_floor is not None and monument_floor >= 0 and episode_id and episode_id in monument_episode_ids:
            if w_after < monument_floor:
                w_after = monument_floor
                floors_applied += 1

        # Keep timestamps monotonic-safe if weight is missing/corrupt.
        _ = now_ts
        if abs(w_after - w_before) > 1e-12:
            changed += 1
        event["weight"] = float(w_after)

    report = {
        "status": "applied",
        "changed_count": int(changed),
        "event_count": int(len(events)),
        "monument_episode_count": int(len(monument_episode_ids)),
        "monument_load_error": monument_err,
        "guards": {
            "monument_locks": int(locked_applied),
            "monument_floors": int(floors_applied),
            "consent_overrides": int(consent_applied),
        },
    }
    return events, report


def _run_forgetting_reweight(config: Any) -> Optional[Dict[str, Any]]:
    return run_repair_cycle(config)


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
