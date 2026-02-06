from __future__ import annotations

import json
import logging
import os
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from eqnet.hub.runtime_contract import HubRuntime
from eqnet.qualia_model import update_qualia_state
from eqnet.runtime.life_indicator import LifeIndicator
from eqnet.runtime.policy import PolicyPrior
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit
from emot_terrain_lab.hub.qualia_logging import append_qualia_telemetry
from emot_terrain_lab.ops.nightly_life_indicator import (
    compute_life_indicator_for_day,
    load_qualia_log,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MEMORY_THERMO_POLICY: Dict[str, Any] = {
    "policy_version": "memory-ops-v1",
    "entropy_model_id": "defrag-observe-v1",
    "enabled_metrics": [
        "memory_item_count",
        "link_count",
        "bytes_estimate",
        "summary_count",
    ],
    "delta_weights": {
        "memory_item_count": 1.0,
        "link_count": 1.0,
        "bytes_estimate": 1.0,
        "summary_count": 1.0,
    },
    "entropy_cost_class_thresholds": {
        "mid": 0.05,
        "high": 0.20,
    },
    "default_phase": "stabilization",
    "phase_profiles": {
        "exploration": {
            "phase_weight_profile": "phase.exploration.v1",
            "delta_weights": {
                "memory_item_count": 1.3,
                "link_count": 1.1,
                "bytes_estimate": 0.8,
                "summary_count": 0.7,
            },
            "entropy_cost_class_thresholds": {"mid": 0.06, "high": 0.22},
        },
        "stabilization": {
            "phase_weight_profile": "phase.stabilization.v1",
            "delta_weights": {
                "memory_item_count": 1.0,
                "link_count": 1.0,
                "bytes_estimate": 1.0,
                "summary_count": 1.0,
            },
            "entropy_cost_class_thresholds": {"mid": 0.05, "high": 0.20},
        },
        "recovery": {
            "phase_weight_profile": "phase.recovery.v1",
            "delta_weights": {
                "memory_item_count": 0.8,
                "link_count": 1.4,
                "bytes_estimate": 1.1,
                "summary_count": 1.2,
            },
            "entropy_cost_class_thresholds": {"mid": 0.04, "high": 0.16},
        },
    },
    "energy_budget_limit": 0.10,
    "output_control_profiles": {
        "normal": "normal_v1",
        "throttled": "cautious_budget_v1",
    },
    "throttle_reason_codes": {
        "budget_exceeded": "BUDGET_EXCEEDED",
    },
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
        qualia_path = self._config.telemetry_dir / f"qualia-{day_key_compact}.jsonl"
        qualia_records = load_qualia_log(qualia_path)
        thermo_policy = _memory_thermo_policy(self._config)
        previous_thermo = self._read_latest().get("memory_thermo")
        phase_ctx = _resolve_phase_context(thermo_policy, previous_thermo)
        defrag_result = _run_defrag_observe(qualia_records, thermo_policy, phase_ctx)
        self._write_latest("memory_thermo", defrag_result)
        life_indicator = compute_life_indicator_for_day(
            qualia_records,
            num_diary_entries=0,
            num_self_reflection_entries=0,
        )
        policy_prior = PolicyPrior()
        self._write_latest("life_indicator", life_indicator)
        self._write_latest("policy_prior", policy_prior)
        self._save_life_indicator(day, life_indicator)
        self._save_policy_prior(policy_prior)
        self._write_nightly_report(day, life_indicator, policy_prior)
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
    raw = getattr(config, "memory_thermo_policy", None)
    if not isinstance(raw, dict):
        return dict(DEFAULT_MEMORY_THERMO_POLICY)
    merged = dict(DEFAULT_MEMORY_THERMO_POLICY)
    merged.update(raw)
    return merged


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
