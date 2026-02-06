from __future__ import annotations

import json
import logging
import os
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
