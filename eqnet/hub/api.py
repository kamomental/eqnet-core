"""Core EQNet hub API (log_moment / run_nightly / query_state)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from eqnet.runtime.life_indicator import LifeIndicator
from eqnet.runtime.policy import PolicyPrior
from eqnet.runtime.state import QualiaState
from eqnet.runtime.turn import CoreState, SafetyConfig
from eqnet.orchestrators.hub_adapter import run_hub_turn
from eqnet.hub.moment_entry import to_moment_entry
from eqnet.hub.output_control import apply_policy_prior
from eqnet.hub.repair_fsm import (
    RepairEvent,
    RepairSnapshot,
    apply_repair_event,
    repair_fingerprint,
)
from eqnet.hub.text_policy import apply_text_policy
from eqnet.hub.trace_keys import (
    get_or_create_episode_id,
    resolve_day_key_from_as_of,
    resolve_day_key_from_date,
    resolve_day_key_from_moment,
)
from eqnet.hub.idempotency import IdempotencyStore, NoopIdempotencyStore
from eqnet.hub.runtime_contract import HubRuntime
from eqnet.runtime.external_runtime import (
    ExternalRuntimeDelegate,
    ExternalRuntimeDelegateV2,
)
from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path
from eqnet.telemetry.trace_writer import append_trace_event
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit
from eqnet.telemetry.mecpe_writer import (
    MecpeWriter,
    MecpeWriterConfig,
    build_minimal_mecpe_payload,
)
from eqnet.persona.loader import PersonaConfig
from emot_terrain_lab.ops.nightly_life_indicator import (
    load_qualia_log,
)


logger = logging.getLogger(__name__)

TRACE_FLAG_ENV = "EQNET_TRACE_V1"
TRACE_DIR_ENV = "EQNET_TRACE_V1_DIR"
RUNTIME_VERSION_ENV = "EQNET_RUNTIME_VERSION"
RUNTIME_IMPL_ENV = "EQNET_RUNTIME_IMPL"
EXTERNAL_RUNTIME_VERSION_ENV = "EQNET_EXTERNAL_RUNTIME_VERSION"
DELEGATION_MODE_ENV = "HUB_DELEGATION_MODE"
MISMATCH_POLICY_ENV = "HUB_MISMATCH_POLICY"
DEFAULT_RUNTIME_VERSION = "eqnet_hub_v1"
TRACE_SCHEMA_VERSION = "trace_v1"
REDACTED_TEXT = "<redacted>"


@dataclass
class EQNetConfig:
    telemetry_dir: Path = Path("telemetry")
    reports_dir: Path = Path("reports")
    state_dir: Path = Path("state")
    trace_dir: Path = Path("telemetry/trace_v1")
    audit_dir: Path = Path("telemetry/audit")
    memory_reference_log_path: Path | None = None
    audit_thresholds: Dict[str, Any] | None = None
    memory_thermo_policy: Dict[str, Any] | None = None
    runtime_policy: Dict[str, Any] | None = None


class EQNetHub:
    """Minimal face of EQNet's "心バックエンド"."""

    def __init__(
        self,
        config: Optional[EQNetConfig] = None,
        *,
        embed_text_fn: Callable[[str], Any],
        persona: Optional[PersonaConfig] = None,
        runtime_delegate: Optional[HubRuntime] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
        runtime_version: Optional[str] = None,
    ) -> None:
        if embed_text_fn is None:
            raise ValueError("embed_text_fn が必要です")
        self.config = config or EQNetConfig()
        self.embed_text_fn = embed_text_fn
        self.persona = persona
        self._latest_qualia_state: Optional[QualiaState] = None
        self._latest_life_indicator: Optional[LifeIndicator] = None
        self._latest_policy_prior: Optional[PolicyPrior] = None
        self._latest_memory_thermo: dict[str, Any] | None = None
        self._repair_snapshot: RepairSnapshot = RepairSnapshot.initial()
        self._trace_safety = SafetyConfig()
        self._runtime_delegate = self._resolve_runtime_delegate(runtime_delegate)
        self._idempotency_store = idempotency_store or NoopIdempotencyStore()
        self._mecpe_writer = MecpeWriter(MecpeWriterConfig(telemetry_dir=self.config.telemetry_dir))
        self._runtime_version = (
            runtime_version
            or os.getenv(RUNTIME_VERSION_ENV)
            or getattr(self._runtime_delegate, "runtime_version", None)
            or DEFAULT_RUNTIME_VERSION
        )
        if persona is not None:
            self._apply_persona_defaults(persona)

    # ------------------------------------------------------------------
    # 1) log_moment: raw_event/raw_text を心の入口へ流す
    # ------------------------------------------------------------------

    def log_moment(
        self,
        raw_event: Any,
        raw_text: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> None:
        idem_key = idempotency_key or self._derive_idempotency_key(
            op="log_moment",
            raw_event=raw_event,
            raw_text=raw_text,
        )
        mode = self._delegation_mode()
        mismatch_policy = self._mismatch_policy()
        delegate_name = self._delegate_name()
        reserved = self._idempotency_store.check_and_reserve(idem_key)
        moment_entry = self._to_moment_entry(raw_event, raw_text)
        repair_before = self._repair_snapshot
        repair_event, repair_reason_codes = self._detect_repair_trigger(raw_event, raw_text)
        repair_after = self._apply_repair_event(
            repair_before,
            event=repair_event,
            reason_codes=repair_reason_codes,
        )
        self._repair_snapshot = repair_after
        input_fingerprint = self._fingerprint_moment_input(raw_event, raw_text)
        if not reserved:
            self._emit_trace_v1(
                moment_entry,
                raw_text,
                raw_event=raw_event,
                runtime_version=self._runtime_version,
                idempotency_key=idem_key,
                idempotency_status="skipped",
                delegate_to=delegate_name,
                delegation_mode=mode,
                mismatch_policy=mismatch_policy,
                mismatch_reason_codes=[],
                moment_input_fingerprint=input_fingerprint,
                qualia_state_fingerprint=None,
                repair_state_before=repair_before.state.value,
                repair_state_after=repair_after.state.value,
                repair_event=repair_event.value,
                repair_reason_codes=list(repair_reason_codes),
                repair_snapshot_fingerprint=repair_fingerprint(repair_after),
            )
            return

        idem_status = "reserved"
        delegate_status = "not_called"
        mismatch_reason_codes: list[str] = []
        qualia_state_fingerprint: Optional[str] = None
        try:
            self._runtime_delegate.log_moment(
                raw_event,
                raw_text,
                idempotency_key=idem_key,
            )
            delegate_status = "ok"
            qualia_state_fingerprint = self._fingerprint_current_qualia_state()

            idem_status = "done"
            self._idempotency_store.mark_done(
                idem_key,
                self._fingerprint_from_moment(moment_entry),
            )
        except Exception as exc:
            mismatch_reason_codes.append("DELEGATE_EXCEPTION")
            idem_status = "failed"
            self._idempotency_store.mark_failed(idem_key, type(exc).__name__)
            raise
        finally:
            self._emit_trace_v1(
                moment_entry,
                raw_text,
                raw_event=raw_event,
                runtime_version=self._runtime_version,
                idempotency_key=idem_key,
                idempotency_status=idem_status,
                delegate_to=delegate_name,
                delegation_mode=mode,
                delegate_status=delegate_status,
                mismatch_policy=mismatch_policy,
                mismatch_reason_codes=mismatch_reason_codes,
                moment_input_fingerprint=input_fingerprint,
                qualia_state_fingerprint=qualia_state_fingerprint,
                repair_state_before=repair_before.state.value,
                repair_state_after=repair_after.state.value,
                repair_event=repair_event.value,
                repair_reason_codes=list(repair_reason_codes),
                repair_snapshot_fingerprint=repair_fingerprint(repair_after),
            )

    # ------------------------------------------------------------------
    # 2) run_nightly: 1 日分の danger/healing/life_indicator を更新
    # ------------------------------------------------------------------

    def run_nightly(
        self,
        date_obj: Optional[date] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> None:
        date_obj = date_obj or date.today()
        date_str = date_obj.strftime("%Y%m%d")
        qualia_path = self.config.telemetry_dir / f"qualia-{date_str}.jsonl"
        qualia_records = load_qualia_log(qualia_path)
        idem_key = idempotency_key or self._derive_nightly_idempotency_key(
            date_obj=date_obj,
            qualia_records=qualia_records,
        )
        mode = self._delegation_mode()
        mismatch_policy = self._mismatch_policy()
        delegate_name = self._delegate_name()
        reserved = self._idempotency_store.check_and_reserve(idem_key)
        if not reserved:
            self._emit_nightly_trace_v1(
                date_obj=date_obj,
                idempotency_key=idem_key,
                idempotency_status="skipped",
                delegate_to=delegate_name,
                delegation_mode=mode,
                delegate_status="not_called",
                mismatch_policy=mismatch_policy,
                mismatch_reason_codes=[],
                life_indicator_fingerprint=None,
                policy_prior_fingerprint=None,
                output_control_fingerprint=None,
                audit_fingerprint=None,
            )
            return

        idem_status = "reserved"
        delegate_status = "not_called"
        mismatch_reason_codes: list[str] = []
        life_indicator_fingerprint: Optional[str] = None
        policy_prior_fingerprint: Optional[str] = None
        output_control_fingerprint: Optional[str] = None
        audit_fingerprint: Optional[str] = None
        memory_thermo: dict[str, Any] | None = None
        try:
            day = date_obj
            self._runtime_delegate.run_nightly(
                day,
                idempotency_key=idem_key,
            )
            delegate_status = "ok"
            current = self._snapshot_nightly_fingerprints(day)
            life_indicator_fingerprint = current["life_indicator_fingerprint"]
            policy_prior_fingerprint = current["policy_prior_fingerprint"]
            output_control_fingerprint = current["output_control_fingerprint"]
            audit_fingerprint = current["audit_fingerprint"]
            memory_thermo = current.get("memory_thermo")

            idem_status = "done"
            result_fingerprint = self._fingerprint_json_payload(
                {
                    "day": date_obj.isoformat(),
                    "life_indicator": life_indicator_fingerprint,
                    "policy_prior": policy_prior_fingerprint,
                    "output_control": output_control_fingerprint,
                    "audit": audit_fingerprint,
                    "delegate_status": delegate_status,
                }
            )
            self._idempotency_store.mark_done(idem_key, result_fingerprint)
        except Exception as exc:
            mismatch_reason_codes.append("DELEGATE_EXCEPTION")
            idem_status = "failed"
            self._idempotency_store.mark_failed(idem_key, type(exc).__name__)
            raise
        finally:
            self._emit_nightly_trace_v1(
                date_obj=date_obj,
                idempotency_key=idem_key,
                idempotency_status=idem_status,
                delegate_to=delegate_name,
                delegation_mode=mode,
                delegate_status=delegate_status,
                mismatch_policy=mismatch_policy,
                mismatch_reason_codes=mismatch_reason_codes,
                life_indicator_fingerprint=life_indicator_fingerprint,
                policy_prior_fingerprint=policy_prior_fingerprint,
                output_control_fingerprint=output_control_fingerprint,
                audit_fingerprint=audit_fingerprint,
                memory_thermo=memory_thermo,
            )
            # Rebuild nightly audit after hub trace emission so coverage checks
            # observe the final closed-loop trace set for the day.
            self._run_nightly_audit(date_obj)

    # ------------------------------------------------------------------
    # 3) query_state: UI/Persona が "今" と "今日" を見る窓
    # ------------------------------------------------------------------

    def query_state(
        self,
        *,
        as_of: Optional[str] = None,
    ) -> dict:
        normalized_as_of = resolve_day_key_from_as_of(as_of) or as_of
        mode = self._delegation_mode()
        mismatch_policy = self._mismatch_policy()
        delegate_name = self._delegate_name()
        delegate_status = "not_called"
        mismatch_reason_codes: list[str] = []
        resolved_day_key: Optional[str] = None
        delegate_state_fingerprint: Optional[str] = None

        if self._runtime_delegate is None:
            raise RuntimeError("query_state requires runtime_delegate")

        raw_delegate_state = self._runtime_delegate.query_state(as_of=normalized_as_of)
        result_state = dict(raw_delegate_state or {})
        delegate_status = "ok"
        delegate_resolved = resolve_day_key_from_as_of(result_state.get("resolved_day_key"))
        if delegate_resolved is not None:
            resolved_day_key = delegate_resolved
        elif normalized_as_of is not None:
            resolved_day_key = resolve_day_key_from_as_of(normalized_as_of)
            mismatch_reason_codes.append("missing_resolved_day_key")
        else:
            mismatch_reason_codes.append("missing_resolved_day_key")
            resolved_day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if (
            normalized_as_of is not None
            and resolved_day_key is not None
            and resolve_day_key_from_as_of(normalized_as_of) != resolved_day_key
        ):
            mismatch_reason_codes.append("resolved_day_key_diff")

        if mismatch_reason_codes:
            delegate_status = "mismatch"
            if mismatch_policy == "fail":
                mismatch_reason_codes.append("policy_fail")

        result_state["resolved_day_key"] = resolved_day_key

        result_fingerprint = self._fingerprint_json_payload(
            _normalize_for_hash(result_state)
        )
        delegate_state_fingerprint = result_fingerprint
        closed_loop_fp = self._snapshot_closed_loop_fingerprints()
        self._emit_query_state_trace_v1(
            as_of=normalized_as_of,
            resolved_day_key=str(result_state.get("resolved_day_key") or resolved_day_key),
            delegate_to=delegate_name,
            delegation_mode=mode,
            delegate_status=delegate_status,
            mismatch_policy=mismatch_policy,
            mismatch_reason_codes=mismatch_reason_codes,
            state_fingerprint=result_fingerprint,
            delegate_state_fingerprint=delegate_state_fingerprint,
            local_state_fingerprint=None,
            life_indicator_fingerprint=closed_loop_fp["life_indicator_fingerprint"],
            policy_prior_fingerprint=closed_loop_fp["policy_prior_fingerprint"],
            output_control_fingerprint=closed_loop_fp["output_control_fingerprint"],
        )
        if delegate_status == "mismatch" and mismatch_policy == "fail":
            raise RuntimeError("query_state shadow mismatch detected under fail policy")
        return result_state

    # --- internal helpers -------------------------------------------------


    def latest_qualia_state(self) -> Optional[QualiaState]:
        """Return the most recent QualiaState if available."""

        return self._latest_qualia_state

    def latest_policy_prior(self) -> PolicyPrior:
        """Return the last computed PolicyPrior (default when None)."""

        return self._latest_policy_prior or PolicyPrior()

    def _to_moment_entry(self, raw_event: Any, raw_text: str) -> Any:
        # TODO: raw_event/raw_text を MomentLogEntry に変換
        return raw_event

    def _append_moment_log(self, moment_entry: Any) -> None:
        # TODO: 既存の MomentLog へ書き込み
        pass

    def _count_diary_entries(self, date_obj: date) -> int:
        return 0

    def _count_self_reflection_entries(self, date_obj: date) -> int:
        return 0

    def _run_danger_healing_and_policy_updates(
        self,
        date_obj: date,
        qualia_records: list[dict],
        life_indicator: LifeIndicator,
    ) -> PolicyPrior:
        # TODO: Danger map / Healing / imagery replay を実行
        return PolicyPrior()

    def _save_life_indicator(self, date_obj: date, li: LifeIndicator) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.state_dir / f"life-indicator-{date_obj.strftime('%Y%m%d')}.json"
        payload = {
            "identity": li.identity_score,
            "qualia": li.qualia_score,
            "meta_awareness": li.meta_awareness_score,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_policy_prior(self, pp: PolicyPrior) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.state_dir / "policy-prior-latest.json"
        path.write_text(json.dumps(pp.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_nightly_report(self, date_obj: date, li: LifeIndicator, pp: PolicyPrior) -> None:
        """Persist a lightweight nightly summary for inspection."""

        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "life_indicator": {
                "identity": li.identity_score,
                "qualia": li.qualia_score,
                "meta_awareness": li.meta_awareness_score,
            },
            "policy_prior": pp.__dict__,
        }
        path = self.config.reports_dir / f"nightly-{date_obj.strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _run_nightly_audit(self, date_obj: date) -> None:
        """Generate read-only nightly audit artifacts for trace_v1."""

        day = date_obj.strftime("%Y-%m-%d")
        trace_root = Path(os.getenv(TRACE_DIR_ENV) or getattr(self.config, "trace_dir", self.config.telemetry_dir / "trace_v1"))
        out_dir = Path(getattr(self.config, "audit_dir", self.config.telemetry_dir / "audit"))
        day_dir = trace_root / day
        if not day_dir.exists():
            logger.info("nightly audit skipped; no trace dir for %s", day)
            return
        if not any(day_dir.glob("*.jsonl")):
            logger.info("nightly audit skipped; no trace files for %s", day)
            return
        thresholds = getattr(self.config, "audit_thresholds", None)
        cfg = NightlyAuditConfig(
            trace_root=trace_root,
            out_root=out_dir,
            date_yyyy_mm_dd=day,
            boundary_threshold=self._trace_safety.boundary_threshold,
            health_thresholds=thresholds,
            memory_reference_log_path=getattr(self.config, "memory_reference_log_path", None),
        )
        try:
            generate_audit(cfg)
        except Exception:
            logger.warning("nightly audit failed for %s", day, exc_info=True)

    def _load_latest_life_indicator(self) -> Optional[LifeIndicator]:
        # TODO: state_dir から最新ファイルを読み込む
        return None

    def _load_latest_policy_prior(self) -> Optional[PolicyPrior]:
        path = self.config.state_dir / "policy-prior-latest.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PolicyPrior(**data)

    def _load_recent_danger_metrics(self) -> dict:
        return {}

    def _load_recent_healing_metrics(self) -> dict:
        return {}

    def _apply_persona_defaults(self, persona: PersonaConfig) -> None:
        initial_pp = persona.qfs.get("initial_policy_prior") if persona.qfs else None
        if initial_pp:
            self._latest_policy_prior = PolicyPrior(
                warmth=float(initial_pp.get("warmth", 0.5)),
                directness=float(initial_pp.get("directness", 0.5)),
                self_disclosure=float(initial_pp.get("self_disclosure", 0.5)),
                calmness=float(initial_pp.get("calmness", 0.5)),
            )
            # extend PolicyPrior with risk/thrill/discount attributes if they exist
            self._latest_policy_prior.risk_aversion = float(initial_pp.get("risk_aversion", 0.5))
            self._latest_policy_prior.thrill_gain = float(initial_pp.get("thrill_gain", 0.5))
            self._latest_policy_prior.discount_rate = float(initial_pp.get("discount_rate", 0.5))
        initial_li = persona.qfs.get("initial_life_indicator") if persona.qfs else None
        if initial_li:
            self._latest_life_indicator = LifeIndicator(
                identity_score=float(initial_li.get("identity", 0.5)),
                qualia_score=float(initial_li.get("qualia", 0.5)),
                meta_awareness_score=float(initial_li.get("meta_awareness", 0.5)),
            ).clamp()

    def _detect_repair_trigger(self, raw_event: Any, raw_text: str) -> tuple[RepairEvent, list[str]]:
        _ = raw_text  # reserved for future heuristic expansion
        trigger = False
        event_value: Any = None
        if isinstance(raw_event, Mapping):
            trigger = bool(raw_event.get("repair_trigger"))
            reason_value = raw_event.get("reason_codes")
            event_value = raw_event.get("repair_event")
        else:
            trigger = bool(getattr(raw_event, "repair_trigger", False))
            reason_value = getattr(raw_event, "reason_codes", None)
            event_value = getattr(raw_event, "repair_event", None)
        event = self._normalize_repair_event(event_value)
        if event is RepairEvent.NONE and trigger:
            event = RepairEvent.TRIGGER
        if event is RepairEvent.NONE:
            return RepairEvent.NONE, []
        reason_codes = self._normalize_repair_reason_codes(reason_value)
        if event is RepairEvent.TRIGGER and not reason_codes:
            reason_codes = ["REPAIR_TRIGGERED"]
        return event, reason_codes

    def _normalize_repair_event(self, value: Any) -> RepairEvent:
        if value is None:
            return RepairEvent.NONE
        text = str(value).strip().upper()
        try:
            return RepairEvent(text)
        except ValueError:
            return RepairEvent.NONE

    def _normalize_repair_reason_codes(self, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        normalized: list[str] = []
        for item in value:
            text = str(item).strip().upper()
            if not text:
                continue
            code = "".join(ch if ("A" <= ch <= "Z" or "0" <= ch <= "9" or ch == "_") else "_" for ch in text)
            code = code.strip("_")
            if not code:
                continue
            normalized.append(code)
        # Keep order while deduplicating.
        return list(dict.fromkeys(normalized))

    def _apply_repair_event(
        self,
        snapshot: RepairSnapshot,
        *,
        event: RepairEvent,
        reason_codes: list[str],
    ) -> RepairSnapshot:
        return apply_repair_event(
            snapshot,
            event=event,
            reason_codes=reason_codes,
            now_ts=datetime.now(timezone.utc).timestamp(),
        )

    def _memory_thermo_contract_fields(
        self,
        *,
        memory_entropy_delta: float | None = None,
        entropy_cost_class: str | None = None,
        irreversible_op: bool | None = None,
        entropy_budget_ok: bool | None = None,
        memory_phase: str | None = None,
        phase_weight_profile: str | None = None,
        value_projection_fingerprint: str | None = None,
        energy_budget_used: float | None = None,
        energy_budget_limit: float | None = None,
        budget_throttle_applied: bool | None = None,
        throttle_reason_code: str | None = None,
        output_control_profile: str | None = None,
        phase_override_applied: bool | None = None,
        policy_version: str | None = None,
        entropy_model_id: str | None = None,
    ) -> dict[str, Any]:
        phase = memory_phase or "stabilization"
        profile = phase_weight_profile or "default"
        projection_fp = value_projection_fingerprint or self._fingerprint_json_payload(
            {"memory_phase": phase, "phase_weight_profile": profile}
        )
        return {
            "memory_entropy_delta": float(memory_entropy_delta if memory_entropy_delta is not None else 0.0),
            "entropy_cost_class": entropy_cost_class or "LOW",
            "irreversible_op": bool(irreversible_op) if irreversible_op is not None else False,
            "entropy_budget_ok": bool(entropy_budget_ok) if entropy_budget_ok is not None else True,
            "memory_phase": phase,
            "phase_weight_profile": profile,
            "value_projection_fingerprint": projection_fp,
            "energy_budget_used": float(energy_budget_used if energy_budget_used is not None else 0.0),
            "energy_budget_limit": float(energy_budget_limit if energy_budget_limit is not None else 0.0),
            "budget_throttle_applied": bool(budget_throttle_applied) if budget_throttle_applied is not None else False,
            "throttle_reason_code": str(throttle_reason_code or ""),
            "output_control_profile": str(output_control_profile or "normal_v1"),
            "phase_override_applied": bool(phase_override_applied) if phase_override_applied is not None else False,
            "policy_version": policy_version or "memory-ops-v1",
            "entropy_model_id": entropy_model_id or "entropy-model-v1",
        }

    def _emit_trace_v1(
        self,
        moment_entry: Any,
        raw_text: str,
        *,
        raw_event: Any = None,
        runtime_version: str,
        idempotency_key: str,
        idempotency_status: str,
        delegate_to: str,
        delegation_mode: str,
        delegate_status: str = "unknown",
        mismatch_policy: str = "warn",
        mismatch_reason_codes: list[str] | None = None,
        moment_input_fingerprint: Optional[str] = None,
        qualia_state_fingerprint: Optional[str] = None,
        repair_state_before: Optional[str] = None,
        repair_state_after: Optional[str] = None,
        repair_event: Optional[str] = None,
        repair_reason_codes: list[str] | None = None,
        repair_snapshot_fingerprint: Optional[str] = None,
    ) -> None:
        if not _env_truthy(TRACE_FLAG_ENV):
            return
        try:
            thin_entry = to_moment_entry(moment_entry)
            if "user_text" not in thin_entry:
                thin_entry["user_text"] = raw_text
            thin_entry["runtime_version"] = runtime_version
            thin_entry["idempotency_key"] = idempotency_key
            allow_raw = os.getenv("EQNET_TRACE_ALLOW_RAW_TEXT") == "1"
            policy = getattr(self._trace_safety, "text_policy", "redact")
            truncate = getattr(self._trace_safety, "text_truncate_chars", 200)
            sanitized_text, text_obs = apply_text_policy(
                thin_entry.get("user_text"),
                policy=policy,
                allow_raw_env=allow_raw,
                truncate_chars=truncate,
            )
            thin_entry["user_text"] = sanitized_text
            if sanitized_text is not None:
                trace_obs = thin_entry.setdefault("trace_observations", {}).setdefault("qualia", {})
                trace_obs["user_text"] = {"policy": policy, **text_obs}
            policy_obs = thin_entry.setdefault("trace_observations", {}).setdefault("policy", {})
            day_key = resolve_day_key_from_moment(_moment_timestamp(moment_entry))
            episode_id = get_or_create_episode_id(moment_entry)
            thermo = dict(self._latest_memory_thermo or {})
            output_control = apply_policy_prior(
                self.latest_policy_prior(),
                day_key=day_key,
                episode_id=episode_id,
                repair_snapshot=self._repair_snapshot,
                budget_throttle_applied=bool(thermo.get("budget_throttle_applied")),
                output_control_profile=(
                    str(thermo.get("output_control_profile"))
                    if thermo.get("output_control_profile")
                    else None
                ),
                throttle_reason_code=(
                    str(thermo.get("throttle_reason_code"))
                    if thermo.get("throttle_reason_code")
                    else None
                ),
            )
            output_control_payload = output_control.to_fingerprint_payload()
            output_control_fingerprint = self._fingerprint_json_payload(output_control_payload)
            policy_obs.update(
                {
                    "event_id": f"hub:{episode_id}:{int(_moment_timestamp(moment_entry).timestamp() * 1000)}",
                    "trace_id": idempotency_key or f"trace:{day_key}:{episode_id}",
                    "delegate_to": delegate_to,
                    "delegation_mode": delegation_mode,
                    "delegate_status": delegate_status,
                    "idempotency_status": idempotency_status,
                    "mismatch_policy": mismatch_policy,
                    "mismatch_reason_codes": list(mismatch_reason_codes or []),
                    "day_key": day_key,
                    "episode_id": episode_id,
                    "control_applied_at": output_control.control_applied_at,
                    "output_control_repair_state": output_control.repair_state,
                    "output_control_profile": output_control.output_control_profile,
                    "throttle_reason_code": output_control.throttle_reason_code,
                    "repair_state_before": repair_state_before or self._repair_snapshot.state.value,
                    "repair_state_after": repair_state_after or self._repair_snapshot.state.value,
                    "repair_event": repair_event or RepairEvent.NONE.value,
                    "repair_reason_codes": list(repair_reason_codes or self._repair_snapshot.reason_codes),
                    "repair_fingerprint": repair_snapshot_fingerprint or repair_fingerprint(self._repair_snapshot),
                    "phase_override_applied": bool(thermo.get("phase_override_applied", False)),
                    **self._memory_thermo_contract_fields(
                        memory_entropy_delta=_maybe_float(thermo.get("memory_entropy_delta")),
                        entropy_cost_class=str(thermo.get("entropy_cost_class") or "LOW"),
                        irreversible_op=bool(thermo.get("irreversible_op")) if "irreversible_op" in thermo else None,
                        entropy_budget_ok=bool(thermo.get("entropy_budget_ok")) if "entropy_budget_ok" in thermo else None,
                        memory_phase=str(thermo.get("memory_phase") or "stabilization"),
                        phase_weight_profile=str(thermo.get("phase_weight_profile") or "default"),
                        value_projection_fingerprint=(
                            str(thermo.get("value_projection_fingerprint"))
                            if thermo.get("value_projection_fingerprint")
                            else None
                        ),
                        energy_budget_used=_maybe_float(thermo.get("energy_budget_used")),
                        energy_budget_limit=_maybe_float(thermo.get("energy_budget_limit")),
                        budget_throttle_applied=(
                            bool(thermo.get("budget_throttle_applied"))
                            if "budget_throttle_applied" in thermo
                            else None
                        ),
                        throttle_reason_code=str(thermo.get("throttle_reason_code") or ""),
                        output_control_profile=str(
                            thermo.get("output_control_profile")
                            or output_control.output_control_profile
                            or "normal_v1"
                        ),
                        policy_version=str(thermo.get("policy_version") or "memory-ops-v1"),
                        entropy_model_id=str(thermo.get("entropy_model_id") or "entropy-model-v1"),
                    ),
                }
            )
            closed_loop_fp = self._snapshot_closed_loop_fingerprints()
            qualia_obs = thin_entry.setdefault("trace_observations", {}).setdefault("qualia", {})
            qualia_obs.update(
                {
                    "moment_input_fingerprint": moment_input_fingerprint,
                    "qualia_state_fingerprint": qualia_state_fingerprint,
                    "life_indicator_fingerprint": closed_loop_fp["life_indicator_fingerprint"],
                    "policy_prior_fingerprint": closed_loop_fp["policy_prior_fingerprint"],
                    "output_control_fingerprint": output_control_fingerprint,
                }
            )

            payload, _stamp = _build_trace_payload(
                thin_entry,
                fallback_session=getattr(self.persona, "persona_id", None),
            )
            text_hash_hint = text_obs.get("sha256") if isinstance(text_obs, dict) else None
            self._emit_mecpe_v0(
                payload=payload,
                raw_text=raw_text,
                raw_event=raw_event if raw_event is not None else moment_entry,
                text_hash_hint=text_hash_hint,
            )
            root_override = os.getenv(TRACE_DIR_ENV)
            base_dir = Path(root_override) if root_override else self.config.telemetry_dir / "trace_v1"
            target = trace_output_path(
                TracePathConfig(base_dir=base_dir, source_loop="hub"),
                timestamp_ms=payload.get("timestamp_ms"),
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            state = CoreState(policy_prior=self.latest_policy_prior())
            run_hub_turn(payload, state, self._trace_safety, target)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning("trace_v1 emit failed", exc_info=exc)

    def _emit_mecpe_v0(
        self,
        *,
        payload: dict[str, Any],
        raw_text: str,
        raw_event: Any,
        text_hash_hint: str | None = None,
    ) -> None:
        try:
            timestamp_ms = int(payload.get("timestamp_ms") or 0)
            turn_id = str(payload.get("turn_id") or "")
            if timestamp_ms <= 0 or not turn_id:
                return
            base = build_minimal_mecpe_payload(
                timestamp_ms=timestamp_ms,
                turn_id=turn_id,
                raw_text=raw_text,
                raw_event=raw_event,
                text_hash_override=text_hash_hint,
            )
            self._mecpe_writer.append_turn(
                timestamp_ms=base["timestamp_ms"],
                turn_id=base["turn_id"],
                prompt_hash=base["prompt_hash"],
                model_version=base["model_version"],
                text_hash=base["text_hash"],
                audio_sha256=base["audio_sha256"],
                video_sha256=base["video_sha256"],
                extra=None,
            )
        except Exception:
            logger.warning("mecpe_v0 emit failed", exc_info=True)

    def _delegation_mode(self) -> str:
        raw = (os.getenv(DELEGATION_MODE_ENV) or "off").strip().lower()
        if raw in {"off", "shadow", "on"}:
            return raw
        return "off"

    def _mismatch_policy(self) -> str:
        raw = (os.getenv(MISMATCH_POLICY_ENV) or "warn").strip().lower()
        if raw in {"warn", "fail"}:
            return raw
        return "warn"

    def _delegate_name(self) -> str:
        if self._runtime_delegate is None:
            return "none"
        klass = self._runtime_delegate.__class__
        return f"{klass.__module__}.{klass.__name__}"

    def _resolve_runtime_delegate(
        self,
        runtime_delegate: Optional[HubRuntime],
    ) -> HubRuntime:
        if runtime_delegate is not None:
            return runtime_delegate
        runtime_impl = (os.getenv(RUNTIME_IMPL_ENV) or "external_v2").strip().lower()
        if runtime_impl == "external":
            version = (
                os.getenv(EXTERNAL_RUNTIME_VERSION_ENV)
                or "external_runtime_v1"
            )
            return ExternalRuntimeDelegate(self, runtime_version=version)
        if runtime_impl == "external_v2":
            version = (
                os.getenv(EXTERNAL_RUNTIME_VERSION_ENV)
                or "external_runtime_v2"
            )
            return ExternalRuntimeDelegateV2(
                config=self.config,
                embed_text_fn=self.embed_text_fn,
                latest_state_writer=self._set_latest_state,
                latest_state_reader=self._get_latest_state,
                runtime_version=version,
                trace_dir_env=TRACE_DIR_ENV,
                boundary_threshold=self._trace_safety.boundary_threshold,
            )
        raise RuntimeError(
            "runtime_delegate resolution failed: set runtime_delegate or EQNET_RUNTIME_IMPL=external_v2/external"
        )

    def _set_latest_state(self, kind: str, value: Any) -> None:
        if kind == "qualia":
            self._latest_qualia_state = value
        elif kind == "life_indicator":
            self._latest_life_indicator = value
        elif kind == "policy_prior":
            self._latest_policy_prior = value
        elif kind == "memory_thermo":
            self._latest_memory_thermo = dict(value or {})

    def _get_latest_state(self) -> Dict[str, Any]:
        return {
            "qualia": self._latest_qualia_state,
            "life_indicator": self._latest_life_indicator,
            "policy_prior": self._latest_policy_prior,
            "memory_thermo": dict(self._latest_memory_thermo or {}),
        }

    def _derive_idempotency_key(
        self,
        *,
        op: str,
        raw_event: Any,
        raw_text: str,
    ) -> str:
        entry = to_moment_entry(raw_event)
        stamp = _moment_timestamp(entry)
        scenario, turn_id, _seed = _derive_identifiers(
            entry,
            getattr(self.persona, "persona_id", None),
        )
        text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        base = "|".join(
            [op, scenario, turn_id, str(int(stamp.timestamp() * 1000)), text_hash]
        )
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
        return f"{op}:{digest[:24]}"

    def _fingerprint_from_moment(self, moment_entry: Any) -> str:
        stamp = _moment_timestamp(moment_entry)
        payload = {
            "timestamp_ms": int(stamp.timestamp() * 1000),
            "turn_id": _entry_value(moment_entry, "turn_id"),
            "session_id": _entry_value(moment_entry, "session_id"),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _fingerprint_moment_input(self, raw_event: Any, raw_text: str) -> str:
        entry = to_moment_entry(raw_event)
        norm_entry = _normalize_for_hash(entry)
        payload = {
            "entry": norm_entry,
            "text_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
        }
        return self._fingerprint_json_payload(payload)

    def _fingerprint_current_qualia_state(self) -> Optional[str]:
        qstate = self._latest_qualia_state
        if qstate is None:
            return None
        vec = getattr(qstate, "qualia_vec", None)
        if vec is None:
            norm = _normalize_for_hash(getattr(qstate, "__dict__", {}))
            return self._fingerprint_json_payload(norm)
        if hasattr(vec, "tolist"):
            values = vec.tolist()
        else:
            values = list(vec)
        floats = []
        for item in values:
            try:
                floats.append(round(float(item), 6))
            except (TypeError, ValueError):
                floats.append(0.0)
        payload = {
            "dimension": len(floats),
            "head": floats[:64],
        }
        return self._fingerprint_json_payload(payload)

    def _derive_nightly_idempotency_key(
        self,
        *,
        date_obj: date,
        qualia_records: list[dict[str, Any]],
    ) -> str:
        first_ts = None
        last_ts = None
        if qualia_records:
            first_ts = qualia_records[0].get("timestamp_ms") or qualia_records[0].get("timestamp")
            last_ts = qualia_records[-1].get("timestamp_ms") or qualia_records[-1].get("timestamp")
        summary = {
            "op": "run_nightly",
            "day": date_obj.isoformat(),
            "count": len(qualia_records),
            "first_ts": first_ts,
            "last_ts": last_ts,
            "runtime_version": self._runtime_version,
            "schema_version": TRACE_SCHEMA_VERSION,
        }
        digest = hashlib.sha256(
            json.dumps(summary, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"run_nightly:{digest[:24]}"

    def _snapshot_nightly_fingerprints(self, day: date) -> dict[str, Any]:
        closed_loop = self._snapshot_closed_loop_fingerprints()
        audit_fingerprint = self._load_audit_fingerprint(day)
        return {
            "life_indicator_fingerprint": closed_loop["life_indicator_fingerprint"],
            "policy_prior_fingerprint": closed_loop["policy_prior_fingerprint"],
            "output_control_fingerprint": closed_loop["output_control_fingerprint"],
            "audit_fingerprint": audit_fingerprint,
            "memory_thermo": dict(self._latest_memory_thermo or {}),
        }

    def _snapshot_closed_loop_fingerprints(self) -> dict[str, Optional[str]]:
        life_indicator_fingerprint = None
        if self._latest_life_indicator is not None:
            life_indicator_fingerprint = self._fingerprint_json_payload(
                {
                    "identity": self._latest_life_indicator.identity_score,
                    "qualia": self._latest_life_indicator.qualia_score,
                    "meta_awareness": self._latest_life_indicator.meta_awareness_score,
                }
            )
        policy_prior_fingerprint = None
        if self._latest_policy_prior is not None:
            policy_prior_fingerprint = self._fingerprint_json_payload(
                _normalize_for_hash(self._latest_policy_prior.__dict__)
            )
        output_control_payload = {
            "policy_prior": _normalize_for_hash(
                self._latest_policy_prior.__dict__ if self._latest_policy_prior is not None else {}
            ),
            "trace_safety": {
                "boundary_threshold": getattr(self._trace_safety, "boundary_threshold", None),
                "text_policy": getattr(self._trace_safety, "text_policy", None),
                "text_truncate_chars": getattr(self._trace_safety, "text_truncate_chars", None),
            },
        }
        output_control_fingerprint = self._fingerprint_json_payload(output_control_payload)
        return {
            "life_indicator_fingerprint": life_indicator_fingerprint,
            "policy_prior_fingerprint": policy_prior_fingerprint,
            "output_control_fingerprint": output_control_fingerprint,
        }

    def _fingerprint_json_payload(self, payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load_audit_fingerprint(self, date_obj: date) -> Optional[str]:
        day = date_obj.strftime("%Y-%m-%d")
        audit_path = Path(getattr(self.config, "audit_dir", self.config.telemetry_dir / "audit")) / f"nightly_audit_{day}.json"
        if not audit_path.exists():
            return None
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return self._fingerprint_json_payload(payload)

    def _emit_nightly_trace_v1(
        self,
        *,
        date_obj: date,
        idempotency_key: str,
        idempotency_status: str,
        delegate_to: str,
        delegation_mode: str,
        delegate_status: str,
        mismatch_policy: str,
        mismatch_reason_codes: list[str],
        life_indicator_fingerprint: Optional[str],
        policy_prior_fingerprint: Optional[str],
        output_control_fingerprint: Optional[str],
        audit_fingerprint: Optional[str],
        memory_thermo: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _env_truthy(TRACE_FLAG_ENV):
            return
        stamp = datetime(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            tzinfo=timezone.utc,
        )
        root_override = os.getenv(TRACE_DIR_ENV)
        base_dir = Path(root_override) if root_override else self.config.telemetry_dir / "trace_v1"
        target = trace_output_path(
            TracePathConfig(base_dir=base_dir, source_loop="hub"),
            timestamp_ms=int(stamp.timestamp() * 1000),
        )
        thermo = dict(memory_thermo or {})
        thermo_fields = self._memory_thermo_contract_fields(
            memory_entropy_delta=_maybe_float(thermo.get("memory_entropy_delta")),
            entropy_cost_class=str(thermo.get("entropy_cost_class") or "MID"),
            irreversible_op=bool(thermo.get("irreversible_op")) if "irreversible_op" in thermo else None,
            entropy_budget_ok=bool(thermo.get("entropy_budget_ok")) if "entropy_budget_ok" in thermo else None,
            memory_phase=str(thermo.get("memory_phase") or "stabilization"),
            phase_weight_profile=str(thermo.get("phase_weight_profile") or "default"),
            value_projection_fingerprint=(
                str(thermo.get("value_projection_fingerprint"))
                if thermo.get("value_projection_fingerprint")
                else None
            ),
            energy_budget_used=_maybe_float(thermo.get("energy_budget_used")),
            energy_budget_limit=_maybe_float(thermo.get("energy_budget_limit")),
            budget_throttle_applied=(
                bool(thermo.get("budget_throttle_applied"))
                if "budget_throttle_applied" in thermo
                else None
            ),
            throttle_reason_code=str(thermo.get("throttle_reason_code") or ""),
            output_control_profile=str(thermo.get("output_control_profile") or "normal_v1"),
            policy_version=str(thermo.get("policy_version") or "memory-ops-v1"),
            entropy_model_id=str(thermo.get("entropy_model_id") or "defrag-observe-v1"),
        )
        event = {
            "event_id": f"hub:nightly:{date_obj.isoformat()}:{idempotency_key}",
            "trace_id": idempotency_key,
            "schema_version": TRACE_SCHEMA_VERSION,
            "source_loop": "hub",
            "runtime_version": self._runtime_version,
            "idempotency_key": idempotency_key,
            "scenario_id": "nightly",
            "turn_id": f"nightly-{date_obj.isoformat()}",
            "seed": abs(hash(idempotency_key)) % 1_000_000 + 1,
            "timestamp_ms": int(stamp.timestamp() * 1000),
            "boundary": {},
            "self": {},
            "prospection": {},
            "policy": {
                "observations": {
                    "hub": {
                        "operation": "run_nightly",
                        "day": date_obj.isoformat(),
                        "day_key": resolve_day_key_from_date(date_obj),
                        "episode_id": "nightly",
                        "control_applied_at": "nightly",
                        "repair_state_before": self._repair_snapshot.state.value,
                        "repair_state_after": self._repair_snapshot.state.value,
                        "repair_event": RepairEvent.NONE.value,
                        "repair_reason_codes": list(self._repair_snapshot.reason_codes),
                        "repair_fingerprint": repair_fingerprint(self._repair_snapshot),
                        "delegation_mode": delegation_mode,
                        "delegate_to": delegate_to,
                        "delegate_status": delegate_status,
                        "idempotency_status": idempotency_status,
                        "mismatch_policy": mismatch_policy,
                        "mismatch_reason_codes": mismatch_reason_codes,
                        "event_id": f"hub:nightly:{date_obj.isoformat()}:{idempotency_key}",
                        "trace_id": idempotency_key,
                        "defrag_status": str(thermo.get("defrag_status") or "unknown"),
                        "defrag_mode": str(thermo.get("defrag_mode") or "unknown"),
                        "defrag_metrics_before": dict(thermo.get("defrag_metrics_before") or {}),
                        "defrag_metrics_after": dict(thermo.get("defrag_metrics_after") or {}),
                        "defrag_metrics_delta": dict(thermo.get("defrag_metrics_delta") or {}),
                        "phase_override_applied": bool(thermo.get("phase_override_applied", False)),
                        **thermo_fields,
                    }
                }
            },
            "qualia": {
                "observations": {
                    "hub": {
                        "life_indicator_fingerprint": life_indicator_fingerprint,
                        "policy_prior_fingerprint": policy_prior_fingerprint,
                        "output_control_fingerprint": output_control_fingerprint,
                        "audit_fingerprint": audit_fingerprint,
                    }
                }
            },
            "invariants": {},
        }
        append_trace_event(target, event)

    def _emit_query_state_trace_v1(
        self,
        *,
        as_of: Optional[str],
        resolved_day_key: str,
        delegate_to: str,
        delegation_mode: str,
        delegate_status: str,
        mismatch_policy: str,
        mismatch_reason_codes: list[str],
        state_fingerprint: str,
        delegate_state_fingerprint: Optional[str],
        local_state_fingerprint: Optional[str],
        life_indicator_fingerprint: Optional[str],
        policy_prior_fingerprint: Optional[str],
        output_control_fingerprint: Optional[str],
    ) -> None:
        if not _env_truthy(TRACE_FLAG_ENV):
            return
        normalized = resolve_day_key_from_as_of(resolved_day_key)
        if normalized is not None:
            stamp = datetime.strptime(normalized, "%Y-%m-%d").replace(
                tzinfo=timezone.utc,
                hour=12,
                minute=0,
                second=0,
            )
        else:
            stamp = datetime.now(timezone.utc)
        now_ms = int(stamp.timestamp() * 1000)
        root_override = os.getenv(TRACE_DIR_ENV)
        base_dir = Path(root_override) if root_override else self.config.telemetry_dir / "trace_v1"
        target = trace_output_path(
            TracePathConfig(base_dir=base_dir, source_loop="hub"),
            timestamp_ms=now_ms,
        )
        seed_source = f"query_state|{as_of}|{resolved_day_key}|{state_fingerprint}"
        event = {
            "event_id": f"hub:query_state:{resolved_day_key}:{now_ms}",
            "trace_id": f"query_state:{resolved_day_key}",
            "schema_version": TRACE_SCHEMA_VERSION,
            "source_loop": "hub",
            "runtime_version": self._runtime_version,
            "idempotency_key": "",
            "scenario_id": "query_state",
            "turn_id": f"query-state-{now_ms}",
            "seed": abs(hash(seed_source)) % 1_000_000 + 1,
            "timestamp_ms": now_ms,
            "boundary": {},
            "self": {},
            "prospection": {},
            "policy": {
                "observations": {
                    "hub": {
                        "operation": "query_state",
                        "as_of": as_of,
                        "resolved_day_key": resolved_day_key,
                        "day_key": resolve_day_key_from_as_of(resolved_day_key) or resolved_day_key,
                        "episode_id": "query_state",
                        "control_applied_at": "query_state",
                        "repair_state_before": self._repair_snapshot.state.value,
                        "repair_state_after": self._repair_snapshot.state.value,
                        "repair_event": RepairEvent.NONE.value,
                        "repair_reason_codes": list(self._repair_snapshot.reason_codes),
                        "repair_fingerprint": repair_fingerprint(self._repair_snapshot),
                        "delegation_mode": delegation_mode,
                        "configured_delegation_mode": delegation_mode,
                        "delegate_to": delegate_to,
                        "delegate_status": delegate_status,
                        "mismatch_policy": mismatch_policy,
                        "mismatch_reason_codes": mismatch_reason_codes,
                        "event_id": f"hub:query_state:{resolved_day_key}:{now_ms}",
                        "trace_id": f"query_state:{resolved_day_key}",
                        **self._memory_thermo_contract_fields(),
                    }
                }
            },
            "qualia": {
                "observations": {
                    "hub": {
                        "state_fingerprint": state_fingerprint,
                        "delegate_state_fingerprint": delegate_state_fingerprint,
                        "local_state_fingerprint": local_state_fingerprint,
                        "life_indicator_fingerprint": life_indicator_fingerprint,
                        "policy_prior_fingerprint": policy_prior_fingerprint,
                        "output_control_fingerprint": output_control_fingerprint,
                    }
                }
            },
            "invariants": {},
        }
        append_trace_event(target, event)

def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_day_key(value: Any) -> Optional[str]:
    return resolve_day_key_from_as_of(value)


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        excluded = {
            "timestamp",
            "timestamp_ms",
            "updated_at",
            "generated_at",
            "created_at",
        }
        normalized_items = []
        for k, v in value.items():
            key = str(k)
            if key in excluded:
                continue
            normalized_items.append((key, _normalize_for_hash(v)))
        return {k: v for k, v in sorted(normalized_items, key=lambda item: item[0])}
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, set):
        normalized = [_normalize_for_hash(v) for v in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
    return value


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _ensure_mapping(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _entry_value(entry: Any, key: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


def _moment_timestamp(entry: Any) -> datetime:
    stamp = _entry_value(entry, "timestamp")
    if isinstance(stamp, datetime):
        dt = stamp
    else:
        ts_ms = _entry_value(entry, "timestamp_ms")
        if ts_ms is not None:
            try:
                dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
        else:
            raw = _entry_value(entry, "ts")
            try:
                dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _derive_identifiers(entry: Any, fallback_session: str | None) -> tuple[str, str, int]:
    scenario = (
        _entry_value(entry, "session_id")
        or _entry_value(entry, "scenario_id")
        or fallback_session
        or "hub"
    )
    scenario = str(scenario)
    turn_raw = _entry_value(entry, "turn_id") or _entry_value(entry, "id")
    seed: int | None = None
    if isinstance(turn_raw, str) and turn_raw.strip():
        suffix = turn_raw.strip()
    elif isinstance(turn_raw, (int, float)):
        idx = int(turn_raw)
        suffix = f"turn-{idx:04d}"
        seed = idx or 1
    else:
        suffix = "turn-0000"
    turn_id = f"{scenario}-{suffix}"
    if seed is None:
        seed = abs(hash(turn_id)) % 1_000_000 + 1
    return scenario, turn_id, seed


def _build_trace_payload(
    entry: Any,
    *,
    fallback_session: str | None,
) -> tuple[dict[str, Any], datetime]:
    stamp = _moment_timestamp(entry)
    scenario_id, turn_id, seed = _derive_identifiers(entry, fallback_session)

    mood = _ensure_mapping(_entry_value(entry, "mood"))
    metrics = _ensure_mapping(_entry_value(entry, "metrics"))
    gate_context = _ensure_mapping(_entry_value(entry, "gate_context"))
    culture = _ensure_mapping(_entry_value(entry, "culture"))
    emotion = _ensure_mapping(_entry_value(entry, "emotion"))

    somatic = {
        "arousal_hint": _first_value(_maybe_float(mood.get("arousal")), _maybe_float(metrics.get("arousal_hint"))),
        "stress_hint": _first_value(
            _maybe_float(mood.get("stress")),
            _maybe_float(metrics.get("stress")),
            _maybe_float(emotion.get("stress")),
        ),
        "fatigue_hint": _first_value(
            _maybe_float(metrics.get("fatigue")),
            _maybe_float(metrics.get("sleep_debt")),
            _maybe_float(emotion.get("mask")),
        ),
        "jitter": _maybe_float(metrics.get("jitter")),
        "proximity": _first_value(
            _maybe_float(metrics.get("proximity")),
            _maybe_float(gate_context.get("proximity")),
            _maybe_float(metrics.get("distance")),
        ),
    }
    somatic = {k: v for k, v in somatic.items() if v is not None}

    sensor_metrics = {k: v for k, v in {**emotion, **metrics}.items() if v is not None}

    context_payload = dict(gate_context)
    mode = _first_value(context_payload.get("mode"), _entry_value(entry, "talk_mode"))
    if mode:
        context_payload["mode"] = mode
    disclosure = _first_value(
        context_payload.get("disclosure_budget"),
        context_payload.get("intimacy"),
        culture.get("intimacy"),
    )
    if disclosure is not None:
        context_payload["disclosure_budget"] = disclosure
    cultural_pressure = _first_value(context_payload.get("cultural_pressure"), culture.get("rho"))
    if cultural_pressure is not None:
        context_payload["cultural_pressure"] = cultural_pressure
    offer = _first_value(context_payload.get("offer_requested"), gate_context.get("request"))
    if offer is not None:
        context_payload["offer_requested"] = offer

    world_payload = {
        "hazard_level": gate_context.get("hazard_level"),
        "ambiguity": gate_context.get("ambiguity"),
        "clarity": gate_context.get("clarity"),
        "social_pressure": _first_value(gate_context.get("social_pressure"), gate_context.get("crowd_pressure")),
        "npc_affect": culture.get("npc_affect") or culture.get("npc_valence"),
    }
    world_payload = {k: v for k, v in world_payload.items() if v is not None}

    trace_observations = _ensure_mapping(_entry_value(entry, "trace_observations"))
    emotion_tag = _entry_value(entry, "emotion_tag")
    if emotion_tag:
        trace_observations.setdefault("self", {})["emotion_tag"] = emotion_tag
    if mode:
        trace_observations.setdefault("policy", {})["talk_mode"] = mode

    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "turn_id": turn_id,
        "timestamp_ms": int(stamp.timestamp() * 1000),
        "seed": seed,
        "user_text": _entry_value(entry, "user_text") or REDACTED_TEXT,
        "runtime_version": _entry_value(entry, "runtime_version") or DEFAULT_RUNTIME_VERSION,
        "idempotency_key": _entry_value(entry, "idempotency_key") or "",
        "somatic": somatic,
        "context": {k: v for k, v in context_payload.items() if v is not None},
        "world": world_payload,
    }
    if sensor_metrics:
        payload["sensor_metrics"] = sensor_metrics
    if trace_observations:
        payload["trace_observations"] = trace_observations
    tags = _entry_value(entry, "tags")
    if tags:
        if isinstance(tags, (list, tuple, set)):
            payload["tags"] = list(tags)
        else:
            payload["tags"] = [tags]
    return payload, stamp
