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
from eqnet.hub.text_policy import apply_text_policy
from eqnet.hub.idempotency import IdempotencyStore, NoopIdempotencyStore
from eqnet.hub.runtime_contract import HubRuntime
from eqnet.runtime.external_runtime import (
    ExternalRuntimeDelegate,
    ExternalRuntimeDelegateV2,
)
from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path
from eqnet.telemetry.trace_writer import append_trace_event
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit
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
    audit_thresholds: Dict[str, Any] | None = None


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
        self._trace_safety = SafetyConfig()
        self._runtime_delegate = self._resolve_runtime_delegate(runtime_delegate)
        self._idempotency_store = idempotency_store or NoopIdempotencyStore()
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
        input_fingerprint = self._fingerprint_moment_input(raw_event, raw_text)
        if not reserved:
            self._emit_trace_v1(
                moment_entry,
                raw_text,
                runtime_version=self._runtime_version,
                idempotency_key=idem_key,
                idempotency_status="skipped",
                delegate_to=delegate_name,
                delegation_mode=mode,
                mismatch_policy=mismatch_policy,
                mismatch_reason_codes=[],
                moment_input_fingerprint=input_fingerprint,
                qualia_state_fingerprint=None,
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
                audit_fingerprint=None,
            )
            return

        idem_status = "reserved"
        delegate_status = "not_called"
        mismatch_reason_codes: list[str] = []
        life_indicator_fingerprint: Optional[str] = None
        policy_prior_fingerprint: Optional[str] = None
        audit_fingerprint: Optional[str] = None
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
            audit_fingerprint = current["audit_fingerprint"]

            idem_status = "done"
            result_fingerprint = self._fingerprint_json_payload(
                {
                    "day": date_obj.isoformat(),
                    "life_indicator": life_indicator_fingerprint,
                    "policy_prior": policy_prior_fingerprint,
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
                audit_fingerprint=audit_fingerprint,
            )

    # ------------------------------------------------------------------
    # 3) query_state: UI/Persona が "今" と "今日" を見る窓
    # ------------------------------------------------------------------

    def query_state(
        self,
        *,
        as_of: Optional[str] = None,
    ) -> dict:
        normalized_as_of = _normalize_day_key(as_of) or as_of
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
        delegate_resolved = _normalize_day_key(result_state.get("resolved_day_key"))
        if delegate_resolved is not None:
            resolved_day_key = delegate_resolved
        elif normalized_as_of is not None:
            resolved_day_key = _normalize_day_key(normalized_as_of)
            mismatch_reason_codes.append("missing_resolved_day_key")
        else:
            mismatch_reason_codes.append("missing_resolved_day_key")
            resolved_day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if (
            normalized_as_of is not None
            and resolved_day_key is not None
            and _normalize_day_key(normalized_as_of) != resolved_day_key
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


    def _emit_trace_v1(
        self,
        moment_entry: Any,
        raw_text: str,
        *,
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
            policy_obs.update(
                {
                    "delegate_to": delegate_to,
                    "delegation_mode": delegation_mode,
                    "delegate_status": delegate_status,
                    "idempotency_status": idempotency_status,
                    "mismatch_policy": mismatch_policy,
                    "mismatch_reason_codes": list(mismatch_reason_codes or []),
                }
            )
            qualia_obs = thin_entry.setdefault("trace_observations", {}).setdefault("qualia", {})
            qualia_obs.update(
                {
                    "moment_input_fingerprint": moment_input_fingerprint,
                    "qualia_state_fingerprint": qualia_state_fingerprint,
                }
            )

            payload, _stamp = _build_trace_payload(
                thin_entry,
                fallback_session=getattr(self.persona, "persona_id", None),
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

    def _get_latest_state(self) -> Dict[str, Any]:
        return {
            "qualia": self._latest_qualia_state,
            "life_indicator": self._latest_life_indicator,
            "policy_prior": self._latest_policy_prior,
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

    def _snapshot_nightly_fingerprints(self, day: date) -> dict[str, Optional[str]]:
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
        audit_fingerprint = self._load_audit_fingerprint(day)
        return {
            "life_indicator_fingerprint": life_indicator_fingerprint,
            "policy_prior_fingerprint": policy_prior_fingerprint,
            "audit_fingerprint": audit_fingerprint,
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
        audit_fingerprint: Optional[str],
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
        event = {
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
                        "delegation_mode": delegation_mode,
                        "delegate_to": delegate_to,
                        "delegate_status": delegate_status,
                        "idempotency_status": idempotency_status,
                        "mismatch_policy": mismatch_policy,
                        "mismatch_reason_codes": mismatch_reason_codes,
                    }
                }
            },
            "qualia": {
                "observations": {
                    "hub": {
                        "life_indicator_fingerprint": life_indicator_fingerprint,
                        "policy_prior_fingerprint": policy_prior_fingerprint,
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
    ) -> None:
        if not _env_truthy(TRACE_FLAG_ENV):
            return
        normalized = _normalize_day_key(resolved_day_key)
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
                        "delegation_mode": delegation_mode,
                        "configured_delegation_mode": delegation_mode,
                        "delegate_to": delegate_to,
                        "delegate_status": delegate_status,
                        "mismatch_policy": mismatch_policy,
                        "mismatch_reason_codes": mismatch_reason_codes,
                    }
                }
            },
            "qualia": {
                "observations": {
                    "hub": {
                        "state_fingerprint": state_fingerprint,
                        "delegate_state_fingerprint": delegate_state_fingerprint,
                        "local_state_fingerprint": local_state_fingerprint,
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
