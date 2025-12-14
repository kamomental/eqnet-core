"""Core EQNet hub API (log_moment / run_nightly / query_state)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from eqnet.qualia_model import update_qualia_state
from eqnet.runtime.life_indicator import LifeIndicator
from eqnet.runtime.policy import PolicyPrior
from eqnet.runtime.state import QualiaState
from eqnet.runtime.turn import CoreState, SafetyConfig
from eqnet.orchestrators.hub_adapter import run_hub_turn
from eqnet.hub.moment_entry import to_moment_entry
from eqnet.hub.text_policy import apply_text_policy
from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit
from eqnet.persona.loader import PersonaConfig
from emot_terrain_lab.hub.qualia_logging import append_qualia_telemetry
from emot_terrain_lab.ops.nightly_life_indicator import (
    compute_life_indicator_for_day,
    load_qualia_log,
)


logger = logging.getLogger(__name__)

TRACE_FLAG_ENV = "EQNET_TRACE_V1"
TRACE_DIR_ENV = "EQNET_TRACE_V1_DIR"
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
        if persona is not None:
            self._apply_persona_defaults(persona)

    # ------------------------------------------------------------------
    # 1) log_moment: raw_event/raw_text を心の入口へ流す
    # ------------------------------------------------------------------

    def log_moment(self, raw_event: Any, raw_text: str) -> None:
        moment_entry = self._to_moment_entry(raw_event, raw_text)
        text_emb = self.embed_text_fn(raw_text)
        qstate = update_qualia_state(prev_state=None, moment_entry=moment_entry, text_embedding=text_emb)
        self._latest_qualia_state = qstate
        append_qualia_telemetry(self.config.telemetry_dir, qstate)
        self._append_moment_log(moment_entry)
        self._emit_trace_v1(moment_entry, raw_text)

    # ------------------------------------------------------------------
    # 2) run_nightly: 1 日分の danger/healing/life_indicator を更新
    # ------------------------------------------------------------------

    def run_nightly(self, date_obj: Optional[date] = None) -> None:
        date_obj = date_obj or date.today()
        date_str = date_obj.strftime("%Y%m%d")
        qualia_path = self.config.telemetry_dir / f"qualia-{date_str}.jsonl"
        qualia_records = load_qualia_log(qualia_path)

        num_diary_entries = self._count_diary_entries(date_obj)
        num_self_reflection_entries = self._count_self_reflection_entries(date_obj)

        life_indicator = compute_life_indicator_for_day(
            qualia_records,
            num_diary_entries=num_diary_entries,
            num_self_reflection_entries=num_self_reflection_entries,
        )
        self._latest_life_indicator = life_indicator
        policy_prior = self._run_danger_healing_and_policy_updates(date_obj, qualia_records, life_indicator)
        self._latest_policy_prior = policy_prior

        self._save_life_indicator(date_obj, life_indicator)
        self._save_policy_prior(policy_prior)
        self._write_nightly_report(date_obj, life_indicator, policy_prior)
        self._run_nightly_audit(date_obj)

    # ------------------------------------------------------------------
    # 3) query_state: UI/Persona が "今" と "今日" を見る窓
    # ------------------------------------------------------------------

    def query_state(self) -> dict:
        latest_q = self._latest_qualia_state
        latest_li = self._latest_life_indicator or self._load_latest_life_indicator()
        latest_pp = self._latest_policy_prior or self._load_latest_policy_prior()

        state: Dict[str, Any] = {
            "latest_qualia": None,
            "life_indicator": None,
            "policy_prior": None,
            "danger": self._load_recent_danger_metrics(),
            "healing": self._load_recent_healing_metrics(),
        }
        if self.persona is not None:
            state["persona"] = {
                "id": self.persona.persona_id,
                "display_name": self.persona.display_name,
                "meta": self.persona.meta,
            }
        if latest_q:
            vec = latest_q.qualia_vec
            dim = int(vec.shape[0]) if hasattr(vec, "shape") else len(vec)
            state["latest_qualia"] = {
                "timestamp": latest_q.timestamp.isoformat(),
                "dimension": dim,
                "qualia_vec": vec.tolist(),
            }
        if latest_li:
            state["life_indicator"] = {
                "identity": latest_li.identity_score,
                "qualia": latest_li.qualia_score,
                "meta_awareness": latest_li.meta_awareness_score,
            }
        if latest_pp:
            state["policy_prior"] = latest_pp.__dict__
        return state

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


    def _emit_trace_v1(self, moment_entry: Any, raw_text: str) -> None:
        if not _env_truthy(TRACE_FLAG_ENV):
            return
        try:
            thin_entry = to_moment_entry(moment_entry)
            if "user_text" not in thin_entry:
                thin_entry["user_text"] = raw_text
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


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
