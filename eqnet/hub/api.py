"""Core EQNet hub API (log_moment / run_nightly / query_state)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from eqnet.qualia_model import update_qualia_state
from eqnet.runtime.life_indicator import LifeIndicator
from eqnet.runtime.policy import PolicyPrior
from eqnet.runtime.state import QualiaState
from eqnet.persona.loader import PersonaConfig
from emot_terrain_lab.hub.qualia_logging import append_qualia_telemetry
from emot_terrain_lab.ops.nightly_life_indicator import (
    compute_life_indicator_for_day,
    load_qualia_log,
)


@dataclass
class EQNetConfig:
    telemetry_dir: Path = Path("telemetry")
    reports_dir: Path = Path("reports")
    state_dir: Path = Path("state")


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


