"""Agent-to-Agent routing with contract guardrails and JSONL auditing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _parse_expires(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("expires_at must be ISO8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _hash_digest(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class ContractSpec:
    session_id: str
    version: str
    intent: str
    from_agent: str
    to_agent: str
    scopes: List[str]
    guardrails: Dict[str, Any]
    expires_at: Optional[dt.datetime]
    cost_budget: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_steps(self) -> int:
        return int(self.guardrails.get("max_steps", 12))


@dataclass
class TurnRecord:
    turn_index: int
    role: str
    actor: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class ScoreRecord:
    turn_index: int
    candidate_id: str
    actor: str
    scores: Dict[str, float]
    objective: Optional[str] = None
    objective_value: Optional[float] = None
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class A2ASession:
    spec: ContractSpec
    status: str = "open"
    turns: List[TurnRecord] = field(default_factory=list)
    scores: List[ScoreRecord] = field(default_factory=list)
    audit_path: Optional[Path] = None
    idempotency: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=lambda: {"turn": {}, "score": {}, "write": {}}
    )
    best_objective: Optional[float] = None

    @property
    def remaining_steps(self) -> int:
        return max(self.spec.max_steps - len(self.turns), 0)


class A2ARouter:
    """Stateful supervisor for A2A conversations."""

    REQUIRED_GUARDRAIL_KEYS = {"max_steps", "tool_timeout_s", "no_recursive"}

    def __init__(self, *, log_dir: str = "logs/a2a", auto_mkdir: bool = True) -> None:
        self._sessions: Dict[str, A2ASession] = {}
        self.log_dir = Path(log_dir)
        if auto_mkdir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- utils
    def _log(self, session: A2ASession, event: Mapping[str, Any]) -> None:
        if session.audit_path is None:
            session.audit_path = self.log_dir / f"{session.spec.session_id}.jsonl"
        payload = {
            "ts": time.time(),
            "session_id": session.spec.session_id,
            **event,
        }
        if "payload" in payload and isinstance(payload["payload"], Mapping):
            payload["result_digest"] = _hash_digest(payload["payload"])
        session.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with session.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_session(self, session_id: str) -> A2ASession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id={session_id}")
        if session.spec.expires_at and dt.datetime.now(dt.timezone.utc) >= session.spec.expires_at:
            session.status = "expired"
            self._log(session, {"event": "session.closed", "reason": "expired"})
            raise RuntimeError("session expired")
        if session.status != "open":
            raise RuntimeError(f"session {session_id} is {session.status}")
        return session

    def _check_idempotency(
        self, session: A2ASession, scope: str, payload: Mapping[str, Any], response: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        key = payload.get("idempotency_key")
        if not key:
            return None
        bucket = session.idempotency.setdefault(scope, {})
        if key in bucket:
            return bucket[key]
        if response is not None:
            bucket[key] = response
        return None

    def _normalize_tool_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    def _resolve_tool_policy(
        self,
        *,
        session: A2ASession,
        metadata: Mapping[str, Any],
    ) -> Dict[str, List[str]]:
        disallow: List[str] = []
        allow: List[str] = []
        spec_meta = session.spec.metadata if isinstance(session.spec.metadata, Mapping) else {}
        spec_tool_policy = spec_meta.get("tool_policy") if isinstance(spec_meta, Mapping) else None
        if isinstance(spec_tool_policy, Mapping):
            disallow = self._normalize_tool_list(spec_tool_policy.get("disallow_tools"))
            allow = self._normalize_tool_list(spec_tool_policy.get("allow_tools"))
        disallow = disallow + self._normalize_tool_list(metadata.get("disallow_tools"))
        allow = allow + self._normalize_tool_list(metadata.get("allow_tools"))
        # Safety precedence: disallow wins over allow.
        disallow_unique = list(dict.fromkeys(disallow))
        allow_unique = [name for name in list(dict.fromkeys(allow)) if name not in set(disallow_unique)]
        return {"disallow_tools": disallow_unique, "allow_tools": allow_unique}

    def _extract_tool_name(self, turn_payload: Mapping[str, Any]) -> Optional[str]:
        direct = turn_payload.get("tool_name")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        tool = turn_payload.get("tool")
        if isinstance(tool, Mapping):
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    def _emit_trace_v1_event(
        self,
        *,
        session_id: str,
        event_type: str,
        tool_name: str | None = None,
        reason_codes: List[str] | None = None,
        online_delta_ids: List[str] | None = None,
    ) -> None:
        if str(os.getenv("EQNET_TRACE_V1") or "").strip().lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path
            from eqnet.telemetry.trace_writer import append_trace_event
        except Exception:
            return
        now_ms = int(time.time() * 1000)
        trace_root = Path(os.getenv("EQNET_TRACE_V1_DIR") or "telemetry/trace_v1")
        target = trace_output_path(
            TracePathConfig(base_dir=trace_root, source_loop="a2a_router"),
            timestamp_ms=now_ms,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized_reasons = [str(code) for code in (reason_codes or []) if isinstance(code, str) and code]
        normalized_delta_ids = [str(item) for item in (online_delta_ids or []) if isinstance(item, str) and item]
        payload: Dict[str, Any] = {
            "schema_version": "trace_v1",
            "source_loop": "a2a_router",
            "scenario_id": "tool_policy",
            "turn_id": f"{session_id}-{event_type}-{now_ms}",
            "seed": abs(hash(f"{session_id}|{event_type}|{now_ms}")) % 1_000_000 + 1,
            "timestamp_ms": now_ms,
            "event_type": event_type,
            "tool_name": str(tool_name or ""),
            "reason_codes": normalized_reasons,
            "boundary": {},
            "self": {},
            "prospection": {"accepted": False},
            "policy": {
                "observations": {
                    "hub": {
                        "operation": "a2a_router",
                        "online_delta_applied": True,
                        "online_delta_ids": normalized_delta_ids,
                        "online_delta_action_types": ["DISALLOW_TOOL"] if event_type == "tool_call_blocked" else [],
                    }
                }
            },
            "qualia": {},
            "invariants": {},
        }
        append_trace_event(target, payload)

    # -------------------------------------------------------------------- public
    def capabilities(self) -> Dict[str, Any]:
        """Return a light-weight capability manifest for discovery endpoints."""
        return {
            "agent_id": "eqnet",
            "version": "2025.10",
            "roles": ["planner", "critic", "vision-ingest"],
            "resources": [
                {"name": "resonance:summary", "format": "json"},
                {"name": "vision:snapshot", "format": "json"},
                {"name": "culture:feedback", "format": "json"},
            ],
            "tools": [
                {"name": "telemetry:vision.push", "io": "json"},
                {"name": "a2a:contract.open", "io": "json"},
                {"name": "a2a:turn.post", "io": "json"},
                {"name": "a2a:score.report", "io": "json"},
            ],
        }

    def open_contract(self, spec_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Register a session if guardrails are satisfied."""
        session_id = str(spec_payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        if session_id in self._sessions:
            raise ValueError(f"session already exists: {session_id}")

        version = str(spec_payload.get("version") or "").strip()
        if not version:
            raise ValueError("version is required")
        scopes = [str(s).strip() for s in (spec_payload.get("scopes") or []) if s]
        if not scopes:
            raise ValueError("scopes must be provided")
        guardrails = dict(spec_payload.get("guardrails") or {})
        missing = self.REQUIRED_GUARDRAIL_KEYS - guardrails.keys()
        if missing:
            raise ValueError(f"guardrails missing keys: {', '.join(sorted(missing))}")
        guardrails["max_steps"] = int(guardrails["max_steps"])
        if guardrails["max_steps"] <= 0:
            raise ValueError("guardrails.max_steps must be positive")
        guardrails["tool_timeout_s"] = float(guardrails["tool_timeout_s"])
        guardrails["no_recursive"] = bool(guardrails["no_recursive"])
        if "objective_min_delta" in guardrails:
            guardrails["objective_min_delta"] = float(guardrails["objective_min_delta"])
        expires_at = _parse_expires(spec_payload.get("expires_at"))
        spec = ContractSpec(
            session_id=session_id,
            version=version,
            intent=str(spec_payload.get("intent") or ""),
            from_agent=str(spec_payload.get("from") or ""),
            to_agent=str(spec_payload.get("to") or ""),
            scopes=scopes,
            guardrails=guardrails,
            expires_at=expires_at,
            cost_budget=float(spec_payload.get("cost_budget_usd") or spec_payload.get("cost_budget") or 0.0),
            metadata=dict(spec_payload.get("metadata") or {}),
        )
        if not spec.intent or not spec.from_agent or not spec.to_agent:
            raise ValueError("intent/from/to are required")

        session = A2ASession(spec=spec)
        self._sessions[spec.session_id] = session
        self._log(
            session,
            {
                "event": "contract.open",
                "actor": spec.from_agent,
                "scopes": scopes,
                "guardrails": guardrails,
                "cost_budget": spec.cost_budget,
                "expires_at": spec.expires_at.isoformat() if spec.expires_at else None,
            },
        )
        limits = {
            "max_steps": spec.max_steps,
            "tool_timeout_s": guardrails["tool_timeout_s"],
        }
        return {
            "accepted": True,
            "session_id": spec.session_id,
            "scopes": scopes,
            "limits": limits,
            "expires_at": spec.expires_at.isoformat() if spec.expires_at else None,
        }

    def append_turn(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        session = self._ensure_session(str(payload.get("session_id")))
        existing = self._check_idempotency(session, "turn", payload)
        if existing is not None:
            return existing
        if session.remaining_steps <= 0:
            session.status = "max_steps"
            self._log(session, {"event": "session.closed", "reason": "max_steps"})
            raise RuntimeError("max steps reached")
        role = str(payload.get("role") or "")
        if role not in {"planner", "actor", "critic", "delegate"}:
            raise ValueError("role must be planner/actor/critic/delegate")
        actor = str(payload.get("actor") or session.spec.from_agent)
        metadata = dict(payload.get("metadata") or {})
        tool_policy = self._resolve_tool_policy(session=session, metadata=metadata)
        turn_payload = dict(payload.get("payload") or {})
        tool_name = self._extract_tool_name(turn_payload)
        disallow_tools = set(tool_policy.get("disallow_tools") or [])
        if isinstance(tool_name, str) and tool_name in disallow_tools:
            online_delta_ids = self._normalize_tool_list(metadata.get("online_delta_ids"))
            event_payload = {
                "event": "tool_call_blocked",
                "turn_index": len(session.turns),
                "actor": actor,
                "role": role,
                "tool_name": tool_name,
                "reason_codes": ["ONLINE_DELTA_TOOL_BLOCKED"],
                "disallow_tools": sorted(disallow_tools),
            }
            self._log(session, event_payload)
            self._emit_trace_v1_event(
                session_id=session.spec.session_id,
                event_type="tool_call_blocked",
                tool_name=tool_name,
                reason_codes=["ONLINE_DELTA_TOOL_BLOCKED"],
                online_delta_ids=online_delta_ids,
            )
            response = {
                "status": "blocked",
                "blocked": True,
                "turn_index": len(session.turns),
                "remaining_steps": session.remaining_steps,
                "reason_codes": ["ONLINE_DELTA_TOOL_BLOCKED"],
                "tool_name": tool_name,
                "audit_path": str(session.audit_path) if session.audit_path else None,
            }
            self._check_idempotency(session, "turn", payload, response)
            return response
        if session.spec.guardrails.get("no_recursive"):
            caller = metadata.get("source_agent")
            callee = metadata.get("target_agent")
            if caller == session.spec.to_agent and callee == session.spec.from_agent:
                session.status = "recursive_blocked"
                self._log(session, {"event": "session.closed", "reason": "no_recursive_violation", "actor": actor})
                raise RuntimeError("no_recursive guardrail violated")
        turn_index = len(session.turns)
        record = TurnRecord(turn_index=turn_index, role=role, actor=actor, payload=turn_payload)
        session.turns.append(record)
        self._log(
            session,
            {
                "event": "turn",
                "turn_index": turn_index,
                "actor": actor,
                "role": role,
                "payload": turn_payload,
                "metadata": metadata,
            },
        )
        response = {
            "status": "ok",
            "turn_index": turn_index,
            "remaining_steps": session.remaining_steps,
            "audit_path": str(session.audit_path) if session.audit_path else None,
        }
        self._check_idempotency(session, "turn", payload, response)
        return response

    def record_score(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        session = self._ensure_session(str(payload.get("session_id")))
        existing = self._check_idempotency(session, "score", payload)
        if existing is not None:
            return existing
        turn_index = int(payload.get("turn_index", len(session.turns) - 1))
        candidate_id = str(payload.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate_id is required")
        scores = dict(payload.get("scores") or {})
        actor = str(payload.get("actor") or session.spec.to_agent)
        objective = payload.get("objective")
        objective_value = payload.get("objective_value")
        if objective_value is not None:
            try:
                objective_value = float(objective_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("objective_value must be numeric") from exc
            min_delta = session.spec.guardrails.get("objective_min_delta")
            if min_delta is not None:
                prev = session.best_objective
                if prev is not None and objective_value - prev < float(min_delta):
                    session.status = "objective_plateau"
                    self._log(
                        session,
                        {
                            "event": "session.closed",
                            "reason": "objective_plateau",
                            "actor": actor,
                            "objective_value": objective_value,
                        },
                    )
                    raise RuntimeError("objective improvement below guardrail")
            if session.best_objective is None or objective_value > session.best_objective:
                session.best_objective = objective_value
        record = ScoreRecord(
            turn_index=turn_index,
            candidate_id=candidate_id,
            actor=actor,
            scores=scores,
            objective=objective,
            objective_value=objective_value,
        )
        session.scores.append(record)
        self._log(
            session,
            {
                "event": "score",
                "turn_index": turn_index,
                "actor": actor,
                "candidate_id": candidate_id,
                "scores": scores,
                "objective": objective,
                "objective_value": objective_value,
            },
        )
        response = {"status": "ok"}
        self._check_idempotency(session, "score", payload, response)
        return response

    def close(self, session_id: str, *, reason: str = "completed") -> Dict[str, Any]:
        session = self._ensure_session(session_id)
        session.status = reason
        self._log(session, {"event": "session.closed", "reason": reason, "actor": session.spec.from_agent})
        return {"status": "closed", "session_id": session_id, "reason": reason}

    def session_snapshot(self, session_id: str) -> Dict[str, Any]:
        session = self._ensure_session(session_id)
        return {
            "session_id": session_id,
            "spec": {
                "version": session.spec.version,
                "intent": session.spec.intent,
                "from": session.spec.from_agent,
                "to": session.spec.to_agent,
                "scopes": session.spec.scopes,
                "guardrails": session.spec.guardrails,
                "expires_at": session.spec.expires_at.isoformat() if session.spec.expires_at else None,
                "cost_budget": session.spec.cost_budget,
                "metadata": session.spec.metadata,
            },
            "status": session.status,
            "turns": [
                {
                    "turn_index": record.turn_index,
                    "role": record.role,
                    "actor": record.actor,
                    "payload": record.payload,
                    "ts": record.ts,
                }
                for record in session.turns
            ],
            "scores": [
                {
                    "turn_index": record.turn_index,
                    "candidate_id": record.candidate_id,
                    "actor": record.actor,
                    "scores": record.scores,
                    "objective": record.objective,
                    "objective_value": record.objective_value,
                    "ts": record.ts,
                }
                for record in session.scores
            ],
            "audit_path": str(session.audit_path) if session.audit_path else None,
            "remaining_steps": session.remaining_steps,
            "best_objective": session.best_objective,
        }


__all__ = ["A2ARouter", "ContractSpec"]
