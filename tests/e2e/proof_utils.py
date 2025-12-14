from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

SCENARIO_PATH = Path("fixtures/scenarios/e2e_proof_001.json")


@dataclass
class ProofScenario:
    scenario_id: str
    date: date
    events: List[dict[str, Any]]


def load_proof_scenario(path: Path = SCENARIO_PATH) -> ProofScenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario_date = date.fromisoformat(data["date"])
    scenario_id = data["scenario_id"]
    events = list(data.get("events", []))
    return ProofScenario(scenario_id=scenario_id, date=scenario_date, events=events)


def prepare_event(raw_event: dict[str, Any], idx: int, scenario_id: str, scenario_date: date) -> dict[str, Any]:
    event = json.loads(json.dumps(raw_event))
    event.setdefault("scenario_id", scenario_id)
    base_ts = datetime.combine(scenario_date, datetime.min.time(), tzinfo=timezone.utc)
    event["timestamp_ms"] = int((base_ts + timedelta(seconds=(idx + 1) * 60)).timestamp() * 1000)
    event.setdefault("turn_id", f"{scenario_id}-turn-{idx + 1:04d}")
    event.setdefault("user_text", "<redacted>")
    return event
