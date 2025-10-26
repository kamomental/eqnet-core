import json
from pathlib import Path
from typing import Any, Dict

from devlife.metrics.kpi import compute_all


def load_tracks(path: str = "logs/episodes.jsonl") -> list[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def main() -> None:
    tracks = load_tracks()
    if not tracks:
        print("No episodes logged yet.")
        return
    kpis = compute_all(tracks)
    for key, value in kpis.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
