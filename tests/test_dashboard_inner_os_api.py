from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from emot_terrain_lab.ops.dashboard import make_handler


def test_dashboard_inner_os_api_serves_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "inner_os_dashboard_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "inner_os_dashboard_snapshot/v1",
                "model": {"name": "qwen-3.5-instruct", "source": "live_list"},
                "same_turn": {
                    "protection_mode": "repair",
                    "commitment_target": "repair",
                    "body_homeostasis_state": "recovering",
                    "relational_continuity_state": "holding_thread",
                    "social_topology_state": "threaded_group",
                },
                "overnight": {"homeostasis_budget_focus": "recovering"},
                "carry_strengths": {"commitment": 0.31},
                "dominant_carry_channel": "commitment",
                "transfer": {"migration_active": False},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    handler_cls = make_handler("127.0.0.1", 8765, snapshot_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/inner-os", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["available"] is True
    assert payload["schema"] == "inner_os_dashboard_snapshot/v1"
    assert payload["same_turn"]["social_topology_state"] == "threaded_group"
    assert payload["model"]["source"] == "live_list"
