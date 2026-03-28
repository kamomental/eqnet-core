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
                    "temporal_membrane_mode": "coherent_reentry",
                    "temporal_reentry_pull": 0.18,
                    "contact_reflection_state": "guarded_reflection",
                    "contact_reflection_style": "reflect_only",
                    "contact_reflect_share": 0.41,
                    "boundary_gate_mode": "narrow",
                    "boundary_transform_mode": "soften",
                    "boundary_withheld_count": 1,
                    "residual_reflection_mode": "withheld",
                    "residual_reflection_strength": 0.44,
                    "body_homeostasis_state": "recovering",
                    "relational_continuity_state": "holding_thread",
                    "social_topology_state": "threaded_group",
                },
                "overnight": {
                    "homeostasis_budget_focus": "recovering",
                    "temporal_membrane_focus": "same_group_reentry",
                    "temporal_reentry_bias": 0.2,
                },
                "carry_strengths": {"commitment": 0.31},
                "dominant_carry_channel": "commitment",
                "transfer": {
                    "migration_active": False,
                    "temporal_membrane_focus": "same_group_reentry",
                    "temporal_reentry_bias": 0.2,
                },
                "temporal_alignment": {
                    "focus_alignment": True,
                    "same_to_overnight_reentry_delta": 0.02,
                    "overnight_to_transfer_reentry_delta": 0.0,
                    "reentry_carry_visible": True,
                    "reentry_carry_strength": 0.27,
                },
                "boundary_alignment": {
                    "gate_mode": "narrow",
                    "transform_mode": "soften",
                    "softened_count": 0,
                    "withheld_count": 1,
                    "deferred_count": 0,
                    "residual_pressure": 0.31,
                    "residual_mode": "withheld",
                    "residual_focus": "unfinished part",
                    "residual_strength": 0.44,
                    "unsaid_pressure_visible": True,
                },
                "contact_alignment": {
                    "state": "guarded_reflection",
                    "style": "reflect_only",
                    "transmit_share": 0.33,
                    "reflect_share": 0.41,
                    "absorb_share": 0.26,
                    "block_share": 0.0,
                    "reflection_visible": True,
                },
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
    assert payload["same_turn"]["temporal_membrane_mode"] == "coherent_reentry"
    assert payload["overnight"]["temporal_membrane_focus"] == "same_group_reentry"
    assert payload["transfer"]["temporal_reentry_bias"] == 0.2
    assert payload["temporal_alignment"]["focus_alignment"] is True
    assert payload["temporal_alignment"]["reentry_carry_strength"] == 0.27
    assert payload["same_turn"]["contact_reflection_state"] == "guarded_reflection"
    assert payload["same_turn"]["boundary_gate_mode"] == "narrow"
    assert payload["same_turn"]["residual_reflection_mode"] == "withheld"
    assert payload["boundary_alignment"]["withheld_count"] == 1
    assert payload["boundary_alignment"]["unsaid_pressure_visible"] is True
    assert payload["contact_alignment"]["style"] == "reflect_only"
    assert payload["contact_alignment"]["reflection_visible"] is True
    assert payload["model"]["source"] == "live_list"
