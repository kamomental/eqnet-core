from __future__ import annotations

import json
from pathlib import Path

from eqnet.telemetry.mecpe_writer import MecpeWriter, MecpeWriterConfig


def test_mecpe_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = MecpeWriter(MecpeWriterConfig(telemetry_dir=tmp_path))
    out = writer.append_turn(
        timestamp_ms=1735689600000,
        turn_id="turn-1",
        prompt_hash="a" * 64,
        model_version="mecpe-dummy-v0",
        text_hash="b" * 64,
        audio_sha256="",
        video_sha256="",
    )
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "schema_version",
        "prompt_hash",
        "model",
        "text_hash",
        "audio_sha256",
        "video_sha256",
        "turn_id",
        "timestamp_ms",
    ]:
        assert key in payload
    assert payload["schema_version"] == "mecpe_record.v0"
    assert payload["model"]["version"] == "mecpe-dummy-v0"

