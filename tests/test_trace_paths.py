from pathlib import Path

from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path


def test_trace_output_path_partition(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("os.getpid", lambda: 111)
    cfg = TracePathConfig(base_dir=tmp_path, source_loop="hub", run_id="runA")
    path = trace_output_path(cfg, timestamp_ms=0)
    assert path.parent.name.count("-") == 2  # YYYY-MM-DD
    assert path.name == "hub-runA-111.jsonl"
