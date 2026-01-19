from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

from collector import Collector, CollectorConfig
from reducer import Reducer, WindowSpec
from emitter import AuditEmitter, AuditEmitterConfig, Emitter, EmitterConfig
from constants import WINDOWS_MS, DEFAULT_CALC_VERSION, CFG_HASH_TAG
from common import read_last_payload_hash, reason_code_hash


ALLOWED_INPUT_TYPES = {"state_snapshot", "value_tag", "edge_log"}


def parse_windows(arg: Optional[str]) -> List[str]:
    if not arg:
        return ["short", "mid", "long"]
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return [w for w in items if w in WINDOWS_MS]


def floor_to_window(ts_ms: int, window_ms: int) -> int:
    return (ts_ms // window_ms) * window_ms


def main() -> int:
    p = argparse.ArgumentParser(prog="eqnet-derived", add_help=True)

    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--mode", dest="mode", required=True, choices=["batch", "follow"])

    p.add_argument("--windows", dest="windows", required=False)
    p.add_argument("--calc-version", dest="calc_version", required=False, default=DEFAULT_CALC_VERSION)

    p.add_argument(
        "--state-join",
        dest="state_join",
        required=False,
        choices=["by_state_id", "nearest_time"],
        default="by_state_id",
    )
    p.add_argument("--max-lag-ms", dest="max_lag_ms", required=False, type=int)

    p.add_argument("--batch-end-ts", dest="batch_end_ts", required=False, type=int)
    p.add_argument("--batch-lookback-ms", dest="batch_lookback_ms", required=False, type=int)

    p.add_argument("--follow-interval-ms", dest="follow_interval_ms", required=False, type=int)

    p.add_argument(
        "--strict-integrity",
        dest="strict_integrity",
        required=False,
        type=lambda x: x.lower() == "true",
        default=True,
    )
    p.add_argument(
        "--emit-control-audit",
        dest="emit_control_audit",
        required=False,
        type=lambda x: x.lower() == "true",
        default=False,
    )
    p.add_argument("--audit-out", dest="audit_out", required=False)

    args = p.parse_args()

    windows = parse_windows(args.windows)
    if not windows:
        return 2

    collector = Collector(
        CollectorConfig(allowed_event_types=ALLOWED_INPUT_TYPES, strict_integrity=bool(args.strict_integrity))
    )
    calc_version_tag = f"{args.calc_version}+cfg.{CFG_HASH_TAG}"
    reducer = Reducer(
        calc_version=str(calc_version_tag),
        max_lag_ms=args.max_lag_ms,
        state_join=args.state_join,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    emitter = Emitter(
        EmitterConfig(
            schema_version="v1.0.0",
            tz="Asia/Tokyo",
            session_id="sess_local",
            run_id="run_local",
        )
    )
    prev_hash = read_last_payload_hash(args.out_path)
    emitter.set_prev_hash(prev_hash)

    audit_emitter = None
    audit_fp = None
    if args.emit_control_audit:
        if not args.audit_out:
            return 2
        os.makedirs(os.path.dirname(args.audit_out) or ".", exist_ok=True)
        audit_emitter = AuditEmitter(
            AuditEmitterConfig(
                schema_version="v1.0.0",
                tz="Asia/Tokyo",
                session_id="sess_local",
                run_id="run_local",
            )
        )
        audit_prev_hash = read_last_payload_hash(args.audit_out)
        audit_emitter.set_prev_hash(audit_prev_hash)
        audit_fp = open(args.audit_out, "a", encoding="utf-8")
        audit_emitter.emit_control_audit(
            audit_fp,
            ts_unix_ms=int(time.time() * 1000),
            action="observe",
            scope={"session_id": "sess_local", "run_id": "run_local"},
            reason_hash=reason_code_hash("OBSERVE_START"),
        )

    def run_batch(end_ts: int) -> None:
        state_events = []
        value_events = []
        edge_events = []

        with open(args.in_path, "r", encoding="utf-8") as f:
            for ev in collector.iter_events(f):
                if ev.event_type == "state_snapshot":
                    state_events.append(ev)
                elif ev.event_type == "value_tag":
                    value_events.append(ev)
                elif ev.event_type == "edge_log":
                    edge_events.append(ev)

        with open(args.out_path, "a", encoding="utf-8") as out:
            for w_label in windows:
                w_ms = WINDOWS_MS[w_label]
                w_end = floor_to_window(end_ts, w_ms)
                start = w_end - w_ms + 1
                end = w_end

                st = [e for e in state_events if start <= e.ts_unix_ms <= end]
                vt = [e for e in value_events if start <= e.ts_unix_ms <= end]
                ed = [e for e in edge_events if start <= e.ts_unix_ms <= end]

                payload_obj = reducer.reduce_window(WindowSpec(label=w_label, window_ms=w_ms), w_end, st, vt, ed)
                if payload_obj is None:
                    continue

                payload = {
                    "kind": "derived_metrics",
                    "window": {"window_ms": payload_obj.window_ms, "end_ts_unix_ms": payload_obj.end_ts_unix_ms},
                    "metrics": payload_obj.metrics,
                    "sources": payload_obj.sources,
                }
                emitter.emit_derived_metrics(out, window_label=w_label, payload=payload, end_ts_unix_ms=w_end)

    def run_follow() -> str:
        interval_ms = args.follow_interval_ms
        if interval_ms is None:
            interval_ms = WINDOWS_MS["short"]

        last_pos = 0
        buffer_events = []
        last_emit = {label: None for label in windows}

        while True:
            try:
                with open(args.in_path, "r", encoding="utf-8") as f:
                    f.seek(last_pos)
                    for line in f:
                        ev = collector.parse_line(line)
                        if ev is not None:
                            buffer_events.append(ev)
                    last_pos = f.tell()

                now_ms = int(time.time() * 1000)
                with open(args.out_path, "a", encoding="utf-8") as out:
                    for w_label in windows:
                        w_ms = WINDOWS_MS[w_label]
                        end_ts = floor_to_window(now_ms, w_ms)
                        if last_emit.get(w_label) == end_ts:
                            continue
                        start = end_ts - w_ms + 1

                        st = [e for e in buffer_events if e.event_type == "state_snapshot" and start <= e.ts_unix_ms <= end_ts]
                        vt = [e for e in buffer_events if e.event_type == "value_tag" and start <= e.ts_unix_ms <= end_ts]
                        ed = [e for e in buffer_events if e.event_type == "edge_log" and start <= e.ts_unix_ms <= end_ts]

                        payload_obj = reducer.reduce_window(WindowSpec(label=w_label, window_ms=w_ms), end_ts, st, vt, ed)
                        if payload_obj is None:
                            continue

                        payload = {
                            "kind": "derived_metrics",
                            "window": {"window_ms": payload_obj.window_ms, "end_ts_unix_ms": payload_obj.end_ts_unix_ms},
                            "metrics": payload_obj.metrics,
                            "sources": payload_obj.sources,
                        }
                        emitter.emit_derived_metrics(out, window_label=w_label, payload=payload, end_ts_unix_ms=end_ts)
                        last_emit[w_label] = end_ts

                cutoff = now_ms - WINDOWS_MS["long"]
                buffer_events[:] = [e for e in buffer_events if e.ts_unix_ms >= cutoff]

                time.sleep(interval_ms / 1000.0)

            except KeyboardInterrupt:
                return "STOP_INTERRUPT"

        return "STOP_NORMAL"

    stop_reason = "STOP_NORMAL"
    try:
        if args.mode == "batch":
            end_ts = args.batch_end_ts
            if end_ts is None:
                ts_list = []
                with open(args.in_path, "r", encoding="utf-8") as f:
                    for ev in collector.iter_events(f):
                        ts_list.append(ev.ts_unix_ms)
                end_ts = max(ts_list) if ts_list else int(time.time() * 1000)
            run_batch(int(end_ts))
            return 0

        if args.mode == "follow":
            stop_reason = run_follow()
            return 0

        return 2
    finally:
        if audit_emitter and audit_fp:
            audit_emitter.emit_control_audit(
                audit_fp,
                ts_unix_ms=int(time.time() * 1000),
                action="stop",
                scope={"session_id": "sess_local", "run_id": "run_local"},
                reason_hash=reason_code_hash(stop_reason),
            )
            audit_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
