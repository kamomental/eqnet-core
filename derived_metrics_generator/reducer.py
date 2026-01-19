from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from common import Event, DerivedMetricsPayload, clamp
from constants import EPS, METRIC_WINDOWS


@dataclass(frozen=True)
class WindowSpec:
    label: str
    window_ms: int


class Reducer:
    """Compute derived metrics per window from event streams."""

    def __init__(
        self, *, calc_version: str, max_lag_ms: Optional[int] = None, state_join: str = "by_state_id"
    ) -> None:
        self._calc_version = calc_version
        self._max_lag_ms = max_lag_ms
        self._state_join = state_join

    @staticmethod
    def _l2_norm(vec: List[float]) -> float:
        return math.sqrt(sum((float(x) * float(x) for x in vec)))

    @staticmethod
    def _variance(xs: List[float]) -> Optional[float]:
        n = len(xs)
        if n <= 1:
            return None
        mean = sum(xs) / n
        return sum((x - mean) ** 2 for x in xs) / (n - 1)

    @staticmethod
    def _cov(xs: List[float], ys: List[float]) -> Optional[float]:
        n = len(xs)
        if n <= 1 or n != len(ys):
            return None
        mx = sum(xs) / n
        my = sum(ys) / n
        return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)

    @staticmethod
    def _corr(xs: List[float], ys: List[float]) -> Optional[float]:
        cov = Reducer._cov(xs, ys)
        if cov is None:
            return None
        vx = Reducer._variance(xs)
        vy = Reducer._variance(ys)
        if vx is None or vy is None or vx <= 0 or vy <= 0:
            return None
        return cov / math.sqrt(vx * vy)

    def _extract_value_deltas(self, value_events: List[Event]) -> List[Tuple[int, str, float]]:
        out: List[Tuple[int, str, float]] = []
        for ev in value_events:
            pv = ev.payload or {}
            delta_vec = pv.get("delta_vec")
            state_id = pv.get("state_id")
            if not isinstance(delta_vec, list) or not isinstance(state_id, str):
                continue
            try:
                s = self._l2_norm([float(x) for x in delta_vec])
            except Exception:
                continue
            out.append((ev.ts_unix_ms, state_id, s))
        return out

    def _extract_state_pe(self, state_events: List[Event]) -> Dict[str, Tuple[int, float]]:
        out: Dict[str, Tuple[int, float]] = {}
        for ev in state_events:
            pv = ev.payload or {}
            state_id = pv.get("state_id")
            features = pv.get("features") or {}
            pe = features.get("pe")
            if not isinstance(state_id, str):
                continue
            try:
                pe_f = float(pe)
            except Exception:
                continue
            prev = out.get(state_id)
            if prev is None or ev.ts_unix_ms >= prev[0]:
                out[state_id] = (ev.ts_unix_ms, pe_f)
        return out

    def _extract_edge_delta_series(
        self, edge_events: List[Event]
    ) -> List[Tuple[int, float, Optional[bool]]]:
        out: List[Tuple[int, float, Optional[bool]]] = []
        for ev in edge_events:
            pv = ev.payload or {}
            dw = pv.get("delta_w")
            ctx = pv.get("context") or {}
            gating = ctx.get("gating") or {}
            replay_flag = gating.get("replay_flag")
            try:
                dw_f = float(dw)
            except Exception:
                continue
            rf: Optional[bool] = replay_flag if isinstance(replay_flag, bool) else None
            out.append((ev.ts_unix_ms, dw_f, rf))
        return out

    def reduce_window(
        self,
        window: WindowSpec,
        end_ts_unix_ms: int,
        state_events: List[Event],
        value_events: List[Event],
        edge_events: List[Event],
    ) -> Optional[DerivedMetricsPayload]:
        metrics: Dict[str, Optional[float]] = {
            "var_d_value": None,
            "edge_autocorr_tau": None,
            "pe_to_dvalue_coupling": None,
            "replay_rate": None,
        }

        state_ids: List[str] = []
        event_hashes: List[str] = []

        for ev in (state_events + value_events + edge_events):
            integrity = (ev.trace or {}).get("integrity") or {}
            ph = integrity.get("payload_hash")
            if isinstance(ph, str) and ph:
                event_hashes.append(ph)

        if window.label in METRIC_WINDOWS["var_d_value"]:
            deltas = self._extract_value_deltas(value_events)
            s_list = [s for _, _, s in deltas]
            metrics["var_d_value"] = self._variance(s_list)

        edge_series = self._extract_edge_delta_series(edge_events)
        if edge_series:
            edge_series.sort(key=lambda x: x[0])
            xs = [dw for _, dw, _ in edge_series]
            ts = [t for t, _, _ in edge_series]

            dt_ms = None
            if len(ts) >= 2:
                dts = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if ts[i] > ts[i - 1]]
                if dts:
                    dts.sort()
                    dt_ms = float(dts[len(dts) // 2])

            if window.label in METRIC_WINDOWS["edge_autocorr_tau"] and dt_ms and len(xs) >= 2:
                x0 = xs[:-1]
                x1 = xs[1:]
                rho = self._corr(x0, x1)
                if rho is not None:
                    rho_c = clamp(rho, EPS, 1.0 - EPS)
                    metrics["edge_autocorr_tau"] = -dt_ms / math.log(rho_c)

            if window.label in METRIC_WINDOWS["replay_rate"]:
                flags = [rf for _, _, rf in edge_series if isinstance(rf, bool)]
                denom = len(flags)
                if denom > 0:
                    numer = sum(1 for rf in flags if rf)
                    metrics["replay_rate"] = float(numer) / float(denom)
                else:
                    metrics["replay_rate"] = None

        if window.label in METRIC_WINDOWS["pe_to_dvalue_coupling"]:
            deltas = self._extract_value_deltas(value_events)
            pe_map = self._extract_state_pe(state_events)

            xs_pe: List[float] = []
            ys_dv: List[float] = []
            for ts_v, state_id, dv_norm in deltas:
                state_tuple = pe_map.get(state_id)
                if state_tuple is None:
                    continue
                ts_s, pe = state_tuple
                if self._state_join == "nearest_time":
                    max_lag = (
                        self._max_lag_ms
                        if self._max_lag_ms is not None
                        else max(1, window.window_ms // 10)
                    )
                    if abs(ts_s - ts_v) > max_lag:
                        continue
                xs_pe.append(pe)
                ys_dv.append(dv_norm)

            cov = self._cov(xs_pe, ys_dv)
            varx = self._variance(xs_pe)
            if cov is not None and varx is not None and varx > 0:
                metrics["pe_to_dvalue_coupling"] = cov / varx
            else:
                metrics["pe_to_dvalue_coupling"] = None

        for ev in state_events:
            pv = ev.payload or {}
            sid = pv.get("state_id")
            if isinstance(sid, str):
                state_ids.append(sid)

        if all(v is None for v in metrics.values()):
            return None

        return DerivedMetricsPayload(
            window_ms=window.window_ms,
            end_ts_unix_ms=end_ts_unix_ms,
            metrics=metrics,
            sources={
                "state_ids": state_ids,
                "event_hashes": event_hashes,
                "calc_version": self._calc_version,
            },
        )
