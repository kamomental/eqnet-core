# -*- coding: utf-8 -*-
"""Thought communication policy and delivery helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from emot_terrain_lab.safety.acl import accept_kind, max_per_turn


@dataclass
class DeliverLog:
    frm: str
    to: str
    kind: str
    gain: float
    ttl_tau: float
    tags: List[str]
    packet_id: str

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["gain"] = float(self.gain)
        data["ttl_tau"] = float(self.ttl_tau)
        return data


class ThoughtBus:
    """Decision layer for sharing thought packets across agents."""

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.cfg = dict(config or {})
        self._recent: Dict[str, float] = {}
        self._last_tx_count: int = 0
        self._gain_cap = float(self.cfg.get("gain_max", 0.1))
        self._last_rejects: List[Dict[str, object]] = []

    def enabled(self) -> bool:
        return bool(self.cfg.get("enable", True))

    # --- policy -------------------------------------------------------------
    def policy_gate(
        self,
        *,
        mode: str,
        risk_p: float,
        read_only: bool,
        tau_rate: float,
        inflammation: float,
        synchrony: Optional[float],
        assoc_defect: float,
        naturality_residual: float,
        avg_entropy: float,
        junk_prob: float,
        tx_count_last: int,
    ) -> Dict[str, object]:
        if not self.enabled():
            return {"allow": False, "gain": 0.0, "reason": "disabled"}
        if read_only:
            return {"allow": False, "gain": 0.0, "reason": "read_only"}
        risk_enter = float(self.cfg.get("risk_enter", 0.4))
        if risk_p >= risk_enter:
            return {"allow": False, "gain": 0.0, "reason": "risk"}
        if synchrony is not None and synchrony > float(self.cfg.get("r_max", 0.78)):
            return {"allow": False, "gain": 0.0, "reason": "over_sync"}
        if assoc_defect > float(self.cfg.get("assoc_defect_th", 0.15)):
            return {"allow": False, "gain": 0.0, "reason": "assoc_defect"}
        if naturality_residual > float(self.cfg.get("naturality_th", 0.25)):
            return {"allow": False, "gain": 0.0, "reason": "naturality"}
        if avg_entropy > float(self.cfg.get("entropy_th", 0.75)):
            return {"allow": False, "gain": 0.0, "reason": "entropy_high"}
        if junk_prob > float(self.cfg.get("junk_th", 0.4)):
            return {"allow": False, "gain": 0.0, "reason": "junk_high"}
        if tx_count_last >= int(self.cfg.get("rate_limit_per_turn", 6)):
            return {"allow": False, "gain": 0.0, "reason": "rate_limited"}
        base_gain = float(self.cfg.get("gain_max", 0.1))
        tau_scale = max(0.6, min(1.2, tau_rate))
        infl_scale = max(0.2, 1.0 - 0.5 * float(inflammation))
        gain = base_gain * tau_scale * infl_scale
        cap = max(0.0, min(self._gain_cap, base_gain))
        gain = max(0.0, min(cap, gain))
        return {"allow": True, "gain": gain, "reason": "ok", "mode": mode}

    # --- delivery -----------------------------------------------------------
    def deliver(
        self,
        *,
        me: str,
        peers: Iterable[str],
        packets: Iterable[Dict[str, object]],
        gate: Dict[str, object],
        tau_now: float,
        cooldown_tau: float = 0.8,
        acl: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DeliverLog]:
        if not gate.get("allow"):
            self._last_tx_count = 0
            return []
        allowed = set(self.cfg.get("channels", ["hypothesis", "constraint", "uncertainty"]))
        base_gain = float(gate.get("gain", 0.0))
        logs: List[DeliverLog] = []
        per_channel_cfg = self.cfg.get("per_channel", {}) or {}
        packets_list = list(packets)
        self._last_rejects = []
        peer_counts: Dict[str, int] = {}
        for peer in peers:
            if peer == me:
                continue
            for packet in packets_list:
                kind = str(packet.get("kind", ""))
                if kind not in allowed:
                    continue
                if acl is not None and not accept_kind(acl, peer, kind):
                    self._last_rejects.append({"frm": me, "to": peer, "kind": kind, "reason": "acl"})
                    continue
                limit = max_per_turn(acl, peer) if acl is not None else None
                if limit is not None and peer_counts.get(peer, 0) >= limit:
                    self._last_rejects.append({"frm": me, "to": peer, "kind": kind, "reason": "acl_quota"})
                    continue
                packet_id = str(packet.get("id", ""))
                if tau_now - self._recent.get(packet_id, -1e9) < cooldown_tau:
                    continue
                ttl_default = float(self.cfg.get("ttl_tau_default", 2.0))
                ttl_tau = float(packet.get("ttl_tau", ttl_default))
                if kind in per_channel_cfg:
                    ttl_tau = float(per_channel_cfg[kind].get("ttl_tau", ttl_tau))
                ttl_tau = max(0.1, ttl_tau)
                tags_raw = packet.get("tags") or []
                tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else [str(tags_raw)]
                gain = base_gain
                if kind in per_channel_cfg:
                    gain = min(gain, float(per_channel_cfg[kind].get("gain", gain)))
                logs.append(
                    DeliverLog(
                        frm=me,
                        to=str(peer),
                        kind=kind,
                        gain=gain,
                        ttl_tau=ttl_tau,
                        tags=tags,
                        packet_id=packet_id,
                    )
                )
                self._recent[packet_id] = float(tau_now)
                peer_counts[peer] = peer_counts.get(peer, 0) + 1
        self._last_tx_count = len(logs)
        return logs

    def last_tx_count(self) -> int:
        return self._last_tx_count

    def set_gain_cap(self, cap: float) -> None:
        try:
            self._gain_cap = max(0.0, float(cap))
        except Exception:
            pass

    def last_rejects(self) -> List[Dict[str, object]]:
        return list(self._last_rejects)


__all__ = ["ThoughtBus", "DeliverLog"]
