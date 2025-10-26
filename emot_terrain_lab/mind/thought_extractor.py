# -*- coding: utf-8 -*-
"""Utilities for extracting shareable thought packets from internal state."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

Kind = Literal["hypothesis", "constraint", "uncertainty"]


@dataclass
class ThoughtPacket:
    """Portable representation of a pre-verbal thought."""

    id: str
    origin: str
    kind: Kind
    vec: List[float]
    entropy: float
    ttl_tau: float
    tags: List[str]
    created_tau: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hash(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha1(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
    return digest[:16]


class ThoughtExtractor:
    """Collapse replay and activation statistics into shareable packets."""

    def __init__(self, *, dim: int = 128, ttl_tau_default: float = 2.0) -> None:
        self.dim = max(1, int(dim))
        self.ttl_tau_default = float(ttl_tau_default)

    # --- public API ---------------------------------------------------------
    def extract(
        self,
        *,
        agent_id: str,
        tau_now: float,
        urk_trace: Optional[Dict[str, Any]],
        se_stats: Optional[Dict[str, Any]],
        policy: Optional[Dict[str, Any]] = None,
    ) -> List[ThoughtPacket]:
        policy = dict(policy or {})
        ttl_tau = float(policy.get("ttl_tau", self.ttl_tau_default))
        tags = list(policy.get("tags", []))
        packets: List[ThoughtPacket] = []

        # Hypothesis packet from replay kernel candidates
        cand_feats = self._candidate_features(urk_trace)
        if cand_feats:
            mean_vec = self._mean_vec(cand_feats)
            packets.append(
                ThoughtPacket(
                    id=_hash({"origin": agent_id, "kind": "hypothesis", "seed": mean_vec[0]}),
                    origin=agent_id,
                    kind="hypothesis",
                    vec=mean_vec,
                    entropy=self._candidate_entropy(urk_trace),
                    ttl_tau=ttl_tau,
                    tags=list(tags),
                    created_tau=float(tau_now),
                )
            )

        # Constraint packet from Saint Eryngium stats
        residual_feats = self._residual_features(se_stats)
        if residual_feats:
            packets.append(
                ThoughtPacket(
                    id=_hash({"origin": agent_id, "kind": "constraint", "seed": residual_feats[0]}),
                    origin=agent_id,
                    kind="constraint",
                    vec=residual_feats,
                    entropy=float(se_stats.get("uncertainty", 0.4)) if se_stats else 0.4,
                    ttl_tau=ttl_tau,
                    tags=list(tags),
                    created_tau=float(tau_now),
                )
            )

        # Uncertainty packet (ephemeral by default)
        uncertainty_vec = self._uncertainty_vec(urk_trace)
        if uncertainty_vec:
            uncertainty_tags = list(tags)
            if "ephemeral" not in uncertainty_tags:
                uncertainty_tags.append("ephemeral")
            packets.append(
                ThoughtPacket(
                    id=_hash({"origin": agent_id, "kind": "uncertainty", "seed": uncertainty_vec[0]}),
                    origin=agent_id,
                    kind="uncertainty",
                    vec=uncertainty_vec,
                    entropy=float(urk_trace.get("uncertainty_entropy", 0.6)) if urk_trace else 0.6,
                    ttl_tau=float(policy.get("ttl_tau_uncertainty", ttl_tau)),
                    tags=uncertainty_tags,
                    created_tau=float(tau_now),
                )
            )

        return packets

    # --- feature helpers ----------------------------------------------------
    def _candidate_features(self, urk_trace: Optional[Dict[str, Any]]) -> List[List[float]]:
        if not urk_trace:
            return []
        cands: Iterable[Dict[str, Any]] = urk_trace.get("candidates") or []
        feats: List[List[float]] = []
        for cand in cands:
            vec = self._vectorise_candidate(cand)
            if vec:
                feats.append(vec)
            if len(feats) >= 3:
                break
        return feats

    def _vectorise_candidate(self, cand: Dict[str, Any]) -> List[float]:
        if not isinstance(cand, dict):
            return []
        base = [
            float(cand.get("U", 0.0)),
            float(cand.get("coherence", 0.0)),
        ]
        summary = cand.get("summary", {}) or {}
        for key in sorted(summary.keys()):
            try:
                base.append(float(summary[key]))
            except Exception:
                continue
        # Additional fallback: metadata counts
        meta = cand.get("metadata", {}) or {}
        if meta:
            for key in sorted(meta.keys()):
                value = meta[key]
                try:
                    base.append(float(value))
                except Exception:
                    continue
        return self._project(base)

    def _residual_features(self, se_stats: Optional[Dict[str, Any]]) -> List[float]:
        if not se_stats:
            return []
        residuals = se_stats.get("residual_moments")
        if isinstance(residuals, list) and residuals:
            cleaned = [float(x) for x in residuals[: self.dim]]
            return self._project(cleaned)
        if "variance" in se_stats:
            vec = [float(se_stats.get("variance", 0.0)), float(se_stats.get("skew", 0.0))]
            return self._project(vec)
        return []

    def _uncertainty_vec(self, urk_trace: Optional[Dict[str, Any]]) -> List[float]:
        if not urk_trace:
            return []
        map_ = urk_trace.get("uncertainty_map") or {}
        pc1 = map_.get("pc1")
        if isinstance(pc1, list) and pc1:
            try:
                vector = [float(x) for x in pc1[: self.dim]]
            except Exception:
                vector = []
        else:
            # fallback: encode spread of candidate utilities
            cands = urk_trace.get("candidates") or []
            utilities = [float(c.get("U", 0.0)) for c in cands]
            if not utilities:
                return []
            mean = sum(utilities) / len(utilities)
            var = sum((u - mean) ** 2 for u in utilities) / max(1, len(utilities))
            vector = [mean, math.sqrt(max(var, 0.0))]
        return self._project(vector)

    def _candidate_entropy(self, urk_trace: Optional[Dict[str, Any]]) -> float:
        if not urk_trace:
            return 0.3
        entropy = urk_trace.get("entropy")
        if entropy is not None:
            try:
                return float(entropy)
            except Exception:
                pass
        cands = urk_trace.get("candidates") or []
        utilities = [float(c.get("U", 0.0)) for c in cands]
        if not utilities:
            return 0.3
        mean = sum(utilities) / len(utilities)
        var = sum((u - mean) ** 2 for u in utilities) / max(1, len(utilities))
        return max(0.0, min(1.0, math.sqrt(var)))

    def _mean_vec(self, feats: Iterable[List[float]]) -> List[float]:
        feats = list(feats)
        if not feats:
            return [0.0] * self.dim
        length = len(feats[0])
        sums = [0.0] * length
        for vec in feats:
            for idx, value in enumerate(vec):
                sums[idx] += value
        mean = [value / len(feats) for value in sums]
        return self._project(mean)

    def _project(self, values: Iterable[float]) -> List[float]:
        vec = [float(v) for v in values]
        if len(vec) >= self.dim:
            return vec[: self.dim]
        return vec + [0.0] * (self.dim - len(vec))


__all__ = ["ThoughtExtractor", "ThoughtPacket"]

