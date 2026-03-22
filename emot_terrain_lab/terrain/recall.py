# -*- coding: utf-8 -*-
import numpy as np

class RecallEngine:
    def __init__(self, episodic, semantic, terrain):
        self.ep = episodic
        self.sm = semantic
        self.terrain = terrain

    def _accessibility(self, center, current):
        d = np.linalg.norm(np.array(center)-np.array(current))
        return float(np.exp(-d))

    def _semantic_replay_bias(self, pattern, query: str):
        signature = pattern.get("working_memory_replay_signature") or {}
        if not isinstance(signature, dict):
            return 0.0
        strength = float(signature.get("strength") or 0.0)
        if strength <= 0.0:
            return 0.0
        focus = str(signature.get("focus") or "").strip().lower()
        anchor = str(signature.get("anchor") or "").strip().lower()
        text = " ".join(
            filter(
                None,
                [
                    str(query or "").strip().lower(),
                    str(pattern.get("abstract_description") or "").strip().lower(),
                ],
            )
        )
        if not text:
            return 0.0
        match = 0.0
        if focus and focus in text:
            match += 0.6
        if anchor and anchor in text:
            match += 0.8
        return min(0.18, strength * match * 0.12)

    def _semantic_recurrence_bias(self, pattern):
        occurrences = float(pattern.get("occurrences") or 0.0)
        recurrence_weight = float(pattern.get("recurrence_weight") or occurrences or 0.0)
        if recurrence_weight <= occurrences or occurrences <= 0.0:
            return 0.0
        reinforcement = min(1.0, (recurrence_weight - occurrences) / max(occurrences, 1.0))
        return min(0.08, reinforcement * 0.08)

    def recall(self, query: str, current_emotion):
        cands = []
        for ep in self.ep.episodes:
            acc = self._accessibility(ep["emotion_pattern"]["center"], current_emotion)
            cands.append((acc, {"layer":"episodic","item":ep}))
        for pat in self.sm.patterns:
            acc = 0.7 * self._accessibility(pat["emotion_signature"], current_emotion)
            acc += self._semantic_replay_bias(pat, query)
            acc += self._semantic_recurrence_bias(pat)
            cands.append((acc, {"layer":"semantic","item":pat}))
        cands.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in cands[:5]]
