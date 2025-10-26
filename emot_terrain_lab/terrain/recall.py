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

    def recall(self, query: str, current_emotion):
        cands = []
        for ep in self.ep.episodes:
            acc = self._accessibility(ep["emotion_pattern"]["center"], current_emotion)
            cands.append((acc, {"layer":"episodic","item":ep}))
        for pat in self.sm.patterns:
            acc = 0.7 * self._accessibility(pat["emotion_signature"], current_emotion)
            cands.append((acc, {"layer":"semantic","item":pat}))
        cands.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in cands[:5]]