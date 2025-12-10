# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Any
import re
from pathlib import Path
import json
import datetime

import numpy as np

from eqnet.voice.context import VoiceContext, TalkMode
from eqnet.voice.voice_field import VoiceField


@dataclass
class QualiaState:
    valence: float = 0.0
    arousal: float = 0.0
    stress: float = 0.0
    love: float = 0.0
    trust: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_np(self) -> np.ndarray:
        return np.array(
            [self.valence, self.arousal, self.stress, self.love, self.trust],
            dtype=float,
        )

    def clip(self) -> None:
        for k, v in self.__dict__.items():
            setattr(self, k, max(-1.0, min(1.0, float(v))))

    def update_from_features(self, features: dict[str, float], lr: float = 0.35) -> None:
        for k, delta in features.items():
            if hasattr(self, k):
                old = getattr(self, k)
                new = (1.0 - lr) * old + lr * delta
                setattr(self, k, new)
        self.clip()


@dataclass
class DiaryEntry:
    timestamp: str
    partner_id: str
    input_text: str
    response_text: str
    qualia_before: dict[str, float]
    qualia_after: dict[str, float]
    tags: list[str]


class DiaryWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: DiaryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


@dataclass
class MemoryIndex:
    counts: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> MemoryIndex:
        if not path.exists():
            return cls(counts={})
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(counts=data.get("counts", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"counts": self.counts}, ensure_ascii=False), encoding="utf-8")

    def update_from_tags(self, tags: list[str]) -> None:
        for t in tags:
            self.counts[t] = self.counts.get(t, 0) + 1


class MarkItDownClient:
    def parse(self, raw_text: str) -> dict[str, Any]:
        return {
            "raw": raw_text,
            "lines": [l for l in raw_text.splitlines() if l.strip()],
        }


POSITIVE_REGEXES = [
    r"嬉しい", r"うれしい", r"嬉しかった", r"うれしかった",
    r"楽しい", r"たのしい", r"楽しかった", r"たのしかった",
    r"よかった", r"助かった", r"ありがたい", r"ありがとう",
    r"感謝して(?:る|ます|います)?", r"好きです", r"大好き", r"幸せ"
]
NEGATIVE_REGEXES = [
    r"つらい", r"辛い", r"しんどい", r"不安", r"怖い", r"こわい",
    r"悲しい", r"かなしい", r"疲れた", r"つかれた",
    r"心配", r"しんぱい", r"つらかった", r"苦しい"
]

def _count_regex_matches(patterns: list[str], text: str) -> int:
    return sum(1 for pat in patterns if re.search(pat, text))

def infer_emotion_features(parsed: dict[str, Any]) -> dict[str, float]:
    text = parsed["raw"]
    pos = _count_regex_matches(POSITIVE_REGEXES, text)
    neg = _count_regex_matches(NEGATIVE_REGEXES, text)

    score = 0.0
    if pos + neg > 0:
        score = (pos - neg) / (pos + neg)

    features: dict[str, float] = {}
    features["valence"] = score
    features["love"] = max(0.0, score)
    features["stress"] = max(0.0, -score)
    features["arousal"] = min(1.0, abs(score) + 0.2)
    return features



class LLMClient:
    def chat(self, system_prompt: str, history: list[dict[str, str]], user_text: str) -> str:
        raise NotImplementedError


class DummyLLMClient(LLMClient):
    def chat(self, system_prompt: str, history: list[dict[str, str]], user_text: str) -> str:
        return f"（ダミー応答）「{user_text}」を受け取ったよ。今はHeartOS miniで試運転中。"


SOFT_PROMPT = """
あなたは EQNet HeartOS の「心の声」です。
相手はすこし疲れていたり、気持ちが揺れているかもしれません。

- 箇条書き／番号付きリストは、ユーザーが「整理して」「手順を教えて」とはっきり頼んだときだけ使う。
- それ以外は短い段落を 1?3 個で、口語のままやわらかく返事をする。
- 相手の言葉を一度受け止めてから、ゆっくり返す。
- 説教や「べき論」ではなく、「こういう選択肢もあるよ」と提案する形で伝える。
"""


@dataclass
class HeartOSConfig:
    diary_path: Path
    memory_index_path: Path
    system_prompt: str = SOFT_PROMPT


class HeartOSMini:
    def __init__(
        self,
        cfg: HeartOSConfig,
        llm: LLMClient | None = None,
        markitdown: MarkItDownClient | None = None,
        voice_field: VoiceField | None = None,
    ) -> None:
        self.cfg = cfg
        self.qualia = QualiaState()
        self.diary = DiaryWriter(cfg.diary_path)
        self.memory_index = MemoryIndex.load(cfg.memory_index_path)
        self.llm = llm or DummyLLMClient()
        self.markitdown = markitdown or MarkItDownClient()
        self.voice_field = voice_field
        self.history: list[dict[str, str]] = []

    def _now_iso(self) -> str:
        return datetime.datetime.now().isoformat()

    def _build_voice_context(self, partner_id: str) -> VoiceContext:
        q_vec = self.qualia.as_np()
        zeros = np.zeros_like(q_vec)
        return VoiceContext(
            q_t=q_vec,
            partner_id=partner_id,
            partner_profile=zeros,
            m_t=zeros,
            M_long=zeros,
            id_t=zeros,
            talk_mode=TalkMode.STREAM,
            extra={"voice_id": "default"},
        )

    def run_turn(self, user_text: str, partner_id: str = "default", speak: bool = False) -> str:
        parsed = self.markitdown.parse(user_text)
        qualia_before = self.qualia.as_dict()
        features = infer_emotion_features(parsed)
        self.qualia.update_from_features(features)
        qualia_after = self.qualia.as_dict()

        response = self.llm.chat(
            system_prompt=self.cfg.system_prompt,
            history=self.history,
            user_text=user_text,
        )

        tags = self._infer_tags(parsed, features)
        entry = DiaryEntry(
            timestamp=self._now_iso(),
            partner_id=partner_id,
            input_text=user_text,
            response_text=response,
            qualia_before=qualia_before,
            qualia_after=qualia_after,
            tags=tags,
        )
        self.diary.append(entry)

        self.memory_index.update_from_tags(tags)
        self.memory_index.save(self.cfg.memory_index_path)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        if speak and self.voice_field is not None:
            ctx = self._build_voice_context(partner_id)
            audio_bytes = self.voice_field.synthesize(response, ctx)
            out_path = Path("./eqnet_data") / "last_response.wav"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_bytes)
            except Exception:
                pass

        return response

    def _infer_tags(self, parsed: dict[str, Any], features: dict[str, float]) -> list[str]:
        tags: list[str] = []
        v = features.get("valence", 0.0)
        s = features.get("stress", 0.0)
        if v > 0.3:
            tags.append("positive")
        elif v < -0.3:
            tags.append("negative")
        if s > 0.4:
            tags.append("stressful")
        if not tags:
            tags.append("neutral")
        return tags


def main() -> None:
    base = Path("./eqnet_data")
    cfg = HeartOSConfig(
        diary_path=base / "diary.jsonl",
        memory_index_path=base / "memory_index.json",
    )
    heart = HeartOSMini(cfg, voice_field=None)

    print("HeartOS mini loop 起動。空行で終了します。")
    while True:
        try:
            user_text = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            break
        resp = heart.run_turn(user_text, partner_id="user_001", speak=False)
        print(f"EQNet> {resp}")
        print(f"  現在のQualia: {heart.qualia.as_dict()}")


if __name__ == "__main__":
    main()

