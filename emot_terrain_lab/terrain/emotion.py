# -*- coding: utf-8 -*-
"""
対話テキストから 9 次元の情動・意識状態ベクトルを推定する。
軸は以下の通り:
    0. sensory   : 感覚的鮮明さ（島皮質・身体感覚を想定）
    1. temporal  : 時間意識（記憶／未来期待）
    2. spatial   : 自他・内外の位置関係
    3. affective : 快不快・情動価
    4. cognitive : 認知的意味づけ／自己語り
    5. social    : 対人的配慮・共感
    6. meta      : 俯瞰・象徴・夢想（DMN）
    7. agency    : 能動性・エンパワメント志向
    8. recursion : 再起／自己参照制御の強さ

LM Studio(OpenAI 互換) が利用可能なら LLM 推定を用い、
未接続時は簡易ヒューリスティクスにフォールバックする。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .llm import chat_json

AXES = [
    "sensory",
    "temporal",
    "spatial",
    "affective",
    "cognitive",
    "social",
    "meta",
    "agency",
    "recursion",
]

AXIS_BOUNDS = {
    "sensory": (0.0, 1.0),
    "temporal": (0.0, 1.0),
    "spatial": (0.0, 1.0),
    "affective": (-1.0, 1.0),
    "cognitive": (0.0, 1.0),
    "social": (-1.0, 1.0),
    "meta": (0.0, 1.0),
    "agency": (-1.0, 1.0),
    "recursion": (0.0, 1.0),
}


def _mock(dialogue: str) -> np.ndarray:
    """LLM が使えない場合に備えた簡易ヒューリスティクス。"""
    lower = dialogue.lower()

    sensory = 0.5 + 0.1 * sum(lower.count(token) for token in ("feel", "touch", "香り", "匂い", "温かい"))
    temporal = 0.3 + 0.1 * sum(lower.count(token) for token in ("過去", "未来", "いつか", "思い出", "plan"))
    spatial = 0.2 + 0.1 * sum(lower.count(token) for token in ("場所", "距離", "近い", "遠い", "内側", "外側"))

    positive = sum(lower.count(token) for token in ("happy", "楽しい", "嬉しい", "美しい"))
    negative = sum(lower.count(token) for token in ("sad", "悲しい", "辛い", "怖い", "恐怖"))
    affective = 0.1 * (positive - negative)

    cognitive = 0.3 + 0.1 * sum(lower.count(token) for token in ("考え", "理解", "意味", "物語", "自己"))
    social = 0.2 + 0.1 * sum(lower.count(token) for token in ("一緒", "共有", "あなた", "仲間", "関係"))
    meta = 0.2 + 0.1 * sum(lower.count(token) for token in ("夢", "象徴", "俯瞰", "瞑想", "空想"))

    agency = 0.1 if any(word in dialogue for word in ("やってみる", "挑戦", "動く", "前に進む", "作る")) else -0.05
    recursion = 0.1 * (dialogue.count("「") + dialogue.count("」") + lower.count("自分") + lower.count("内省"))

    values = [sensory, temporal, spatial, affective, cognitive, social, meta, agency, recursion]
    vec = np.array(
        [float(np.clip(val, *AXIS_BOUNDS[axis])) for val, axis in zip(values, AXES)],
        dtype=float,
    )
    return vec


def extract_emotion(dialogue: str, temperature: float = 0.2) -> np.ndarray:
    """
    対話文字列を解析して感情ベクトルを返す。
    USE_LLM=0 のとき、または LLM 接続に失敗した場合はヒューリスティクスを利用。
    """
    use_llm = os.getenv("USE_LLM", "0") != "0"
    if not use_llm:
        return _mock(dialogue)

    axis_spec = ", ".join(
        f"{name}(-1..1)" if name in {"affective", "social", "agency"} else f"{name}(0..1)"
        for name in AXES
    )
    system_prompt = (
        "あなたは対話から 9 次元の情動・意識状態を推定するアナライザーです。"
        "以下のキーについて指定範囲に正規化した数値のみを JSON で返してください。"
        f"{{{axis_spec}}}"
    )
    user_prompt = f"発話: {dialogue}\nJSON で返答してください。"

    payload: Optional[dict] = chat_json(system_prompt, user_prompt, temperature=temperature)
    if not payload:
        return _mock(dialogue)

    clipped = [
        float(np.clip(float(payload.get(axis, 0.0)), *AXIS_BOUNDS[axis]))
        for axis in AXES
    ]
    return np.array(clipped, dtype=float)
