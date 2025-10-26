# -*- coding: utf-8 -*-
"""
擬似的な感情セッションログを生成するスクリプト。
各対話ごとに `terrain.emotion.extract_emotion` を通して
ベクトルを付与し、JSONL 形式で書き出す。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.emotion import extract_emotion

POSITIVE_DIALOGUES = [
    "今日は配信が大成功で、とても満足しています！",
    "新しい挑戦は緊張したけれど、終わってみれば楽しかった。",
    "視聴者と笑い合えて最高の一日になりました。",
]

NEGATIVE_DIALOGUES = [
    "少し失敗してしまい、落ち込んでいます。",
    "なんだか孤独を感じてしまった。",
    "確信が持てず不安な気持ちが続いている。",
]

NEUTRAL_DIALOGUES = [
    "淡々と作業をこなした一日でした。",
    "新しい資料に触れて、静かに勉強しました。",
    "少し分かったような気もするけれど、特別な感情はないです。",
]


def generate_dialogue() -> str:
    """重み付きでポジ/ネガ/ニュートラルから一文を選ぶ。"""
    bag = random.choices(
        [POSITIVE_DIALOGUES, NEGATIVE_DIALOGUES, NEUTRAL_DIALOGUES],
        weights=[0.4, 0.3, 0.3],
    )[0]
    sentence = random.choice(bag)
    if random.random() < 0.3:
        sentence += "!"
    return sentence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--per_week", type=int, default=4, help="1 週間あたりの対話数")
    parser.add_argument("--out", type=str, default="data/logs.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(2025, 9, 1)
    with out_path.open("w", encoding="utf-8") as f:
        for user_idx in range(args.users):
            user_id = f"user_{user_idx:03d}"
            for week_idx in range(args.weeks):
                for slot_idx in range(args.per_week):
                    dialogue = generate_dialogue()
                    emotion = extract_emotion(dialogue).tolist()
                    timestamp = (
                        start + timedelta(days=2 * slot_idx + week_idx * 7 + user_idx)
                    ).isoformat()
                    row = {
                        "session_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "timestamp": timestamp,
                        "dialogue": dialogue,
                        "emotion_vec": emotion,
                        "topic_tags": [],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
