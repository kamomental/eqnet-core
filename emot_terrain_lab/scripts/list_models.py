# -*- coding: utf-8 -*-
"""LM Studio(OpenAI 互換サーバ)で利用可能なモデルを列挙する補助スクリプト。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.llm import get_llm, list_models


def main() -> None:
    llm = get_llm()
    models = list_models(llm.base_url, llm.api_key)
    if not models:
        print(
            "LM Studio サーバに接続できないか、モデルが未ダウンロードです。"
            "Local Server を ON にしてモデルを追加してください。"
        )
        return

    print("利用可能なモデル一覧:")
    for model in models:
        print(f" - {model}")
    if llm.model:
        print(f"\n現在の優先モデル: {llm.model}")
    else:
        print("\n現在の優先モデルは未選択です。")


if __name__ == "__main__":
    main()
