from __future__ import annotations
from pathlib import Path

from heartos_mini import HeartOSConfig, HeartOSMini
from fallback_llm_client import FallbackLLMClient
from caching_llm_client import CachingLLMClient
from eqnet.voice.voice_field import VoiceField
from eqnet.voice.impl_provisional.prosody_rulebased_v0 import SimpleRuleBasedMapper
from eqnet.voice.impl_provisional.ma_planner_rulebased_v0 import RuleBasedMaPlannerV0
from eqnet.voice.impl_provisional.tts_dummy_v0 import DummyTTSClient


def main() -> None:
    base = Path("./eqnet_data")
    cfg = HeartOSConfig(
        diary_path=base / "diary.jsonl",
        memory_index_path=base / "memory_index.json",
    )

    llm = CachingLLMClient(FallbackLLMClient())
    voice_field = VoiceField(
        prosody_mapper=SimpleRuleBasedMapper(),
        ma_planner=RuleBasedMaPlannerV0(),
        tts=DummyTTSClient(),
    )

    heart = HeartOSMini(cfg, llm=llm, voice_field=voice_field)

    print("HeartOS mini + DummyVoice 起動（LM Studio or DummyLLM）。空行で終了します。")
    while True:
        try:
            user_text = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            break

        resp = heart.run_turn(user_text, partner_id="user_001", speak=True)
        print(f"EQNet> {resp}")
        print(f"  現在のQualia: {heart.qualia.as_dict()}")

    print("終了します。")


if __name__ == "__main__":
    main()
