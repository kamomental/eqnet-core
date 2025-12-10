from __future__ import annotations
from dataclasses import dataclass
from .context import VoiceContext, ProsodyVec
from .prosody_mapper import EmotionProsodyMapper
from .ma_planner import MaPlanner, MaPlan


class TTSClient:
    """Abstract TTS client."""

    def speak(self, text: str, prosody: ProsodyVec) -> bytes:
        raise NotImplementedError


@dataclass
class VoiceField:
    prosody_mapper: EmotionProsodyMapper
    ma_planner: MaPlanner
    tts: TTSClient

    def synthesize(self, text: str, ctx: VoiceContext) -> bytes:
        ma_plan: MaPlan = self.ma_planner.plan(ctx)
        voice_id = ctx.extra.get("voice_id", "default") if ctx.extra else "default"
        prosody: ProsodyVec = self.prosody_mapper.map(ctx.q_t, voice_id)

        final_text = text
        if ma_plan.insert_filler and ma_plan.filler_text:
            final_text = f"{ma_plan.filler_text}、{text}"

        return self.tts.speak(final_text, prosody)
