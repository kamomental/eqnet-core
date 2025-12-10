from __future__ import annotations
from ..ma_planner import MaPlanner, MaPlan
from ..context import VoiceContext, TalkMode


class RuleBasedMaPlannerV0(MaPlanner):
    """Voice Field §2 / §7 MaPlanner v0 (provisional)."""

    def plan(self, ctx: VoiceContext) -> MaPlan:
        q_t = ctx.q_t
        arousal = float(q_t[1])

        if ctx.talk_mode == TalkMode.STREAM:
            base_silence = 250
        elif ctx.talk_mode == TalkMode.INTIMATE:
            base_silence = 800
        elif ctx.talk_mode == TalkMode.PRESENCE:
            base_silence = 2000
        else:
            base_silence = 500

        silence_ms = int(base_silence * (1.0 - 0.3 * arousal))
        insert_filler = ctx.talk_mode == TalkMode.STREAM and arousal < 0.7
        filler_text = "えっと" if insert_filler else None

        return MaPlan(
            silence_ms=max(100, silence_ms),
            insert_filler=insert_filler,
            filler_text=filler_text,
        )
