"""Micro-act templating and style guard for EQCore-first conversational flow."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import yaml

from emot_terrain_lab.rag.reweighter import reweight
from emot_terrain_lab.rag.source_trust import trust_score
from emot_terrain_lab.rag.explain import explain_selection

from emot_terrain_lab.eqcore.state import Affect, EmotionState, Stance
from emot_terrain_lab.nlg.reaction_planner import AffectSnapshot, ReactionPlanner

if TYPE_CHECKING:
    from emot_terrain_lab.rag.retriever import RetrievalHit


@dataclass
class ComposerConfig:
    """Configuration for loading templates and style guides."""

    templates_path: Union[str, Path] = Path("nlg/templates.yaml")
    style_path: Union[str, Path] = Path("nlg/style.yaml")
    reactions_path: Union[str, Path] = Path("config/nlg/reactions.yaml")
    default_persona: str = "soft_airhead"
    seed: Optional[int] = None


@dataclass
class ComposerContext:
    """Inputs required for rendering a micro-act."""

    stance: Union[str, Stance]
    micro_act: str
    user_utterance: str = ""
    present_focus: Optional[str] = None
    affect: Optional[Affect] = None
    mood: Optional[EmotionState] = None
    rag_hits: Sequence["RetrievalHit"] = field(default_factory=tuple)
    guide_option: Optional[str] = None
    safety_hint: Optional[str] = None
    request_narrator: bool = False
    seed: Optional[int] = None
    style: Optional[Dict[str, Any]] = None
    persona_tag: Optional[str] = None
    autopilot: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    domain: Optional[str] = None
    allow_reaction: bool = True


@dataclass
class ComposerOutput:
    """Rendered text bundle."""

    text: str
    stance: str
    micro_act: str
    template_key: str
    segments: Sequence[str]
    narrator: Optional[str]
    warnings: Sequence[str]
    context_used: Dict[str, Any] = field(default_factory=dict)


class _SafeFormatDict(dict):
    """Format map that returns empty strings for missing keys."""

    def __missing__(self, key: str) -> str:
        return ""


class TemplateBank:
    """Holds micro-act templates and helper phrases."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemplateBank":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Template file must be a mapping.")
        return cls(raw)

    def get_lines(self, stance: str, micro_act: str) -> Sequence[str]:
        stances = self._data.get("stances", {})
        block = stances.get(stance, {})
        entry = block.get(micro_act)
        if not entry:
            available = ", ".join(block.keys()) or "none"
            raise KeyError(f"Unknown micro_act '{micro_act}' for stance '{stance}' (available: {available})")
        lines = entry.get("lines")
        if not isinstance(lines, Iterable):
            raise ValueError(f"Template '{stance}.{micro_act}' must have iterable 'lines'.")
        return list(lines)

    def choose_opener(self, stance: str, rng: random.Random) -> str:
        pool = self._data.get("openers", {}).get(stance, [])
        if pool:
            return rng.choice(pool)
        return ""

    def choose_breath_prompt(self, rng: random.Random) -> str:
        prompts = self._data.get("breath_prompts", {})
        if not prompts:
            return ""
        key = rng.choice(list(prompts.keys()))
        return str(prompts[key])

    def choose_care_prompt(self, rng: random.Random) -> str:
        prompts = self._data.get("care_prompts", {})
        if not prompts:
            return ""
        key = rng.choice(list(prompts.keys()))
        return str(prompts[key])

    def choose_safety(self, rng: random.Random) -> str:
        prompts = self._data.get("safety_defaults", {})
        if not prompts:
            return ""
        key = rng.choice(list(prompts.keys()))
        return str(prompts[key])

    def narrator_lines(self) -> Sequence[str]:
        return list(self._data.get("narrator", {}).get("guide_summary", {}).get("lines", []))


class StyleGuide:
    """Style guard enforcing pause tokens, banned phrases, and cadence endings."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.banned_global = tuple(data.get("banned_global", []))
        self.banned_by_stance: Dict[str, Tuple[str, ...]] = {
            stance: tuple(words) for stance, words in data.get("banned_by_stance", {}).items()
        }
        self.endings: Dict[str, Tuple[str, ...]] = {
            stance: tuple(endings) for stance, endings in data.get("endings", {}).items()
        }
        self.pause_specs: Dict[str, Dict[str, Any]] = data.get("pauses", {})
        self.max_length = {stance: int(limit) for stance, limit in data.get("max_length", {}).items()}
        self.fallback_pause = str(data.get("fallback_pause", "…"))
        narrator = data.get("narrator", {})
        self.narrator_prefix = str(narrator.get("prefix", "ここからを整えます。"))
        self.narrator_agreement = str(narrator.get("agreement_line", "ここまででよさそうですか。"))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StyleGuide":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Style file must be a mapping.")
        return cls(raw)

    def enforce(self, stance: str, text: str) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        for phrase in self.banned_global:
            if phrase and phrase in text:
                warnings.append(f"banned phrase detected: '{phrase}'")
        for phrase in self.banned_by_stance.get(stance, ()):
            if phrase and phrase in text:
                warnings.append(f"{stance} style violation: '{phrase}'")

        text, pause_warns = self._ensure_pause(stance, text)
        warnings.extend(pause_warns)

        text, ending_warn = self._ensure_ending(stance, text)
        if ending_warn:
            warnings.append(ending_warn)

        limit = self.max_length.get(stance)
        if limit and len(text) > limit:
            warnings.append(f"{stance} text length {len(text)} exceeds {limit}")

        return text, warnings

    def _ensure_pause(self, stance: str, text: str) -> Tuple[str, List[str]]:
        spec = self.pause_specs.get(stance)
        if not spec:
            return text, []
        tokens = spec.get("tokens", [])
        min_count = int(spec.get("min_count", 1))
        total = sum(text.count(tok) for tok in tokens)
        if total >= min_count:
            return text, []
        adjusted = text.rstrip() + self.fallback_pause
        return adjusted, [f"{stance} pause inserted"]

    def _ensure_ending(self, stance: str, text: str) -> Tuple[str, Optional[str]]:
        endings = self.endings.get(stance)
        if not endings:
            return text, None
        if any(text.endswith(end) for end in endings):
            return text, None
        trimmed = text.rstrip("。…!\n 　")
        adjusted = trimmed + endings[0]
        return adjusted, f"{stance} ending adjusted to '{endings[0]}'"


class Composer:
    """Render micro-acts from EQCore state into text with style safety."""

    def __init__(self, config: ComposerConfig) -> None:
        self.config = config
        self.templates = TemplateBank.load(config.templates_path)
        self.style = StyleGuide.load(config.style_path)
        self._base_rng = random.Random(config.seed)
        self._reactions_cfg = self._load_reactions_cfg(Path(config.reactions_path))
        self._reaction_defaults = dict(self._reactions_cfg.get("defaults") or {})

    def compose(self, ctx: ComposerContext) -> ComposerOutput:
        stance = ctx.stance.mode if isinstance(ctx.stance, Stance) else str(ctx.stance)
        stance = stance.lower()
        rng = self._spawn_rng(ctx.seed)

        prepared_hits, rag_stats = self._prepare_rag_hits(ctx.rag_hits)
        lines = self.templates.get_lines(stance, ctx.micro_act)
        mapping = self._build_mapping(ctx, stance, rng, prepared_hits, rag_stats)
        formatted = [line.format_map(mapping) for line in lines]
        text = self._join_lines(stance, formatted)

        text, warnings = self.style.enforce(stance, text)

        reaction_meta = None
        if ctx.allow_reaction:
            text, reaction_meta = self._maybe_append_reaction(ctx, stance, text)

        narrator_text = self._make_narrator(ctx, stance, mapping) if ctx.request_narrator else None

        return ComposerOutput(
            text=text,
            stance=stance,
            micro_act=ctx.micro_act,
            template_key=f"{stance}.{ctx.micro_act}",
            segments=formatted,
            narrator=narrator_text,
            warnings=warnings,
            context_used={**dict(mapping), "rag_stats": rag_stats, "reaction": reaction_meta},
        )

    # ------------------------------------------------------------------ helpers

    def _spawn_rng(self, seed: Optional[int]) -> random.Random:
        if seed is not None:
            return random.Random(seed)
        # derive a child rng to avoid impacting global
        state = self._base_rng.getstate()
        child = random.Random()
        child.setstate(state)
        # advance base rng so next call differs
        self._base_rng.random()
        return child

    def _build_mapping(
        self,
        ctx: ComposerContext,
        stance: str,
        rng: random.Random,
        rag_hits: Sequence["RetrievalHit"],
        rag_stats: Dict[str, float],
    ) -> _SafeFormatDict:
        affect = ctx.affect
        mapping = _SafeFormatDict()
        mapping["opener"] = self.templates.choose_opener(stance, rng)
        mapping["present_frame"] = self._present_frame(ctx)
        mapping["pause"] = self.style.fallback_pause
        mapping["feeling"] = self._feeling_descriptor(affect)
        mapping["breath_hint"] = self._breath_hint(affect, rng)
        mapping["care_hint"] = self._care_hint(affect, rng)
        mapping["guide_option"] = self._guide_option(rag_hits, ctx.guide_option)
        mapping["safety_hint"] = ctx.safety_hint or self.templates.choose_safety(rng)
        mapping["mood_tone"] = self._mood_tone(ctx.mood)
        mapping["rag_cue"] = self._rag_cue(rag_hits)
        mapping["rag_summary"] = rag_stats.get("summary", "")
        mapping["prefix"] = self.style.narrator_prefix
        mapping["agreement_line"] = self.style.narrator_agreement
        return mapping

    def _load_reactions_cfg(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data or {}

    def _prepare_rag_hits(
        self, hits: Sequence["RetrievalHit"]
    ) -> Tuple[Sequence["RetrievalHit"], Dict[str, float]]:
        prepared = []
        trusts: List[float] = []
        junks: List[float] = []
        for hit in hits:
            meta = getattr(hit, "metadata", {}) or {}
            meta = dict(meta)
            trust = trust_score(meta)
            junk = float(meta.get("junk_prob", meta.get("junkiness", 0.0) or 0.0))
            weight = reweight(getattr(hit, "score", 1.0), junk_prob=junk, trust=trust)
            meta["trust_score"] = trust
            meta["junk_prob"] = junk
            meta["hygiene_weight"] = weight
            try:
                new_hit = replace(hit, metadata=meta, score=weight)
            except Exception:
                new_hit = hit
            prepared.append(new_hit)
            trusts.append(trust)
            junks.append(junk)
        summary = explain_selection(prepared, limit=2)
        stats = {
            "summary": summary,
            "avg_trust": sum(trusts) / len(trusts) if trusts else 0.0,
            "avg_junk_prob": sum(junks) / len(junks) if junks else 0.0,
        }
        return prepared, stats

    def _present_frame(self, ctx: ComposerContext) -> str:
        if ctx.present_focus:
            return ctx.present_focus
        snippet = ctx.user_utterance.strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if not snippet:
            return "いまの気持ち"
        max_len = 26
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip() + "…"
        return f"「{snippet}」の感じ"

    def _feeling_descriptor(self, affect: Optional[Affect]) -> str:
        if affect is None:
            return "いまの気持ち"
        val = affect.valence
        if val <= -0.45:
            return "重たさ"
        if val <= -0.1:
            return "揺らぎ"
        if val <= 0.25:
            return "余韻"
        if val <= 0.6:
            return "あたたかさ"
        return "広がり"

    def _breath_hint(self, affect: Optional[Affect], rng: random.Random) -> str:
        if affect is None:
            return self.templates.choose_breath_prompt(rng)
        if affect.arousal >= 0.7:
            return "息を数えて、ゆっくり五つ目で吐いてみましょう。"
        if affect.arousal <= 0.3:
            return "穏やかな息を続けて、胸の動きを静かに感じてみましょう。"
        return self.templates.choose_breath_prompt(rng)

    def _care_hint(self, affect: Optional[Affect], rng: random.Random) -> str:
        if affect is None:
            return self.templates.choose_care_prompt(rng)
        if affect.care >= 0.6:
            return "支えを頼る勇気"
        if affect.care <= 0.2:
            return "自分への小さな優しさ"
        return self.templates.choose_care_prompt(rng)

    def _guide_option(self, hits: Sequence["RetrievalHit"], guide_override: Optional[str]) -> str:
        if guide_override:
            return guide_override
        for hit in hits:
            suggestion = getattr(hit, "suggestion", None)
            if suggestion:
                return str(suggestion)
            numeric = getattr(hit, "numeric", None)
            if numeric and isinstance(numeric, dict):
                label = numeric.get("label") or numeric.get("attribute")
                value = numeric.get("value")
                unit = numeric.get("unit", "")
                direction = numeric.get("direction", "")
                if value is not None:
                    snippet = f"{label or '指標'}を{value}{unit}"
                    if direction:
                        snippet += f"に{direction}"
                    return snippet
        return "一番気になる地点を言葉にしてみること"

    def _rag_cue(self, hits: Sequence["RetrievalHit"]) -> str:
        if not hits:
            return ""
        top = hits[0]
        cue = getattr(top, "cue", None)
        if cue:
            return str(cue)
        text = getattr(top, "text", "")
        if not text:
            return ""
        snippet = re.sub(r"\s+", " ", str(text)).strip()
        max_len = 32
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip() + "…"
        return snippet

    def _mood_tone(self, mood: Optional[EmotionState]) -> str:
        if mood is None:
            return "ニュートラル"
        if mood.valence <= -0.4:
            return "陰"
        if mood.valence >= 0.4:
            return "陽"
        return "静"

    def _join_lines(self, stance: str, lines: Sequence[str]) -> str:
        if stance == "guide":
            return "\n".join(line.strip() for line in lines if line.strip())
        return " ".join(line.strip() for line in lines if line.strip())

    def _make_narrator(
        self,
        ctx: ComposerContext,
        stance: str,
        mapping: _SafeFormatDict,
    ) -> Optional[str]:
        if stance != "guide":
            return None
        lines = self.templates.narrator_lines()
        if not lines:
            return None
        formatted = [line.format_map(mapping) for line in lines]
        narrator_text = "\n".join(formatted)
        narrator_text, _ = self.style.enforce("guide", narrator_text)
        return narrator_text

    # ------------------------------------------------------------------ reactions

    def _maybe_append_reaction(self, ctx: ComposerContext, stance: str, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        reaction_bank = self._reactions_cfg.get("persona") or {}
        persona = self._resolve_persona(ctx, reaction_bank)
        if not persona:
            return text, None
        persona_cfg = reaction_bank.get(persona) or reaction_bank.get(self._reaction_defaults.get("persona", ""))
        if not persona_cfg:
            return text, None
        affect = self._affect_snapshot(ctx)
        heart = self._heartiness_from_ctx(ctx)
        seed_key = str(ctx.user_id or ctx.seed or persona)
        planner = ReactionPlanner(persona, persona_cfg, seed_key=seed_key)
        extras = {
            "user_snippet": self._user_snippet(ctx),
            "topic": ctx.present_focus or self._user_snippet(ctx),
            "stance": stance,
        }
        protected = bool("```" in text or text.strip().endswith("```"))
        choice = planner.plan(affect, heartiness=heart, extras=extras, protected=protected)
        if not choice or not choice.text.strip():
            return text, None
        updated = self._append_reaction_text(text, choice.text, heart)
        meta = {
            "persona": choice.persona,
            "zone": choice.zone,
            "band": choice.band,
            "family": choice.family,
        }
        return updated, meta

    def _resolve_persona(self, ctx: ComposerContext, bank: Dict[str, Any]) -> Optional[str]:
        if ctx.persona_tag:
            tag = ctx.persona_tag
        elif ctx.style and ctx.style.get("persona"):
            tag = str(ctx.style.get("persona"))
        else:
            tag = self.config.default_persona
        tag = str(tag or "").strip()
        if not tag:
            return None
        if bank and (tag not in bank):
            fallback = self._reaction_defaults.get("persona")
            if fallback and fallback in bank:
                return fallback
            if not bank:
                return None
        return tag

    def _affect_snapshot(self, ctx: ComposerContext) -> AffectSnapshot:
        valence = 0.0
        arousal = 0.0
        certainty = 0.7
        if ctx.mood is not None:
            valence = float(ctx.mood.valence)
            arousal = float(ctx.mood.arousal)
            certainty = float(ctx.mood.acceptance)
        elif ctx.affect is not None:
            valence = float(ctx.affect.valence)
            arousal = float(ctx.affect.arousal)
        care = float(getattr(ctx.affect, "care", 0.0) if ctx.affect else 0.0)
        novelty = float(getattr(ctx.affect, "novelty", 0.0) if ctx.affect else 0.0)
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        return AffectSnapshot(
            valence=valence,
            arousal=arousal,
            social=max(0.0, min(1.0, care)),
            novelty=max(0.0, min(1.0, novelty)),
            certainty=max(0.0, min(1.0, certainty)),
        )

    def _heartiness_from_ctx(self, ctx: ComposerContext) -> float:
        heart = 0.4
        autopilot = ctx.autopilot or {}
        if isinstance(autopilot, dict):
            if "mid" in autopilot and isinstance(autopilot["mid"], dict):
                heart = float(autopilot["mid"].get("heartiness", heart))
            else:
                heart = float(autopilot.get("heartiness", heart))
        return max(0.0, min(1.0, heart))

    def _user_snippet(self, ctx: ComposerContext) -> str:
        snippet = (ctx.user_utterance or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > 32:
            snippet = snippet[:32].rstrip() + "…"
        return snippet

    def _append_reaction_text(self, base_text: str, reaction_text: str, heartiness: float) -> str:
        if not reaction_text:
            return base_text
        pause_ms = int(200 + heartiness * 500)
        brk = f" [[BRK:{pause_ms}]]"
        if not base_text.strip():
            return reaction_text + brk
        if base_text.endswith("\n"):
            return base_text + reaction_text + brk
        return f"{base_text.rstrip()} {reaction_text}{brk}"
